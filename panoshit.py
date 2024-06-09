import re
import shutil
import subprocess
import sys
import time
import os
import traceback
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from io import TextIOBase, StringIO
from pathlib import Path
from random import Random
from threading import Thread
from typing import List, Optional, Iterable
from PIL import Image as PilImage, UnidentifiedImageError

MAX_BLEND_PIXELS = 100
MIN_HORIZ_INC = 30
VERT_INC = 15

class ControlPointArrangementStrategy(Enum):
    EQUAL = 0
    AUTO = 1
    RANDOM = 2
    ROUGH = 2

class Lens(Enum):
    RECTILINEAR = 0
    PANORAMIC_CYLINDRICAL = 1
    CIRCULAR_FISHEYE = 2
    FULL_FRAME_FISHEYE = 3
    EQUIRECTANGULAR = 4
    ORTHOGRAPHIC = 8
    STEREOGRAPHIC = 10
    FISHEYE_THOBY = 20
    EQUISOLID = 21

ALL_LENSES = [l for l in Lens]

class Projection(Enum):
    RECTILINEAR = 0
    CYLINDRICAL = 1
    EQUIRECTANGULAR = 2
    FISHEYE = 3
    STEREOGRAPHIC = 4
    MERCATOR = 5
    TRANSMERCATOR = 6
    SINUSOIDAL = 7
    LAMBERT_CYLINDRICAL_EQUAL_AREA = 8
    LAMBERT_EQUAL_AREA_AZIMUTHAL = 9
    ALBERS_EQUAL_AREA_CONIC = 10
    MILLER_CYLINDRICAL = 11
    PANINI = 12
    ARCHITECTURAL = 13
    ORTHOGRAPHIC = 14
    EQUISOLID = 15
    EQUIRECTANGULAR_PANINI = 16
    BIPLANE = 17
    TRIPLANE = 18
    PANINI_GENERAL = 19
    THOBY_PROJECTION = 20
    HAMMER_AITOFF_EQUAL_AREA = 21

ALL_PROJECTIONS = [p for p in Projection]

class PanoImage:
    #SUPPORTED_EXTENSIONS = ["jpg", ".jpeg", "png", "tif", "tiff"] # I can't tell what formats hugin supports so uh let's just use these
    index: int
    width: int
    height: int
    p: float
    y: float
    image: PilImage.Image
    path: str

    def __init__(self, path: str, index: int, pil: Optional[PilImage.Image] = None):
        self.path = path
        self.index = index
        if pil:
            self.image = pil # if this does not actually correspond to path, that's on you
        else:
            self.image = PilImage.open(path)
        self.width = self.image.size[0]
        self.height = self.image.size[1]

    def set_index(self, index: int):
        self.index = index

    def align(self, vertAngle: float, horizAngle: float):
        self.p = vertAngle
        self.y = horizAngle

    def copy(self) -> "PanoImage":
        return PanoImage(self.path, self.index, self.image)

@dataclass
class ControlPoint:
    left_index: int
    right_index: int
    left_x: float
    left_y: float
    right_x: float
    right_y: float

    def __str__(self):
        return f"c n{self.left_index} N{self.right_index} x{self.left_x} y{self.left_y} X{self.right_x} Y{self.right_y} t0"


class Logger():
    def __init__(self, stream: TextIOBase, print: bool = False):
        self.stream = stream
        self.print = print
        self.start = datetime.now()
        start_timestamp = self.start.strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"Log started at {start_timestamp}")

    def log(self, msg: str, newline: bool = True):
        timestamp = time.time()
        ending = '\n' if newline else ''
        if self.print:
            print(msg)
        self.stream.write(f"[{timestamp}] {msg}{ending}")

    def close(self):
        end = datetime.now()
        end_timestamp = end.strftime("%Y-%m-%d %H:%M:%S")
        difference = end - self.start

        self.log(f"Log ended at {end_timestamp}")
        hours = difference.total_seconds() // 3600
        minutes = (difference.total_seconds() - hours*3600) // 60
        seconds = difference.total_seconds() - hours*3600 - minutes*60
        self.log(f"Run time {hours}:{minutes}:{seconds}")
        self.stream.close()

def read_directory(path: str) -> List[str]:
    images: List[str] = list()
    for f in os.listdir(path):
        f_path = os.path.join(path, f)
        if os.path.isfile(f_path):
            try:
                image = PilImage.open(f_path) # TODO this is going to cost something but I'm on a bus
                images.append(f_path)
                print(f)
            except UnidentifiedImageError as e:
                pass
    if len(images) == 0:
        raise FileNotFoundError(f"no supported images found in {path}")
    images.sort()
    return images

def run_cmd(command: str, log: Logger):
    log.log("$ " + command, newline=False)
    args = command.split(" ")

    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1,  universal_newlines=True) as p:
        try:
            for line in p.stdout:
                log.log(line, newline=False)
            if p.returncode != 0:
                log.log("error jc!")
            stderr = p.stderr.read()
            log.log(stderr)
        except UnicodeDecodeError as e:
            print("could not log: ")
            print(line)
            print(stderr)

class Pano:
    control_points: List[ControlPoint]
    images: List[PanoImage]

    def __init__(self,
                 image_paths: List[str],
                 width: int,
                 strategy: ControlPointArrangementStrategy,
                 points_per_pair: int,
                 pixel_width: int,
                 pixel_height: int,
                 lens_fov: Optional[float] = 0,
                 fov: Optional[float] = 179,
                 vfov: Optional[float] = 140,
                 batch_id: Optional[int] = None,
                 #use_static_seed: bool = False,
                 static_seed: Optional[int] = None,):
        now = int(time.time() * 1000000)
        if batch_id:
            self.batch_id = batch_id
        else:
            self.batch_id = now
        if static_seed is not None:
            self.seed = static_seed
        else:
            self.seed = now
        if lens_fov != 0:
            self.lens_fov = lens_fov
        else:
            self.lens_fov = fov / width
        self.strategy = strategy
        self.points_per_pair = points_per_pair
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.fov = fov
        self.vfov = vfov
        self.random = Random(self.seed)
        self.width = width
        self.images: List[PanoImage] = list()
        for i, path in enumerate(image_paths):
            self.images.append(PanoImage(path, i))
        self.height = (len(self.images) // width) + (0 if len(self.images) % width == 0 else 1)
        self.control_points = None
        self._generate_grid()

    def _align(self):
        # precondition: grid exists
        # run this after every grid reshuffle
        print("aligning")
        #max_vert = 140
        max_vert = self.vfov
        #vert_increment = VERT_INC
        aspect_ratio = self.images[0].width / self.images[0].height
        vert_increment = (self.lens_fov / aspect_ratio) - 3

        total_vert = min(vert_increment * self.height * 1.0, max_vert)

        #vert_increment = (self.vfov / self.height) - 3
        #vert_increment = self.lens_fov - 1
        #total_vert = min(vert_increment * self.height, self.vfov)
        #total_vert = self.vfov
        #horiz_increment = min(self.fov / self.width, MIN_HORIZ_INC)
        horiz_increment = self.lens_fov - 3
        total_horiz = min(self.fov, horiz_increment * self.width)
        #vert_increment = total_vert / self.height
        northernmost = (total_vert - vert_increment) / 2
        easternmost = (total_horiz - horiz_increment) / 2
        for i, row in enumerate(self.grid):
            vert = northernmost - i*vert_increment
            for j, image in enumerate(row):
                horiz = (-1 * easternmost) + j*horiz_increment
                image.align(vert, horiz)

    def _generate_grid(self):
        print("generating grid")
        self.grid = list()
        for i in range(self.height):
            row = list()
            for j in range(self.width):
                k = j + i*self.width
                if k >= len(self.images):
                    break
                row.append(self.images[k])
            self.grid.append(row)
        self._align()

    def shuffle_images(self):
        self.random.shuffle(self.images)
        for i, image in enumerate(self.images):
            image.set_index(i)
        self._generate_grid()
        self._create_control_points()


    def _create_vertical_cp(self, top: PanoImage, bottom: PanoImage) -> Iterable[ControlPoint]:
        result = list()
        sections = self.points_per_pair + 1
        if self.strategy == ControlPointArrangementStrategy.EQUAL:
            top_x_inc = top.width / sections
            top_y = min(top.height - (top.height / sections), top.height - MAX_BLEND_PIXELS)
            bottom_x_inc = bottom.width / sections
            bottom_y = min(bottom.height / sections, MAX_BLEND_PIXELS)
            for i in range(self.points_per_pair):
                result.append(ControlPoint(top.index, bottom.index, top_x_inc*(i+1), top_y, bottom_x_inc*(i+1), bottom_y))
        elif self.strategy == ControlPointArrangementStrategy.RANDOM:
            top_y_bound = min(top.height - (top.height / sections), top.height - MAX_BLEND_PIXELS)
            bottom_y_bound = min(bottom.height / sections, MAX_BLEND_PIXELS)
            for i in range(self.points_per_pair):
                top_x = self.random.randint(0, top.width)
                bottom_x = self.random.randint(0, bottom.width)
                top_y = self.random.randint(top_y_bound, top.height)
                bottom_y = self.random.randint(0, bottom_y_bound)
                result.append(ControlPoint(top.index, bottom.index, top_x, top_y, bottom_x, bottom_y))
        elif self.strategy == ControlPointArrangementStrategy.ROUGH:
            raise NotImplementedError()
        else:
            raise ValueError()
        return result

    def _create_horizontal_cp(self, left: PanoImage, right: PanoImage) -> Iterable[ControlPoint]:
        result = list()
        sections = self.points_per_pair + 1
        if self.strategy == ControlPointArrangementStrategy.EQUAL:
            left_x = min(left.width - (left.width / sections), left.width - MAX_BLEND_PIXELS)
            left_y_inc = left.height / sections
            right_x = min(right.width / sections, MAX_BLEND_PIXELS)
            right_y_inc = right.height / sections
            for i in range(self.points_per_pair):
                result.append(ControlPoint(left.index, right.index, left_x, left_y_inc*(i+1), right_x, right_y_inc*(i+1)))
        elif self.strategy == ControlPointArrangementStrategy.RANDOM:
            left_x_bound = min(left.width - (left.width / sections), left.width - MAX_BLEND_PIXELS)
            right_x_bound = min(right.width / sections, MAX_BLEND_PIXELS)
            for i in range(self.points_per_pair):
                left_x = self.random.randint(left_x_bound, left.width)
                right_x = self.random.randint(0, right_x_bound)
                left_y = self.random.randint(0, left.height)
                right_y = self.random.randint(0, right.height)
                result.append(ControlPoint(left.index, right.index, left_x, left_y, right_x,right_y))
        elif self.strategy == ControlPointArrangementStrategy.ROUGH:
            raise NotImplementedError()
        else:
            raise ValueError()
        return result

    def _create_control_points(self):
        self.control_points = list()
        for i, row in enumerate(self.grid):
            for j, image in enumerate(row):
                if i > 0:
                    top = self.grid[i - 1][j]
                    cps = self._create_vertical_cp(top, image)
                    self.control_points.extend(cps)
                if j > 0:
                    left = self.grid[i][j - 1]
                    cps = self._create_horizontal_cp(left, image)
                    self.control_points.extend(cps)

    def generate(self,
                 outdir_path: Optional[str],
                 lenses: Iterable[Lens] = ALL_LENSES,
                 projections: Iterable[Projection] = ALL_PROJECTIONS,
                 threads: int = 4) -> str:
            if self.control_points is None:
                self._create_control_points()
            if outdir_path:
                proj_directory = os.path.join(outdir_path, str(self.batch_id))
            else:
                proj_directory = os.path.join(os.getcwd(), str(self.batch_id))
            if not os.path.exists(proj_directory):
                Path(proj_directory).mkdir(parents=True, exist_ok=True)
            #os.chdir(proj_directory)
            file_args = " ".join(f.path for f in self.images)
            #threads = []
            log_path = os.path.join(proj_directory, "log.txt")
            log = Logger(open(log_path, 'w'), print=True)
            future_results = []
            with ThreadPoolExecutor(max_workers=threads) as thread_pool:
                # TODO use hsi?
                for l in lenses:
                    outfile1 = os.path.join(proj_directory, f"pano_{self.batch_id}_s{self.seed}_cp{self.strategy.name}_l{l.name}_lf{self.lens_fov}.pto")
                    command = f"pto_gen -o {outfile1} -f {self.lens_fov} -p {l.value} {file_args}"
                    run_cmd(command, log)
                    for p in projections:
                        log.log(f"submitting job ({l.name}, {p.name})")
                        future_results.append(thread_pool.submit(self.run, proj_directory, l, p, outfile1, threads))
                for future in future_results:
                    try:
                        log.log(f"Log: {future.result()}")
                    except Exception as e:
                        log.log(f"Exception: {traceback.format_exception(e)}")
            log.close()
            return log_path
            #        t = Thread(target=self.run, args=[proj_directory, l, p, outfile1])
            #        t.start()
            #        threads.append(t)
            #for t in threads:
            #    t.join()

    def get_optimized_file_list(self, pto_path: str) -> List[str]:
        files = []
        pattern = re.compile(r"^.+Vm5 n\"(.+)\"")
        with open(pto_path, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            match = pattern.match(line)
            file = match.group(1)
            files.append(file)
        return files

    def run(self, proj_directory: str, l: Lens, p: Projection, pto_file: str, threads: int) -> str:
        run_dir = os.path.join(proj_directory, f"l{l.name}_p{p.name}")
        os.mkdir(run_dir)
        log_path = os.path.join(run_dir, "log.txt")
        log = Logger(open(log_path, 'w'), print=False)
        os.environ['OMP_NUM_THREADS'] = str(threads)

        outfile2 = f"pano_{self.batch_id}_s{self.seed}_cp{self.strategy.name}_l{l.name}_lf{self.lens_fov}_f{self.fov}_p{p.name}.pto"
        log.log(outfile2)
        outfile2 = os.path.join(run_dir, outfile2)
        command = f"pano_modify -o {outfile2} -p {p.value} --fov={self.fov}x{self.vfov} --canvas {self.pixel_width}x{self.pixel_height} -c {pto_file}"
        run_cmd(command, log)
        with open(outfile2, "a") as f:
            for cp in self.control_points:
                f.write(str(cp) + '\n')

        outfile3 = f"pano_{self.batch_id}_s{self.seed}_cp{self.strategy.name}_l{l.name}_lf{self.lens_fov}_f{self.fov}_p{p.name}_a.pto"
        outfile3 = os.path.join(run_dir, outfile3)
        with open(outfile2, "r") as f2:
            lines = f2.readlines()
        with open(outfile3, "w") as f3:
            i = 0
            for line in lines:
                if line[0] == 'i':
                    line = line.replace("p0", f"p{self.images[i].p}")
                    line = line.replace("y-0", f"y{self.images[i].y}")
                    i += 1
                f3.write(line)

        if self.strategy == ControlPointArrangementStrategy.RANDOM:
            outfile4 = f"pano_{self.batch_id}_s{self.seed}_cp{self.strategy.name}_l{l.name}_lf{self.lens_fov}_f{self.fov}_p{p.name}_a_o.pto"
            outfile4 = os.path.join(run_dir, outfile4)
            cmd = f"pto_var --opt=y,p --anchor={len(self.images) // 2} -o {outfile4} {outfile3}"
            run_cmd(cmd, log)

            outfile5 = f"pano_{self.batch_id}_s{self.seed}_cp{self.strategy.name}_l{l.name}_lf{self.lens_fov}_f{self.fov}_p{p.name}_a_o_a.pto"
            outfile5 = os.path.join(run_dir, outfile5)
            command = f"autooptimiser -a -o {outfile5} {outfile4}"
            run_cmd(command, log)
            #files = self.get_optimized_file_list(outfile5)
            final_pto_file = outfile5
        else:
            #files = [image.path for image in self.images]
            final_pto_file = outfile3

        #prefix = os.path.splitext(os.path.basename(final_pto_file))[0]
        prefix = "nona_map"
        temp_tif_prefix = os.path.join(run_dir, prefix)
        nona_cmd = f"nona -v -z LZW -r ldr -m TIFF_m -o {temp_tif_prefix} {final_pto_file}"
        run_cmd(nona_cmd, log)

        final_outfile = os.path.join(run_dir, f"{os.path.splitext(os.path.basename(final_pto_file))[0]}.tif")
        enblend_cmd = f"enblend -f{self.pixel_width}x{self.pixel_height}  --compression=LZW  -o {final_outfile} -- "
        for i in range(len(self.images)):
            j = str(i).rjust(4, '0')
            basename = f"{prefix}{j}.tif"
            path = os.path.join(run_dir, basename)
            enblend_cmd += f" {path}"
        run_cmd(enblend_cmd, log)

        log.close()
        #for i in range(len(self.images)):
        #    try:
        #        j = str(i).rjust(4, '0')
        #        basename = f"{prefix}{j}.tif"
        #        path = os.path.join(run_dir, basename)
        #        os.remove(path)
        #    except FileNotFoundError:
        #        pass

        #dry_run = f"hugin_executor --stitching -t {threads} -d --prefix {final_file} {final_file}"
        #log.log("Dry run commands for reference:")
        #run_cmd(dry_run, log)
        #command = f"hugin_executor --stitching -t {threads} --prefix {final_file} {final_file}"
        #run_cmd(command, log)

        return log_path