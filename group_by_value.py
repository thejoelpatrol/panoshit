import os
import argparse
import shutil
import subprocess
import sys
import traceback
import json
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Dict

EXTENSIONS = ["png", "jpg", "gif", "bmp", "tif", "tiff"]
THREADS = 6

class Continue(Exception):
    pass

def run_magick(image_path: str) -> Dict:
    args = ["convert", image_path, "-colorspace", "hsl", "json:"]
    try:
        output = subprocess.check_output(args)
        result = json.loads(output)
        return result[0]["image"]
    except OSError as e:
        print(traceback.format_exc())
        print("is imagemagick installed?")
        sys.exit(1)


def find_images(directory_path: str) -> List[str]:
    result = list()
    for f in os.listdir(directory_path):
        try:
            for e in EXTENSIONS:
                if f.endswith('.' + e):
                    result.append(f)
                    raise Continue()
        except Continue:
            continue
    return result

def process_image(path: str, cutoff: int, light: List[str], dark: List[str], light_path: str, dark_path: str):
    result = run_magick(path)
    lightness = result["channelStatistics"]["blue"]
    if lightness["mean"] >= cutoff:
        light.append(path)
        shutil.copy(path, os.path.join(light_path, os.path.basename(path)))
    else:
        dark.append(path)
        shutil.copy(path, os.path.join(dark_path, os.path.basename(path)))

def main(args: argparse.Namespace):
    images = find_images(args.input_path)
    if len(images) == 0:
        raise FileNotFoundError(f"No files with images extensions found in {args.input_path}")

    light: List[str] = list()
    dark : List[str] = list()

    light_path = os.path.join(args.input_path, f"light_{args.cutoff}")
    dark_path = os.path.join(args.input_path, f"dark_{args.cutoff}")
    try:
        os.mkdir(light_path)
        os.mkdir(dark_path)
    except FileExistsError:
        pass

    images.sort()
    future_results = list()
    with ThreadPoolExecutor(max_workers=THREADS) as thread_pool:
        for f in images:
            path = os.path.join(args.input_path, f)
            future_results.append(thread_pool.submit(process_image, path, args.cutoff, light, dark, light_path, dark_path))
    for i, r in enumerate(future_results):
        while not r.done():
            time.sleep(0.1)
        
        e = r.exception()
        if (e):
            print(traceback.format_exception(e))
            sys.exit(1)

    print(f"light_{args.cutoff}: {len(light)}")
    print(f"dark_{args.cutoff}: {len(dark)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="group_by_value.py",
                          description="A utility to separate a set of images into bright/dark using imagemagick (`convert` must be installed)")
    parser.add_argument("input_path", help="the directory of images; subdirectories will be created")
    parser.add_argument("cutoff", type=int, help="lightness 0-255; 255==white")
    args = parser.parse_args()
    main(args)