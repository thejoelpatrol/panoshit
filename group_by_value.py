import os
import argparse
import shutil
import subprocess
import sys
import traceback
import json
from typing import List, Dict

EXTENSIONS = ["png", "jpg", "gif", "bmp", "tif", "tiff"]

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
                if f.endswith(e):
                    result.append(f)
                    raise Continue()
        except Continue:
            continue
    return result

def main(args: argparse.Namespace):
    images = find_images(args.input_path)
    if len(images) == 0:
        raise FileNotFoundError(f"No files with images extensions found in {args.input_path}")

    light = list()
    dark = list()

    light_path = os.path.join(args.input_path, f"light_{args.cutoff}")
    dark_path = os.path.join(args.input_path, f"dark_{args.cutoff}")
    os.mkdir(light_path)
    os.mkdir(dark_path)

    images.sort()
    for f in images:
        path = os.path.join(args.input_path, f)
        print(path)
        result = run_magick(path)
        lightness = result["channelStatistics"]["blue"]
        if lightness["mean"] >= args.cutoff:
            light.append(path)
            shutil.copy(path, os.path.join(light_path, f))
        else:
            dark.append(path)
            shutil.copy(path, os.path.join(dark_path, f))

    print(f"light_{args.cutoff}: {len(light)}")
    print(f"dark_{args.cutoff}: {len(dark)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="group_by_value.py",
                          description="A utility to separate a set of images into bright/dark using imagemagick (`convert` must be installed)")
    parser.add_argument("input_path", help="the directory of images; subdirectories will be created")
    parser.add_argument("cutoff", type=int, help="lightness 0-255; 255==white")
    args = parser.parse_args()
    main(args)