import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.manual_crop import crop_upper_surface


parser = argparse.ArgumentParser(description="Create a separate prototype crown crop")
parser.add_argument("obj")
parser.add_argument("--output-dir", default="data/raw/Teeth3DS/manual_crop")
parser.add_argument("--z-fraction", type=float, default=0.58)
args = parser.parse_args()
print(crop_upper_surface(args.obj, args.output_dir, args.z_fraction))