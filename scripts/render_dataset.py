import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.dataset_builder import build_sample


parser = argparse.ArgumentParser(description="Render one Teeth3DS OBJ into a reconstruction-ready sample")
parser.add_argument("obj")
parser.add_argument("--output-root", default="data/processed")
parser.add_argument("--blender", default="blender")
parser.add_argument("--resolution", type=int, default=640)
args = parser.parse_args()
print(build_sample(args.obj, args.output_root, args.blender, args.resolution))