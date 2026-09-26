import argparse
import subprocess
from pathlib import Path


parser = argparse.ArgumentParser(description="Reconstruct a calibrated point cloud from rendered depth views")
parser.add_argument("sample")
parser.add_argument("--blender", default="blender")
parser.add_argument("--output", default=None)
parser.add_argument("--stride", type=int, default=3)
args = parser.parse_args()
sample = Path(args.sample).resolve()
output = Path(args.output).resolve() if args.output else sample / "reconstruction.ply"
script = Path(__file__).resolve().parents[1] / "reconstruction" / "blender_depth_fusion.py"
subprocess.run([args.blender, "-b", "--python", str(script), "--", "--sample", str(sample), "--output", str(output), "--stride", str(args.stride)], check=True)
print(output)