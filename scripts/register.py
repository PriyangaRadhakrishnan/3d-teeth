import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registration.icp import register


parser = argparse.ArgumentParser(description="Run ICP against the original Teeth3DS OBJ")
parser.add_argument("reconstruction")
parser.add_argument("original_obj")
parser.add_argument("--transform", required=True)
parser.add_argument("--output", default=None)
parser.add_argument("--sample-count", type=int, default=50000)
args = parser.parse_args()
output = Path(args.output) if args.output else Path(args.reconstruction).with_name("registered_reconstruction.ply")
print(register(args.reconstruction, args.original_obj, args.transform, output, args.sample_count))