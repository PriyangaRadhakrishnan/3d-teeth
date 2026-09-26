import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction.gaussian_train import train


parser = argparse.ArgumentParser(description="Train the geometry-initialized Gaussian baseline")
parser.add_argument("sample")
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--max-points", type=int, default=12000)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
print(train(args.sample, args.iterations, args.max_points, args.device))