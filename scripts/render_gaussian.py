import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction.gaussian_render import render_model


def render_gaussian(sample_dir, device="cuda"):
    return render_model(sample_dir, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a trained Gaussian checkpoint")
    parser.add_argument("sample")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(render_model(args.sample, args.device))