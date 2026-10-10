"""
CLI script to evaluate 3D dental reconstruction accuracy and computational profile.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_reconstruction import evaluate_sample_reconstruction
from evaluation.profiling import ComputationalProfiler
from evaluation.ablation import run_ablation_comparison


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation and Profiling on 3D Dental Reconstruction")
    parser.add_argument("sample", help="Path to processed sample directory (e.g. data/processed/00OMSZGW_upper)")
    parser.add_argument("--reference", default=None, help="Path to ground truth reference model (.obj or .ply)")
    parser.add_argument("--ablation", action="store_true", help="Run comparative ablation comparison")
    parser.add_argument("--points", type=int, default=50000, help="Number of surface points for evaluation")
    args = parser.parse_args()

    sample_dir = Path(args.sample).resolve()
    if not sample_dir.exists():
        print(f"Error: Sample directory not found: {sample_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"Evaluating Sample: {sample_dir.name}")
    print("=" * 60)

    # 1. Run accuracy evaluation
    eval_res = evaluate_sample_reconstruction(
        sample_dir=sample_dir,
        reference_model_path=args.reference,
        sample_count=args.points
    )

    # 2. Run ablation study if requested
    if args.ablation:
        print("\n" + "=" * 60)
        print("Running Experimental Ablation Study...")
        print("=" * 60)
        ablation_res = run_ablation_comparison(
            sample_dir=sample_dir,
            reference_path=args.reference,
            sample_points=args.points
        )

    print("\nEvaluation successfully completed.")


if __name__ == "__main__":
    main()
