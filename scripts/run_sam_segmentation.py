"""
Phase 1 CLI: Run SAM-based segmentation on an intraoral dental sample.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.sam_segmentation import segment_dental_sample, DEFAULT_CHECKPOINT, DEFAULT_MODEL_TYPE


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Segment dental images using Segment Anything Model (SAM)")
    parser.add_argument("sample", help="Path to sample directory containing an images/ folder (e.g. data/processed/SAMPLE_upper)")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Path to SAM weights file (.pth)")
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE, choices=["vit_b", "vit_l", "vit_h"], help="SAM model architecture")
    parser.add_argument("--device", default=None, help="Execution device ('cuda' or 'cpu', defaults to auto-detect)")
    args = parser.parse_args()

    sample_dir = Path(args.sample).resolve()
    if not sample_dir.exists():
        print(f"Error: Sample directory not found: {sample_dir}", file=sys.stderr)
        sys.exit(1)

    result = segment_dental_sample(
        sample_dir=sample_dir,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print(f"Phase 1 SAM Segmentation Completed Successfully!")
    print(f"Sample: {result['sample_id']}")
    print(f"Views Processed: {result['views_processed']}")
    print("=" * 60)
    for view, data in result["results"].items():
        print(f"  [{view}] Area: {data['area_percent']:.1f}% | IoU: {data['iou_confidence']:.3f} | Mask: {data['mask_path']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
