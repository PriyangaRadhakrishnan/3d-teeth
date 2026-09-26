"""
Utility script to download official Segment Anything Model (SAM) checkpoints.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

SAM_CHECKPOINTS = {
    "vit_b": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "size_mb": 375,
    },
    "vit_l": {
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "size_mb": 1250,
    },
    "vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "size_mb": 2560,
    },
}


def download_checkpoint(model_type: str = "vit_b", target_dir: str | Path = "checkpoints") -> Path:
    if model_type not in SAM_CHECKPOINTS:
        raise ValueError(
            f"Unsupported SAM model type '{model_type}'. Choose from: {list(SAM_CHECKPOINTS.keys())}"
        )

    info = SAM_CHECKPOINTS[model_type]
    out_dir = Path(target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / info["filename"]

    if out_path.exists() and out_path.stat().st_size > 1024 * 1024 * 10:
        print(f"SAM checkpoint already exists at: {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")
        return out_path

    url = info["url"]
    print(f"Downloading official SAM {model_type} checkpoint (~{info['size_mb']} MB)...")
    print(f"Source: {url}")
    print(f"Destination: {out_path}")

    def progress_callback(blocks_transferred, block_size, total_size):
        if total_size > 0:
            downloaded = blocks_transferred * block_size
            pct = min(100.0, downloaded / total_size * 100)
            mb = downloaded / (1024 * 1024)
            tot_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\rDownloading: {pct:.1f}% ({mb:.1f}/{tot_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(out_path), reporthook=progress_callback)
    print("\nDownload complete!")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download official SAM checkpoints")
    parser.add_argument("--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b",
                        help="SAM architecture (default: vit_b, 375 MB)")
    parser.add_argument("--output-dir", default="checkpoints",
                        help="Directory to save the checkpoint (default: checkpoints/)")
    args = parser.parse_args()

    download_checkpoint(model_type=args.model_type, target_dir=args.output_dir)


if __name__ == "__main__":
    main()
