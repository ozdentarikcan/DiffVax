#!/usr/bin/env python3
"""Download the DiffVax dataset from Hugging Face Hub.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --data-dir /path/to/data
"""

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from diffvax.utils import ensure_dataset_in_data_dir


def main():
    parser = argparse.ArgumentParser(description="Download the DiffVax dataset")
    parser.add_argument(
        "--data-dir", type=str, default=os.path.join(_project_root, "data"),
        help="Directory to store the dataset (default: data/)",
    )
    parser.add_argument(
        "--repo-id", type=str, default="ozdentarikcan/DiffVaxDataset",
        help="Hugging Face dataset repository ID",
    )
    args = parser.parse_args()

    print(f"Downloading dataset from {args.repo_id}...")
    data_dir = ensure_dataset_in_data_dir(repo_id=args.repo_id, data_dir=args.data_dir)
    print(f"Dataset ready at: {data_dir}")

    # Print summary
    for split in ["train", "validation"]:
        img_dir = os.path.join(str(data_dir), split, "images")
        if os.path.isdir(img_dir):
            count = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
            print(f"  {split}: {count} images")


if __name__ == "__main__":
    main()
