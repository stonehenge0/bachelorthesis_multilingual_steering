#!/usr/bin/env python3
"""
inspect_pt.py

Quickly scan a folder for .pt files, load them, and print their contents.

Usage:
  python inspect_pt.py /path/to/folder
"""

import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect .pt files in a folder.")
    parser.add_argument("folder", type=str, help="Folder to scan for .pt files")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[error] '{folder}' is not a folder")
        return 1

    pt_files = sorted(folder.glob("*.pt"))
    if not pt_files:
        print("No .pt files found.")
        return 0

    print("Found .pt files:")
    for f in pt_files:
        print(" -", f)

    for f in pt_files:
        print(f"\n=== {f} ===")
        try:
            obj = torch.load(f, map_location="cpu")
            print(obj)
        except Exception as e:
            print(f"[error] Could not load {f}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
