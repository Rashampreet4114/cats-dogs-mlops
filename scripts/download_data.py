#!/usr/bin/env python
"""Download the Kaggle Dogs vs Cats dataset and organize a subset into
data/raw/{cat,dog}/ for the preprocessing pipeline.

Requires a free Kaggle account + API token (~/.kaggle/kaggle.json), see README.

Usage:
    python scripts/download_data.py                       # default: 2000 images/class
    python scripts/download_data.py --per-class 500        # smaller/faster
    python scripts/download_data.py --dataset salader/dogs-vs-cats --full
"""

import argparse
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = "salader/dogs-vs-cats"


def _check_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    return kaggle_json.exists() or bool(has_env)


def _download_dataset(dataset: str, dest_dir: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading Kaggle dataset '{dataset}' ...")
    api.dataset_download_files(dataset, path=str(dest_dir), unzip=False)

    zips = list(dest_dir.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"No zip file downloaded to {dest_dir}")
    for zip_path in zips:
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)


def _classify(path: Path) -> str | None:
    name = (path.name + " " + " ".join(p.name for p in path.parents)).lower()
    if "cat" in name:
        return "cat"
    if "dog" in name:
        return "dog"
    return None


def _collect_images(search_dir: Path) -> dict:
    buckets = {"cat": [], "dog": []}
    exts = {".jpg", ".jpeg", ".png"}
    for path in search_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        cls = _classify(path)
        if cls:
            buckets[cls].append(path)
    return buckets


def organize_subset(
    source_dir: Path, output_dir: Path, per_class: int | None, seed: int = 42
) -> dict:
    rng = random.Random(seed)
    buckets = _collect_images(source_dir)

    counts = {}
    for cls, files in buckets.items():
        rng.shuffle(files)
        selected = files if per_class is None else files[:per_class]
        out_dir = output_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in selected:
            shutil.copy2(f, out_dir / f.name)
        counts[cls] = len(selected)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--per-class",
        type=int,
        default=2000,
        help="Max images per class to keep (fast local training). Ignored with --full.",
    )
    parser.add_argument("--full", action="store_true", help="Keep the entire dataset")
    parser.add_argument(
        "--output-dir", default=str(ROOT / "data" / "raw"), help="Where to place cat/ and dog/"
    )
    args = parser.parse_args()

    if not _check_kaggle_credentials():
        print(
            "Kaggle API credentials not found.\n"
            "Create a free Kaggle account, then generate an API token at "
            "https://www.kaggle.com/settings -> 'Create New Token', and save it to "
            "~/.kaggle/kaggle.json (chmod 600), or set KAGGLE_USERNAME / KAGGLE_KEY "
            "environment variables. See README.md for details.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    per_class = None if args.full else args.per_class

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _download_dataset(args.dataset, tmp_dir)
        counts = organize_subset(tmp_dir, output_dir, per_class)

    print(f"Done. Images written to {output_dir}: {counts}")


if __name__ == "__main__":
    main()
