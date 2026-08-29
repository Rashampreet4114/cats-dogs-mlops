#!/usr/bin/env python
"""Generate a tiny synthetic image set into data/raw/{cat,dog}/ so the full
pipeline (split -> train -> API) can be exercised and verified without
waiting on a real Kaggle download.

NOT real training data - for pipeline smoke-testing only. Replace with
scripts/download_data.py output before training a model you actually care
about the accuracy of.
"""

import argparse
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def make_image(rng: random.Random, base_color: tuple) -> Image.Image:
    size = (256, 256)
    image = Image.new("RGB", size)
    pixels = image.load()
    for x in range(size[0]):
        for y in range(size[1]):
            noise = rng.randint(-20, 20)
            pixels[x, y] = tuple(max(0, min(255, c + noise)) for c in base_color)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=60)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)

    # Distinct base colors per class so the CNN has a learnable (if trivial) signal.
    class_colors = {"cat": (200, 120, 60), "dog": (60, 120, 200)}

    for cls, color in class_colors.items():
        out_dir = output_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            img = make_image(rng, color)
            img.save(out_dir / f"{cls}.{i}.jpg", quality=85)

    print(f"Generated {args.per_class} synthetic images per class in {output_dir}")


if __name__ == "__main__":
    main()
