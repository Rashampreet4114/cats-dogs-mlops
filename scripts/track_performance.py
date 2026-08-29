#!/usr/bin/env python
"""Post-deployment model performance tracking (M5).

Sends a batch of held-out test images (with known true labels) to the running
inference API's /predict endpoint, compares predictions to ground truth, and
writes an accuracy report to reports/post_deploy_metrics.json.

Usage:
    python scripts/track_performance.py --api-url http://localhost:8000 \
        --test-dir data/processed/test --max-per-class 50
"""

import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def run(api_url: str, test_dir: Path, max_per_class: int) -> dict:
    predictions = []
    correct = 0
    total = 0

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        true_label = class_dir.name
        files = sorted(class_dir.glob("*"))[:max_per_class]

        for f in files:
            with open(f, "rb") as fh:
                resp = requests.post(
                    f"{api_url}/predict",
                    files={"file": (f.name, fh, "image/jpeg")},
                    timeout=10,
                )
            resp.raise_for_status()
            body = resp.json()
            predicted_label = body["label"]

            is_correct = predicted_label == true_label
            correct += int(is_correct)
            total += 1
            predictions.append(
                {
                    "file": f.name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "probabilities": body["probabilities"],
                    "correct": is_correct,
                }
            )

    accuracy = correct / total if total else 0.0
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "api_url": api_url,
        "total_requests": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--test-dir", default=str(ROOT / "data" / "processed" / "test"))
    parser.add_argument("--max-per-class", type=int, default=50)
    parser.add_argument(
        "--output", default=str(ROOT / "reports" / "post_deploy_metrics.json")
    )
    args = parser.parse_args()

    report = run(args.api_url, Path(args.test_dir), args.max_per_class)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print(
        f"Sent {report['total_requests']} requests, accuracy={report['accuracy']} "
        f"-> {output_path}"
    )


if __name__ == "__main__":
    main()
