#!/usr/bin/env python3
"""
Convert the downloaded raw datasets into the CSV files expected by training.

Outputs:
- data/amazon_reviews_train.csv
- data/mexwell_reviews.csv
"""

from __future__ import annotations

import argparse
import bz2
import csv
import json
from pathlib import Path
from typing import Iterable


DEFAULT_AMAZON_TRAIN = Path("/Users/sansii/Downloads/archive-2/train.ft.txt.bz2")
DEFAULT_FAKE_DIR = Path("/Users/sansii/Downloads/archive/fake")
DEFAULT_TRUE_DIR = Path("/Users/sansii/Downloads/archive/true")


def clean_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())


def iter_amazon_rows(source_path: Path) -> Iterable[tuple[int, str]]:
    label_map = {
        "__label__1": 1,
        "__label__2": 5,
    }

    with bz2.open(source_path, "rt", encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(" ", 1)
            if len(parts) != 2 or parts[0] not in label_map:
                continue

            rating = label_map[parts[0]]
            review = clean_text(parts[1])
            if review:
                yield rating, review


def convert_amazon_reviews(source_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        for rating, review in iter_amazon_rows(source_path):
            writer.writerow([rating, review])
            count += 1

    return count


def iter_fake_reviews(fake_dir: Path) -> Iterable[str]:
    for path in sorted(fake_dir.glob("*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as source:
            for raw_line in source:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                review = clean_text(payload.get("Answer", {}).get("review_text", ""))
                if review:
                    yield review


def iter_true_reviews(true_dir: Path) -> Iterable[str]:
    for path in sorted(true_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as source:
            for raw_line in source:
                review = clean_text(raw_line)
                if review:
                    yield review


def convert_mexwell(fake_dir: Path, true_dir: Path, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fake_count = 0
    true_count = 0

    with open(output_path, "w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["review_text", "label"])

        for review in iter_true_reviews(true_dir):
            writer.writerow([review, "OR"])
            true_count += 1

        for review in iter_fake_reviews(fake_dir):
            writer.writerow([review, "YP"])
            fake_count += 1

    return true_count, fake_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local dataset CSV files.")
    parser.add_argument("--amazon-train", type=Path, default=DEFAULT_AMAZON_TRAIN)
    parser.add_argument("--fake-dir", type=Path, default=DEFAULT_FAKE_DIR)
    parser.add_argument("--true-dir", type=Path, default=DEFAULT_TRUE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()

    amazon_output = args.output_dir / "amazon_reviews_train.csv"
    mexwell_output = args.output_dir / "mexwell_reviews.csv"

    amazon_count = convert_amazon_reviews(args.amazon_train, amazon_output)
    true_count, fake_count = convert_mexwell(args.fake_dir, args.true_dir, mexwell_output)

    print(f"Wrote {amazon_count} Amazon reviews to {amazon_output}")
    print(f"Wrote {true_count} genuine and {fake_count} fake reviews to {mexwell_output}")


if __name__ == "__main__":
    main()
