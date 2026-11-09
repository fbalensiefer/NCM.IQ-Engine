"""Compute per-prompt lexical quality separately from latency reporting."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean


def rouge_l(prediction, reference):
    left = re.findall(r"\w+", prediction.casefold())
    right = re.findall(r"\w+", reference.casefold())
    row = [0] * (len(right) + 1)
    for token in left:
        previous = row
        row = [0]
        for j, ref in enumerate(right, start=1):
            row.append(
                previous[j - 1] + 1 if token == ref else max(row[-1], previous[j])
            )
    return 2 * row[-1] / max(len(left) + len(right), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--references", type=Path, default=Path(__file__).with_name("references.json")
    )
    args = parser.parse_args()
    rows = json.loads(args.report.read_text())["generation"]
    refs = json.loads(args.references.read_text())
    if not refs or len(rows) < len(refs):
        parser.error(
            "Provide a reference per prompt and at least one benchmark repetition per prompt"
        )
    scores = []
    for i, (item, reference) in enumerate(zip(rows, refs), start=1):
        score = rouge_l(item["text"], reference)
        scores.append(score)
        print(f"{i}. {score:.3f} {item['text']}")
    print(f"Mean lexical ROUGE-L F1: {mean(scores):.3f}")


if __name__ == "__main__":
    main()
