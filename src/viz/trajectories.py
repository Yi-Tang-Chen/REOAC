"""Summaries for operator usage trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, Iterable, List


def summarize_operator_usage(records: Iterable[Dict[str, object]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        operator_id = record.get("operator_id")
        if operator_id:
            counter[str(operator_id)] += 1
    return dict(counter)


def load_jsonl(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize operator trajectories")
    parser.add_argument("--input", required=True, help="JSONL file containing operator_id per step")
    parser.add_argument("--output", default="operator_usage.json")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    summary = summarize_operator_usage(records)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
