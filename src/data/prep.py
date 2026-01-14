"""Prepare raw datasets into processed JSONL format."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable.
    class _TqdmNoOp:
        def update(self, _count: int = 1) -> None:
            return None

        def close(self) -> None:
            return None

    def _tqdm(iterable=None, **_kwargs):
        if iterable is None:
            return _TqdmNoOp()
        return iterable


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("data", [])
    return []


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _normalize_record(record: Dict[str, Any], task: str) -> Dict[str, Any]:
    prompt = record.get("prompt") or record.get("question") or record.get("input") or ""
    target = record.get("target_answer") or record.get("answer") or record.get("output") or ""
    return {
        "id": record.get("id") or record.get("qid") or record.get("idx") or "",
        "prompt": prompt,
        "target_answer": target,
        "task": record.get("task") or task,
        "metadata": {key: value for key, value in record.items() if key not in {"prompt", "question", "input", "target_answer", "answer", "output"}},
    }


def prepare_data(input_path: str, output_path: str, task: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.endswith(".jsonl"):
        records = _load_jsonl(input_path)
    else:
        records = _load_json(input_path)

    with open(output_path, "w", encoding="utf-8") as handle:
        for record in _tqdm(records, desc="prep", unit="record", disable=len(records) <= 1):
            normalized = _normalize_record(record, task)
            handle.write(json.dumps(normalized) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare JSON/JSONL datasets")
    parser.add_argument("--input", required=True, help="Path to raw JSON or JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--task", default="gsm8k", help="Task label for the dataset")
    args = parser.parse_args()
    prepare_data(args.input, args.output, args.task)


if __name__ == "__main__":
    main()
