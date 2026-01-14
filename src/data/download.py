"""Download GSM8K or MATH datasets from Hugging Face and normalize to JSONL."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - optional dependency.
    raise RuntimeError("datasets is required for downloading data") from exc

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


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to load config files") from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


DEFAULT_MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _default_hf_dataset(task: str) -> Tuple[str, Optional[str]]:
    if task == "gsm8k":
        return "openai/gsm8k", "main"
    if task == "math":
        return "EleutherAI/hendrycks_math", None
    return task, None


def _normalize_gsm8k(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    prompt = example.get("question") or example.get("prompt") or ""
    target = example.get("answer") or example.get("target_answer") or ""
    return {
        "id": example.get("id") or f"gsm8k-{idx}",
        "prompt": prompt,
        "target_answer": target,
        "task": "gsm8k",
        "metadata": {key: value for key, value in example.items() if key not in {"question", "prompt", "answer", "target_answer"}},
    }


def _normalize_math(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    prompt = example.get("problem") or example.get("question") or example.get("prompt") or ""
    target = example.get("solution") or example.get("answer") or example.get("target_answer") or ""
    metadata = {
        key: value
        for key, value in example.items()
        if key not in {"problem", "question", "prompt", "solution", "answer", "target_answer"}
    }
    return {
        "id": example.get("id") or f"math-{idx}",
        "prompt": prompt,
        "target_answer": target,
        "task": "math",
        "metadata": metadata,
    }


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    _ensure_dir(path)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
            count += 1
    return count


def _math_subjects(hf_config: Optional[object], subjects: Optional[List[str]]) -> List[str]:
    if subjects:
        return list(subjects)
    if isinstance(hf_config, list):
        return [str(item) for item in hf_config]
    if isinstance(hf_config, str):
        return [hf_config]
    return list(DEFAULT_MATH_SUBJECTS)


def _dump_math_example(root: str, split: str, subject: str, idx: int, example: Dict[str, Any]) -> None:
    out_dir = os.path.join(root, split, subject)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{idx:05d}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(example, handle)


def download_dataset(
    task: str,
    split: str,
    output_path: str,
    hf_dataset: Optional[str] = None,
    hf_config: Optional[object] = None,
    raw_output_path: Optional[str] = None,
    limit: Optional[int] = None,
    math_subjects: Optional[List[str]] = None,
    dump_math_subjects: bool = False,
    math_subject_root: Optional[str] = None,
) -> int:
    dataset_name, default_config = _default_hf_dataset(task)
    dataset_name = hf_dataset or dataset_name
    hf_config = hf_config if hf_config is not None else default_config

    records: List[Dict[str, Any]] = []
    raw_records: List[Dict[str, Any]] = []

    if task == "math" and dataset_name == "EleutherAI/hendrycks_math":
        subjects = _math_subjects(hf_config, math_subjects)
        for subject in subjects:
            try:
                dataset = load_dataset(dataset_name, subject, split=split)
            except Exception as exc:
                hint = (
                    f"Failed to download dataset '{dataset_name}' subject '{subject}' split '{split}'. "
                    "Check network access or set dataset.math_subjects."
                )
                raise RuntimeError(hint) from exc
            for idx, example in enumerate(_tqdm(dataset, desc=f"{subject}/{split}", unit="sample", disable=False)):
                if limit is not None and idx >= limit:
                    break
                normalized = _normalize_math(example, idx)
                normalized["metadata"]["subject"] = subject
                if not normalized["prompt"] or not normalized["target_answer"]:
                    continue
                records.append(normalized)
                if raw_output_path:
                    raw_records.append(dict(example))
                if dump_math_subjects and math_subject_root:
                    _dump_math_example(math_subject_root, split, subject, idx, example)
    else:
        try:
            dataset = load_dataset(dataset_name, hf_config, split=split)
        except Exception as exc:
            hint = (
                f"Failed to download dataset '{dataset_name}' split '{split}'. "
                "Check network access or set dataset.hf_dataset/hf_config. "
                "For MATH use hf_dataset='EleutherAI/hendrycks_math'."
            )
            raise RuntimeError(hint) from exc

        for idx, example in enumerate(_tqdm(dataset, desc=f"download-{task}-{split}", unit="sample", disable=False)):
            if limit is not None and idx >= limit:
                break
            if task == "gsm8k":
                normalized = _normalize_gsm8k(example, idx)
            else:
                normalized = _normalize_math(example, idx)
            if not normalized["prompt"] or not normalized["target_answer"]:
                continue
            records.append(normalized)
            if raw_output_path:
                raw_records.append(dict(example))

    count = _write_jsonl(output_path, records)
    if raw_output_path:
        _write_jsonl(raw_output_path, raw_records)
    return count


def _resolve_paths(config: Dict[str, Any], task: str, split: str) -> Tuple[str, Optional[str], str, Optional[str]]:
    dataset_cfg = config.get("dataset", {})
    task = dataset_cfg.get("task") or task
    data_root = os.environ.get("REOAC_DATA_ROOT", "")
    train_path = dataset_cfg.get("train_path") or "dataset/processed/gsm8k_train.jsonl"
    eval_path = dataset_cfg.get("eval_path") or "dataset/processed/gsm8k_test.jsonl"
    raw_train_path = dataset_cfg.get("raw_train_path")
    raw_eval_path = dataset_cfg.get("raw_eval_path")

    if task == "math":
        train_path = dataset_cfg.get("train_path") or "dataset/processed/math_train.jsonl"
        eval_path = dataset_cfg.get("eval_path") or "dataset/processed/math_test.jsonl"
        raw_train_path = dataset_cfg.get("raw_train_path") or "dataset/raw/math_train.jsonl"
        raw_eval_path = dataset_cfg.get("raw_eval_path") or "dataset/raw/math_test.jsonl"
    else:
        raw_train_path = raw_train_path or "dataset/raw/gsm8k_train.jsonl"
        raw_eval_path = raw_eval_path or "dataset/raw/gsm8k_test.jsonl"

    if data_root:
        train_path = os.path.join(data_root, train_path) if not os.path.isabs(train_path) else train_path
        eval_path = os.path.join(data_root, eval_path) if not os.path.isabs(eval_path) else eval_path
        raw_train_path = os.path.join(data_root, raw_train_path) if raw_train_path and not os.path.isabs(raw_train_path) else raw_train_path
        raw_eval_path = os.path.join(data_root, raw_eval_path) if raw_eval_path and not os.path.isabs(raw_eval_path) else raw_eval_path

    if split == "train":
        return task, raw_train_path, train_path, raw_train_path
    return task, raw_eval_path, eval_path, raw_eval_path


def download_from_config(config: Dict[str, Any], split: str, save_raw: bool, limit: Optional[int] = None) -> None:
    dataset_cfg = config.get("dataset", {})
    task = dataset_cfg.get("task") or "gsm8k"
    hf_dataset = dataset_cfg.get("hf_dataset")
    hf_config = dataset_cfg.get("hf_config")
    math_subjects = dataset_cfg.get("math_subjects")
    dump_math_subjects = bool(dataset_cfg.get("dump_math_subjects", False))
    math_subject_root = dataset_cfg.get("math_subject_root", "dataset/raw/math")
    data_root = os.environ.get("REOAC_DATA_ROOT", "")
    if data_root and math_subject_root and not os.path.isabs(math_subject_root):
        math_subject_root = os.path.join(data_root, math_subject_root)
    task, raw_path, output_path, raw_output = _resolve_paths(config, task, split)
    raw_output_path = raw_output if save_raw else None

    count = download_dataset(
        task=task,
        split=split,
        output_path=output_path,
        hf_dataset=hf_dataset,
        hf_config=hf_config,
        raw_output_path=raw_output_path,
        limit=limit,
        math_subjects=math_subjects,
        dump_math_subjects=dump_math_subjects,
        math_subject_root=math_subject_root,
    )
    print(f"[download] {task} {split}: wrote {count} records to {output_path}")
    if raw_output_path:
        print(f"[download] raw {task} {split}: wrote to {raw_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize datasets")
    parser.add_argument("--dataset", choices=["gsm8k", "math"], default=None)
    parser.add_argument("--config", help="Optional config path to derive output paths")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--output", help="Override processed output path")
    parser.add_argument("--raw-output", help="Optional raw output path")
    parser.add_argument("--hf-dataset", help="Override HF dataset name")
    parser.add_argument("--hf-config", help="Override HF dataset config")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()

    if args.config:
        config = _load_yaml(args.config)
        dataset_cfg = config.get("dataset", {})
        task = dataset_cfg.get("task") or args.dataset or "gsm8k"
        splits = ["train", "test"] if args.split == "all" else [args.split]
        for split in splits:
            download_from_config(config, split=split, save_raw=args.save_raw, limit=args.limit)
        return

    task = args.dataset or "gsm8k"
    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        output_path = args.output
        raw_output_path = args.raw_output if args.save_raw else None
        if not output_path:
            if task == "gsm8k":
                output_path = f"dataset/processed/gsm8k_{split}.jsonl"
                raw_output_path = raw_output_path or f"dataset/raw/gsm8k_{split}.jsonl"
            else:
                output_path = f"dataset/processed/math_{split}.jsonl"
                raw_output_path = raw_output_path or f"dataset/raw/math_{split}.jsonl"
        download_dataset(
            task=task,
            split=split,
            output_path=output_path,
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            raw_output_path=raw_output_path,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
