"""Bootstrap data/model assets for local runs."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

from src.data.prep import prepare_data

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


def _jsonl_count(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _resolve_data_path(path: str) -> str:
    if not path:
        return path
    data_root = os.environ.get("REOAC_DATA_ROOT", "")
    if data_root and not os.path.isabs(path):
        return os.path.join(data_root, path)
    return path


def _toy_prompt_gsm8k(a: int, b: int, mode: int) -> Dict[str, str]:
    if mode % 2 == 0:
        prompt = f"If you have {a} apples and get {b} more, how many apples do you have?"
        answer = str(a + b)
    else:
        prompt = f"Each box has {a} pencils. If you have {b} boxes, how many pencils total?"
        answer = str(a * b)
    return {"prompt": prompt, "target_answer": answer}


def _toy_prompt_math(a: int, b: int, mode: int) -> Dict[str, str]:
    if mode % 2 == 0:
        prompt = f"Solve for x: x + {a} = {b}."
        answer = str(b - a)
    else:
        prompt = f"Compute {a}^2 + {b}^2."
        answer = str(a * a + b * b)
    return {"prompt": prompt, "target_answer": answer}


def _write_toy_jsonl(path: str, task: str, count: int, seed: int) -> None:
    rng = random.Random(seed)
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        for idx in _tqdm(range(count), desc=f"toy-{task}", unit="sample", disable=count <= 1):
            a = rng.randint(1, 12)
            b = rng.randint(1, 12)
            if task == "math":
                sample = _toy_prompt_math(a, b, idx)
            else:
                sample = _toy_prompt_gsm8k(a, b, idx)
            sample.update({
                "id": f"toy-{task}-{idx}",
                "task": task,
                "metadata": {"source": "toy"},
            })
            handle.write(json.dumps(sample) + "\n")


def ensure_dataset(config: Dict[str, Any], split: str, default_count: int) -> str:
    dataset_cfg = config.get("dataset", {})
    task = dataset_cfg.get("task") or ("math" if "math" in str(dataset_cfg.get("train_path", "")) else "gsm8k")
    seed = int(config.get("seed", 0))
    allow_toy = bool(dataset_cfg.get("allow_toy", False))
    download_on_missing = bool(dataset_cfg.get("download_on_missing", False))

    if split == "train":
        path = dataset_cfg.get("train_path") or dataset_cfg.get("path")
    else:
        path = dataset_cfg.get("eval_path") or dataset_cfg.get("test_path")
    path = _resolve_data_path(path)

    if not path:
        raise RuntimeError("Dataset path is not configured. Set dataset.train_path/eval_path in config.")

    if _jsonl_count(path) > 0:
        return path

    raw_path = dataset_cfg.get(f"raw_{split}_path") or dataset_cfg.get("raw_path")
    raw_path = _resolve_data_path(raw_path)
    if raw_path and os.path.exists(raw_path):
        prepare_data(raw_path, path, task)
        if _jsonl_count(path) > 0:
            return path

    if download_on_missing:
        try:
            from src.data.download import download_from_config
        except ImportError as exc:
            raise RuntimeError("datasets is required for download_on_missing") from exc
        download_from_config(config, split=split, save_raw=True)
        if _jsonl_count(path) > 0:
            return path

    if allow_toy:
        _write_toy_jsonl(path, task=task, count=default_count, seed=seed)
        return path

    raise RuntimeError(
        "Dataset is missing. Download data first (scripts/prep_data.sh) "
        "or set dataset.allow_toy=true if you want toy data for testing."
    )


def verify_model_assets(config: Dict[str, Any]) -> List[str]:
    backbone_cfg = config.get("backbone", {})
    checkpoints: List[str] = []
    for key in ("checkpoint_path", "checkpoint", "model_path"):
        value = backbone_cfg.get(key)
        if value:
            checkpoints.append(str(value))
    missing = [path for path in checkpoints if not os.path.exists(path)]
    return missing


def load_assets(config_path: str, train_count: int, eval_count: int) -> None:
    config = _load_yaml(config_path)
    train_path = ensure_dataset(config, split="train", default_count=train_count)
    eval_path = ensure_dataset(config, split="eval", default_count=eval_count)

    missing = verify_model_assets(config)
    if missing:
        print("[asset-loader] Missing model checkpoints:")
        for path in missing:
            print(f"  - {path}")
        print("[asset-loader] Using stub backbone wrappers; training will still run.")

    print(f"[asset-loader] Train data: {train_path} ({_jsonl_count(train_path)} records)")
    print(f"[asset-loader] Eval data: {eval_path} ({_jsonl_count(eval_path)} records)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap data/model assets")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--train-count", type=int, default=32, help="Toy train size if missing")
    parser.add_argument("--eval-count", type=int, default=16, help="Toy eval size if missing")
    args = parser.parse_args()
    load_assets(args.config, train_count=args.train_count, eval_count=args.eval_count)


if __name__ == "__main__":
    main()
