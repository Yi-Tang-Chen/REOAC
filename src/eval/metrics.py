"""Evaluation pipeline for REOAC."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch

from src.actor.operator_policy import OperatorPolicy
from src.backbone.lora import LoRAConfig, LoRAManager
from src.backbone.mdlm_wrapper import MDLMWrapper
from src.backbone.sedd_wrapper import SEDDWrapper
from src.backbone.simple_backbone import SimpleBackbone
from src.critic.aggregators import FEATURE_DIM
from src.critic.rel_energy import RelationalEnergyCritic
from src.operators.definitions import load_operator_specs, operator_ids
from src.rl.rollout import RolloutEngine
from src.eval.verifier_gsm8k import grade_gsm8k_answer
from src.eval.verifier_math import grade_math_answer

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


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _select_backbone(config: Dict[str, Any]):
    backbone_cfg = config.get("backbone", {})
    name = str(backbone_cfg.get("name", "mdlm"))
    if name == "simple":
        return SimpleBackbone(backbone_cfg)
    if name == "sedd":
        return SEDDWrapper(backbone_cfg)
    return MDLMWrapper(backbone_cfg)


def _load_checkpoints(
    checkpoint_dir: str,
    backbone: Any,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    finetune_mode: str,
    device: torch.device,
) -> None:
    actor_path = os.path.join(checkpoint_dir, "actor.pt")
    critic_path = os.path.join(checkpoint_dir, "critic.pt")
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    if os.path.exists(critic_path):
        critic.load_state_dict(torch.load(critic_path, map_location="cpu"))

    if finetune_mode == "lora":
        lora_path = os.path.join(checkpoint_dir, "backbone_lora")
        if os.path.isdir(lora_path):
            lora_cfg = LoRAConfig()
            lora_manager = LoRAManager(mode="lora", config=lora_cfg)
            backbone.model = lora_manager.load_lora(backbone.model, lora_path)
    else:
        backbone_path = os.path.join(checkpoint_dir, "backbone.pt")
        if os.path.exists(backbone_path):
            backbone.model.load_state_dict(torch.load(backbone_path, map_location="cpu"))

    actor.to(device)
    critic.to(device)
    backbone.model.to(device)


def evaluate(config_path: str, task: str, limit: int = 0, checkpoint_dir: Optional[str] = None) -> Dict[str, float]:
    config = _load_yaml(config_path)
    dataset_cfg = config.get("dataset", {})
    eval_path = dataset_cfg.get("eval_path") or dataset_cfg.get("test_path")
    prompt_field = dataset_cfg.get("prompt_field", "prompt")
    answer_field = dataset_cfg.get("answer_field", "target_answer")
    task_field = dataset_cfg.get("task_field", "task")
    dataset = _load_jsonl(eval_path)
    if limit > 0:
        dataset = dataset[:limit]

    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone = _select_backbone(config)
    backbone.to(device)

    operator_specs = load_operator_specs(config)
    operator_id_list = operator_ids(operator_specs)
    rollout_cfg = dict(config.get("rollout", {}))
    rollout_cfg["selection_mode"] = "argmax"
    rollout_cfg["branch_steps"] = []

    use_hidden = bool(rollout_cfg.get("use_hidden", True))
    hidden_size = int(getattr(backbone, "hidden_size", 0)) if use_hidden else 0
    if use_hidden and hidden_size <= 0:
        use_hidden = False
        rollout_cfg["use_hidden"] = False
    feature_dim = FEATURE_DIM + (hidden_size if use_hidden else 0)

    actor_cfg = config.get("actor", {})
    critic_cfg = config.get("critic", {})
    actor = OperatorPolicy(
        feature_dim=feature_dim,
        num_operators=len(operator_id_list),
        hidden_size=int(actor_cfg.get("hidden_size", 128)),
        dropout=float(actor_cfg.get("dropout", 0.0)),
        seed=int(config.get("seed", 0)),
    ).to(device)
    critic = RelationalEnergyCritic(
        feature_dim=feature_dim,
        num_operators=len(operator_id_list),
        hidden_size=int(critic_cfg.get("hidden_size", 128)),
        dropout=float(critic_cfg.get("dropout", 0.0)),
    ).to(device)

    actor.eval()
    critic.eval()
    if hasattr(backbone, "model"):
        backbone.model.eval()

    eval_cfg = config.get("eval", {})
    finetune_mode = config.get("finetune_mode", "lora")
    ckpt_dir = checkpoint_dir or eval_cfg.get("checkpoint_dir")
    if ckpt_dir:
        _load_checkpoints(ckpt_dir, backbone, actor, critic, finetune_mode, device)

    rollout_engine = RolloutEngine(
        backbone=backbone,
        actor=actor,
        critic=critic,
        operator_specs=operator_specs,
        config=rollout_cfg,
        device=device,
    )

    verifier = grade_gsm8k_answer if task == "gsm8k" else grade_math_answer

    correct = 0
    total_cost = 0.0
    total_steps = 0

    for sample in _tqdm(dataset, desc="eval", unit="sample", disable=len(dataset) <= 1):
        sample_task = sample.get(task_field) or sample.get("task")
        if sample_task and sample_task != task:
            continue
        prompt = sample.get(prompt_field, sample.get("prompt", ""))
        target_answer = sample.get(answer_field, sample.get("target_answer"))
        episode = rollout_engine.rollout_episode(
            prompt,
            task=task,
            verifier=verifier,
            target_answer=target_answer,
        )
        correct += int(episode.reward > 0.0)
        total_cost += episode.cost
        total_steps += len(episode.steps)

    count = max(len(dataset), 1)
    return {
        "accuracy": correct / count,
        "avg_cost": total_cost / count,
        "avg_steps": total_steps / count,
        "samples": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate REOAC")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--task", required=True, choices=["gsm8k", "math"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    metrics = evaluate(args.config, args.task, limit=args.limit, checkpoint_dir=args.checkpoint_dir)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
