"""Training loop for REOAC."""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.actor.operator_policy import OperatorPolicy
from src.backbone.lora import LoRAConfig, LoRAManager, select_finetune_parameters
from src.backbone.mdlm_wrapper import MDLMWrapper
from src.backbone.sedd_wrapper import SEDDWrapper
from src.backbone.simple_backbone import SimpleBackbone
from src.critic.aggregators import FEATURE_DIM
from src.critic.rel_energy import RelationalEnergyCritic
from src.operators.definitions import load_operator_specs, operator_ids
from src.rl.buffer import RolloutBuffer
from src.rl.losses import compute_actor_loss, compute_backbone_loss, compute_critic_loss, compute_losses
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


def _save_yaml(data: Dict[str, Any], path: str) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to save config files") from exc
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _init_distributed() -> tuple[bool, int, int, int]:
    if not dist.is_available():
        return False, 0, 0, 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def _unwrap_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def _select_backbone(config: Dict[str, Any]):
    backbone_cfg = config.get("backbone", {})
    name = str(backbone_cfg.get("name", "e2d2"))
    if name == "simple":
        return SimpleBackbone(backbone_cfg)
    if name == "sedd":
        return SEDDWrapper(backbone_cfg)
    return MDLMWrapper(backbone_cfg)


def _load_dataset(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset_cfg = config.get("dataset", {})
    path = dataset_cfg.get("train_path") or dataset_cfg.get("path")
    prompt_field = dataset_cfg.get("prompt_field", "prompt")
    answer_field = dataset_cfg.get("answer_field", "target_answer")
    task_field = dataset_cfg.get("task_field", "task")
    data = _load_jsonl(path) if path else []
    if data:
        normalized = []
        for item in data:
            normalized.append(
                {
                    "id": item.get("id"),
                    "prompt": item.get(prompt_field, item.get("prompt", "")),
                    "target_answer": item.get(answer_field, item.get("target_answer", "")),
                    "task": item.get(task_field, item.get("task", "gsm8k")),
                }
            )
        return normalized
    raise RuntimeError("Training dataset not found. Download data first.")


def _build_output_dirs(config: Dict[str, Any]) -> str:
    logging_cfg = config.get("logging", {})
    output_root = logging_cfg.get("output_dir", "runs")
    exp_name = logging_cfg.get("exp_name", "reoac")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_root, exp_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "viz"), exist_ok=True)
    return run_dir


def _write_metric(run_dir: str, payload: Dict[str, Any]) -> None:
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _save_checkpoint(
    run_dir: str,
    backbone: Any,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    lora_manager: LoRAManager,
    finetune_mode: str,
) -> None:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    actor_model = _unwrap_ddp(actor)
    critic_model = _unwrap_ddp(critic)
    backbone_model = _unwrap_ddp(backbone.model)
    torch.save(actor_model.state_dict(), os.path.join(ckpt_dir, "actor.pt"))
    torch.save(critic_model.state_dict(), os.path.join(ckpt_dir, "critic.pt"))

    if finetune_mode == "lora":
        lora_path = os.path.join(ckpt_dir, "backbone_lora")
        lora_manager.save_lora(backbone_model, lora_path)
    else:
        torch.save(backbone_model.state_dict(), os.path.join(ckpt_dir, "backbone.pt"))


def train(config_path: str, finetune_mode_override: str | None = None) -> None:
    # Silence known HF MDLM/E2D2 FutureWarnings about torch.cuda.amp.autocast deprecation.
    warnings.filterwarnings(
        "ignore",
        message="`torch.cuda.amp.autocast.*deprecated",
        category=FutureWarning,
    )
    config = _load_yaml(config_path)
    is_distributed, rank, local_rank, world_size = _init_distributed()
    base_seed = int(config.get("seed", 0))
    seed = base_seed + rank if is_distributed else base_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    run_dir = ""
    if not is_distributed or rank == 0:
        run_dir = _build_output_dirs(config)
        _save_yaml(config, os.path.join(run_dir, "config.yaml"))
    if is_distributed:
        obj_list = [run_dir]
        dist.broadcast_object_list(obj_list, src=0)
        run_dir = obj_list[0]

    backbone = _select_backbone(config)
    backbone.to(device)

    operator_specs = load_operator_specs(config)
    operator_id_list = operator_ids(operator_specs)
    rollout_cfg = dict(config.get("rollout", {}))
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

    finetune_mode = finetune_mode_override or config.get("finetune_mode", "lora")
    lora_cfg = LoRAConfig(**config.get("lora", {})) if config.get("lora") else LoRAConfig()
    lora_manager = LoRAManager(mode=finetune_mode, config=lora_cfg)
    if finetune_mode == "lora":
        backbone.model = lora_manager.apply(backbone.model)
        backbone.model.to(device)

    if is_distributed and world_size > 1:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if torch.cuda.is_available() else {}
        actor = DDP(actor, **ddp_kwargs)
        critic = DDP(critic, **ddp_kwargs)
        if hasattr(backbone, "model"):
            backbone.model = DDP(backbone.model, **ddp_kwargs)

    optim_cfg = config.get("optim", {})
    actor_lr = float(optim_cfg.get("actor_lr", 1e-4))
    critic_lr = float(optim_cfg.get("critic_lr", 1e-4))
    backbone_lr = float(optim_cfg.get("backbone_lr", 1e-5))

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=critic_lr)
    backbone_params = select_finetune_parameters(backbone.model, finetune_mode)
    backbone_opt = torch.optim.AdamW(backbone_params, lr=backbone_lr) if backbone_params else None

    rollout_cfg = dict(config.get("rollout", {}))
    if is_distributed:
        rollout_cfg["seed"] = int(rollout_cfg.get("seed", base_seed)) + rank
    rollout_engine = RolloutEngine(
        backbone=backbone,
        actor=actor,
        critic=critic,
        operator_specs=operator_specs,
        config=rollout_cfg,
        device=device,
    )
    buffer = RolloutBuffer(capacity=int(config.get("buffer_size", 1000)), seed=seed)

    dataset = _load_dataset(config)
    num_iterations = int(config.get("num_iterations", 1))
    episodes_per_iter = int(config.get("episodes_per_iter", 2))
    critic_steps = int(config.get("update", {}).get("critic_steps", 1))
    actor_steps = int(config.get("update", {}).get("actor_steps", 1))
    backbone_steps = int(config.get("update", {}).get("backbone_steps", 1))

    total_episodes = num_iterations * episodes_per_iter
    if is_distributed:
        total_episodes *= world_size
    progress = _tqdm(
        total=total_episodes,
        desc="rollout",
        unit="episode",
        disable=(total_episodes <= 1 or (is_distributed and rank != 0)),
    )

    for iteration in range(num_iterations):
        actor.eval()
        critic.eval()
        if hasattr(backbone, "model"):
            backbone.model.eval()

        for step in range(episodes_per_iter):
            sample_idx = (iteration * episodes_per_iter + step)
            if is_distributed:
                sample_idx = sample_idx * world_size + rank
            sample = dataset[sample_idx % len(dataset)]
            task = sample.get("task", "gsm8k")
            verifier = grade_gsm8k_answer if task == "gsm8k" else grade_math_answer
            episode = rollout_engine.rollout_episode(
                sample["prompt"],
                task=task,
                verifier=verifier,
                target_answer=sample.get("target_answer"),
            )
            buffer.add_episode(episode)

            if not is_distributed or rank == 0:
                suffix = f"iter{iteration}_step{step}"
            else:
                suffix = f"iter{iteration}_step{step}_rank{rank}"
            sample_path = os.path.join(run_dir, "samples", f"{suffix}.json")
            if not is_distributed or rank == 0:
                with open(sample_path, "w", encoding="utf-8") as handle:
                    json.dump({
                        "prompt": episode.prompt,
                        "final_text": episode.final_text,
                        "reward": episode.reward,
                        "cost": episode.cost,
                    }, handle, indent=2)
            progress.update(1)
            # 釋放單次 episode 暫存，避免佔用過多記憶體
            del episode

        steps = buffer.all_steps()
        actor.train()
        critic.train()
        if hasattr(backbone, "model"):
            backbone.model.train()

        for _ in range(critic_steps):
            critic_opt.zero_grad()
            loss = compute_critic_loss(steps, critic, operator_id_list, device)
            loss.backward()
            critic_opt.step()

        for _ in range(actor_steps):
            actor_opt.zero_grad()
            loss = compute_actor_loss(
                steps,
                actor,
                critic,
                device,
                temperature=float(config.get("losses", {}).get("actor_temperature", 1.0)),
            )
            loss.backward()
            actor_opt.step()

        if backbone_opt is not None:
            for _ in range(backbone_steps):
                backbone_opt.zero_grad()
                loss = compute_backbone_loss(
                    steps,
                    backbone,
                    device,
                    advantage_positive=bool(config.get("losses", {}).get("advantage_positive", True)),
                    advantage_clip=float(config.get("losses", {}).get("advantage_clip", 0.0)),
                )
                loss.backward()
                backbone_opt.step()

        metrics = compute_losses(
            steps,
            actor,
            critic,
            backbone,
            operator_id_list,
            device,
            config.get("losses", {}),
        )
        # 釋放 steps 相關暫存並嘗試清理顯存
        del steps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not is_distributed or rank == 0:
            _write_metric(run_dir, {
                "iteration": iteration,
                "critic_loss": metrics.critic_loss,
                "actor_loss": metrics.actor_loss,
                "backbone_loss": metrics.backbone_loss,
                "total_loss": metrics.total_loss,
                "episodes": len(buffer),
            })

            _save_checkpoint(run_dir, backbone, actor, critic, lora_manager, finetune_mode)

    progress.close()
    if not is_distributed or rank == 0:
        print(f"[train] finished. run_dir={run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train REOAC")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--finetune-mode", choices=["lora", "full"], default=None)
    args = parser.parse_args()
    train(args.config, finetune_mode_override=args.finetune_mode)


if __name__ == "__main__":
    main()
