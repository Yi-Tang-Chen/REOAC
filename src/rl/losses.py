"""Loss definitions for critic, actor, and backbone updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from src.rl.types import RolloutStep


@dataclass
class LossMetrics:
    critic_loss: float
    actor_loss: float
    backbone_loss: float
    total_loss: float


def _stack_features(steps: List[RolloutStep], device: torch.device) -> torch.Tensor:
    features = torch.tensor([step.feature_vector for step in steps], device=device, dtype=torch.float32)
    return features


def compute_critic_loss(
    steps: List[RolloutStep],
    critic: torch.nn.Module,
    operator_id_list: List[str],
    device: torch.device,
) -> torch.Tensor:
    if not steps:
        return torch.tensor(0.0, device=device)
    features = _stack_features(steps, device)
    q_scores = critic(features)
    op_index = {op_id: idx for idx, op_id in enumerate(operator_id_list)}
    chosen = torch.tensor(
        [op_index.get(step.decision.operator_id, 0) for step in steps],
        device=device,
        dtype=torch.long,
    )
    returns = torch.tensor([step.return_value for step in steps], device=device, dtype=torch.float32)
    pred = q_scores.gather(1, chosen.unsqueeze(1)).squeeze(1)
    return F.mse_loss(pred, returns)


def compute_actor_loss(
    steps: List[RolloutStep],
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    if not steps:
        return torch.tensor(0.0, device=device)
    features = _stack_features(steps, device)
    with torch.no_grad():
        q_scores = critic(features)
        target_probs = torch.softmax(q_scores / max(temperature, 1e-6), dim=-1)
    actor_logits = actor(features).logits
    log_probs = F.log_softmax(actor_logits, dim=-1)
    loss = -(target_probs * log_probs).sum(dim=-1).mean()
    return loss


def compute_backbone_loss(
    steps: List[RolloutStep],
    backbone: torch.nn.Module,
    device: torch.device,
    advantage_positive: bool = True,
    advantage_clip: float = 0.0,
    backward: bool = False,
) -> torch.Tensor:
    total_count = 0
    for step in steps:
        if not step.decision.update_positions or not step.target_tokens:
            continue
        for pos in step.decision.update_positions:
            if pos < len(step.target_tokens):
                total_count += 1

    if total_count == 0:
        return torch.tensor(0.0, device=device)

    if not backward:
        with torch.no_grad():
            total_loss = torch.tensor(0.0, device=device)
            for step in steps:
                if not step.decision.update_positions or not step.target_tokens:
                    continue
                input_ids = torch.tensor(step.state.tokens, device=device).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                context_mask = torch.zeros_like(input_ids)
                if step.state.prompt_len > 0:
                    context_mask[0, : step.state.prompt_len] = 1
                outputs = backbone.forward(
                    input_ids,
                    step.state.timestep,
                    attention_mask=attention_mask,
                    context_mask=context_mask,
                )
                logits = outputs.logits[0]
                advantage = float(step.return_value)
                if advantage_positive:
                    advantage = max(advantage, 0.0)
                if advantage_clip and advantage_clip > 0:
                    advantage = max(min(advantage, advantage_clip), -advantage_clip)
                weight = torch.tensor(advantage, device=device, dtype=torch.float32)

                for pos in step.decision.update_positions:
                    if pos < len(step.target_tokens):
                        target = torch.tensor(step.target_tokens[pos], device=device, dtype=torch.long)
                        loss = F.cross_entropy(logits[pos].unsqueeze(0), target.unsqueeze(0), reduction="none")
                        total_loss = total_loss + loss * weight
            return total_loss / total_count

    total_loss_value = 0.0
    for step in steps:
        if not step.decision.update_positions or not step.target_tokens:
            continue
        input_ids = torch.tensor(step.state.tokens, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        context_mask = torch.zeros_like(input_ids)
        if step.state.prompt_len > 0:
            context_mask[0, : step.state.prompt_len] = 1
        outputs = backbone.forward(
            input_ids,
            step.state.timestep,
            attention_mask=attention_mask,
            context_mask=context_mask,
        )
        logits = outputs.logits[0]
        advantage = float(step.return_value)
        if advantage_positive:
            advantage = max(advantage, 0.0)
        if advantage_clip and advantage_clip > 0:
            advantage = max(min(advantage, advantage_clip), -advantage_clip)
        weight = float(advantage)
        step_loss = None

        for pos in step.decision.update_positions:
            if pos < len(step.target_tokens):
                target = torch.tensor(step.target_tokens[pos], device=device, dtype=torch.long)
                loss = F.cross_entropy(logits[pos].unsqueeze(0), target.unsqueeze(0), reduction="none")
                step_loss = loss if step_loss is None else step_loss + loss

        if step_loss is None:
            continue

        weighted = step_loss * weight
        (weighted / total_count).backward()
        total_loss_value += float(weighted.detach().cpu().item())

    return torch.tensor(total_loss_value / total_count, device=device)


def compute_losses(
    steps: List[RolloutStep],
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    backbone: torch.nn.Module,
    operator_id_list: List[str],
    device: torch.device,
    config: Dict[str, float],
) -> LossMetrics:
    with torch.no_grad():
        critic_loss = compute_critic_loss(steps, critic, operator_id_list, device)
        actor_loss = compute_actor_loss(
            steps,
            actor,
            critic,
            device,
            temperature=float(config.get("actor_temperature", 1.0)),
        )
        backbone_loss = compute_backbone_loss(
            steps,
            backbone,
            device,
            advantage_positive=bool(config.get("advantage_positive", True)),
            advantage_clip=float(config.get("advantage_clip", 0.0)),
        )
        total_loss = (
            float(config.get("critic_weight", 1.0)) * critic_loss
            + float(config.get("actor_weight", 1.0)) * actor_loss
            + float(config.get("backbone_weight", 1.0)) * backbone_loss
        )
    return LossMetrics(
        critic_loss=float(critic_loss.detach().cpu().item()),
        actor_loss=float(actor_loss.detach().cpu().item()),
        backbone_loss=float(backbone_loss.detach().cpu().item()),
        total_loss=float(total_loss.detach().cpu().item()),
    )
