"""Operator policy for selecting high-level actions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    probs: torch.Tensor


class OperatorPolicy(torch.nn.Module):
    def __init__(self, feature_dim: int, num_operators: int, hidden_size: int = 128, dropout: float = 0.0, seed: Optional[int] = None) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_operators),
        )

    def forward(self, features: torch.Tensor) -> PolicyOutput:
        logits = self.net(features)
        probs = torch.softmax(logits, dim=-1)
        return PolicyOutput(logits=logits, probs=probs)

    def select(self, operator_ids: List[str], policy_output: PolicyOutput, mode: str = "sample") -> str:
        if not operator_ids:
            raise ValueError("No operator ids provided")
        logits = policy_output.logits.detach()
        if logits.dim() == 2:
            logits = logits[0]
        if mode == "argmax":
            idx = int(torch.argmax(logits).item())
            return operator_ids[idx]
        if mode == "sample":
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.multinomial(probs, 1).item())
            return operator_ids[idx]
        raise ValueError(f"Unknown selection mode: {mode}")
