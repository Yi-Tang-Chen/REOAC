"""Lightweight backbone for tests and local debugging."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.backbone.tokenizer import SimpleTokenizer


@dataclass
class BackboneOutput:
    logits: torch.Tensor
    hidden: torch.Tensor


class SimpleBackbone(torch.nn.Module):
    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        super().__init__()
        self.config = config or {}
        self.tokenizer = self.config.get("tokenizer") or SimpleTokenizer()
        self.hidden_size = int(self.config.get("hidden_size", 32))
        self.vocab_size = int(self.config.get("vocab_size", 128))
        self.embed = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        self.proj = torch.nn.Linear(self.hidden_size, self.vocab_size)
        self.model = self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def tokenize(self, prompt: str) -> List[int]:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return [idx % self.vocab_size for idx in ids]

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: int,
        cond: Optional[object] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        hidden = self.embed(x_t)
        logits = self.proj(hidden)
        return BackboneOutput(logits=logits, hidden=hidden)

    def get_hidden(self, x_t: torch.Tensor, timestep: int, cond: Optional[object] = None) -> torch.Tensor:
        return self.forward(x_t, timestep, cond=cond).hidden

    def apply_unmask(self, x_t: torch.Tensor, logits: torch.Tensor, update_positions: List[int]) -> torch.Tensor:
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        new_tokens = x_t.clone()
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        for pos in update_positions:
            if 0 <= pos < new_tokens.shape[1]:
                new_tokens[0, pos] = int(torch.argmax(logits[0, pos]).item())
        return new_tokens.squeeze(0)

    def step(
        self,
        x_t: torch.Tensor,
        timestep: int,
        operator_mask: List[bool],
        cond: Optional[object] = None,
    ) -> Tuple[torch.Tensor, BackboneOutput]:
        output = self.forward(x_t, timestep, cond=cond)
        update_positions = [idx for idx, flag in enumerate(operator_mask) if flag]
        x_next = self.apply_unmask(x_t, output.logits, update_positions)
        return x_next, output
