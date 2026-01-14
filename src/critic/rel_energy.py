"""Relational-energy critic producing alignment maps and operator scores."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from src.critic.aggregators import AggregationOutput, aggregate_features
from src.operators.definitions import operator_ids


_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+\.\d+|\d+|\d+/\d+)%?")


def _token_text(token: object) -> str:
    return str(token)


def _is_numeric(token: str) -> bool:
    return bool(_NUMBER_PATTERN.fullmatch(token))


def _is_operator(token: str) -> bool:
    return bool(re.search(r"[+\-*/=<>&^_]", token))


def _similarity(prompt_token: str, gen_token: str) -> float:
    if prompt_token == gen_token:
        return 1.0
    if _is_numeric(prompt_token) and _is_numeric(gen_token):
        return 0.7 if prompt_token == gen_token else 0.4
    if _is_operator(prompt_token) and _is_operator(gen_token):
        return 0.6
    if prompt_token.lower() == gen_token.lower():
        return 0.5
    return 0.1


@dataclass
class RelationalEnergyOutput:
    relation: List[List[float]]
    operator_scores: Dict[str, float]
    features: AggregationOutput
    feature_vector: List[float]


class RelationalEnergyCritic(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_operators: int,
        hidden_size: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_operators = num_operators
        self.q_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_operators),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q_head(features)

    def compute_relation(
        self, prompt_tokens: Sequence[object], gen_tokens: Sequence[object]
    ) -> List[List[float]]:
        prompt_text = [_token_text(tok) for tok in prompt_tokens]
        gen_text = [_token_text(tok) for tok in gen_tokens]
        relation: List[List[float]] = []
        for prompt_tok in prompt_text:
            row = []
            for gen_tok in gen_text:
                row.append(_similarity(prompt_tok, gen_tok))
            relation.append(row)
        return relation

    def evaluate(
        self,
        prompt_tokens: Sequence[object],
        gen_tokens: Sequence[object],
        key_prompt_mask: Optional[Sequence[bool]] = None,
        logits: Optional[List[List[float]]] = None,
        prev_relation: Optional[List[List[float]]] = None,
        special_token_ids: Optional[Dict[str, int]] = None,
        special_tokens: Optional[Sequence[str]] = None,
        gen_text: Optional[str] = None,
        operator_id_list: Optional[List[str]] = None,
        hidden_vector: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
    ) -> RelationalEnergyOutput:
        relation = self.compute_relation(prompt_tokens, gen_tokens)
        features = aggregate_features(
            relation=relation,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            key_prompt_mask=key_prompt_mask,
            logits=logits,
            prev_relation=prev_relation,
            special_token_ids=special_token_ids,
            special_tokens=special_tokens,
            gen_text=gen_text,
        )
        op_ids = operator_id_list or operator_ids()
        vector = features.feature_vector()
        if hidden_vector:
            vector = vector + list(hidden_vector)
        feature_tensor = torch.tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)
        q_scores = self.forward(feature_tensor)
        operator_scores = {
            op_id: float(q_scores[0, idx].item()) for idx, op_id in enumerate(op_ids)
        }
        return RelationalEnergyOutput(
            relation=relation,
            operator_scores=operator_scores,
            features=features,
            feature_vector=vector,
        )
