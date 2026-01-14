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

    def _build_feature_vector(
        self,
        prompt_tokens: Sequence[object],
        gen_tokens: Sequence[object],
        key_prompt_mask: Optional[Sequence[bool]] = None,
        logits: Optional[List[List[float]]] = None,
        prev_relation: Optional[List[List[float]]] = None,
        special_token_ids: Optional[Dict[str, int]] = None,
        special_tokens: Optional[Sequence[str]] = None,
        gen_text: Optional[str] = None,
        hidden_vector: Optional[List[float]] = None,
    ) -> tuple[List[List[float]], AggregationOutput, List[float]]:
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
        vector = features.feature_vector()
        if hidden_vector:
            vector = vector + list(hidden_vector)
        return relation, features, vector

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
        relation, features, vector = self._build_feature_vector(
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            key_prompt_mask=key_prompt_mask,
            logits=logits,
            prev_relation=prev_relation,
            special_token_ids=special_token_ids,
            special_tokens=special_tokens,
            gen_text=gen_text,
            hidden_vector=hidden_vector,
        )
        op_ids = operator_id_list or operator_ids()
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

    def evaluate_batch(
        self,
        prompt_tokens_batch: Sequence[Sequence[object]],
        gen_tokens_batch: Sequence[Sequence[object]],
        key_prompt_mask_batch: Optional[Sequence[Sequence[bool]]] = None,
        logits_batch: Optional[Sequence[Sequence[Sequence[float]]]] = None,
        prev_relation_batch: Optional[Sequence[Optional[List[List[float]]]]] = None,
        special_token_ids: Optional[Dict[str, int]] = None,
        special_tokens: Optional[Sequence[str]] = None,
        gen_texts: Optional[Sequence[Optional[str]]] = None,
        operator_id_list: Optional[List[str]] = None,
        hidden_vectors: Optional[Sequence[Optional[List[float]]]] = None,
        device: Optional[torch.device] = None,
    ) -> List[RelationalEnergyOutput]:
        if len(prompt_tokens_batch) != len(gen_tokens_batch):
            raise ValueError("prompt_tokens_batch and gen_tokens_batch must have the same length")
        batch = len(prompt_tokens_batch)
        if key_prompt_mask_batch is not None and len(key_prompt_mask_batch) != batch:
            raise ValueError("key_prompt_mask_batch length mismatch")
        if logits_batch is not None and len(logits_batch) != batch:
            raise ValueError("logits_batch length mismatch")
        if prev_relation_batch is not None and len(prev_relation_batch) != batch:
            raise ValueError("prev_relation_batch length mismatch")
        if gen_texts is not None and len(gen_texts) != batch:
            raise ValueError("gen_texts length mismatch")
        if hidden_vectors is not None and len(hidden_vectors) != batch:
            raise ValueError("hidden_vectors length mismatch")

        relations: List[List[List[float]]] = []
        features_list: List[AggregationOutput] = []
        feature_vectors: List[List[float]] = []
        for idx in range(batch):
            relation, features, vector = self._build_feature_vector(
                prompt_tokens=prompt_tokens_batch[idx],
                gen_tokens=gen_tokens_batch[idx],
                key_prompt_mask=key_prompt_mask_batch[idx] if key_prompt_mask_batch is not None else None,
                logits=logits_batch[idx] if logits_batch is not None else None,
                prev_relation=prev_relation_batch[idx] if prev_relation_batch is not None else None,
                special_token_ids=special_token_ids,
                special_tokens=special_tokens,
                gen_text=gen_texts[idx] if gen_texts is not None else None,
                hidden_vector=hidden_vectors[idx] if hidden_vectors is not None else None,
            )
            relations.append(relation)
            features_list.append(features)
            feature_vectors.append(vector)

        if not feature_vectors:
            return []

        op_ids = operator_id_list or operator_ids()
        feature_tensor = torch.tensor(feature_vectors, dtype=torch.float32, device=device)
        q_scores = self.forward(feature_tensor)
        outputs: List[RelationalEnergyOutput] = []
        for idx in range(batch):
            operator_scores = {
                op_id: float(q_scores[idx, j].item()) for j, op_id in enumerate(op_ids)
            }
            outputs.append(
                RelationalEnergyOutput(
                    relation=relations[idx],
                    operator_scores=operator_scores,
                    features=features_list[idx],
                    feature_vector=feature_vectors[idx],
                )
            )
        return outputs
