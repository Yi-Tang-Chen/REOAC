"""Relational-energy critic producing alignment maps and operator scores."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from src.critic.aggregators import AggregationOutput, aggregate_features
from src.operators.definitions import OperatorId, operator_ids


_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+\.\d+|\d+|\d+/\d+)%?")


def _token_text(token: object) -> str:
    return str(token)


def _is_numeric(token: str) -> bool:
    return bool(_NUMBER_PATTERN.fullmatch(token))


def _is_operator(token: str) -> bool:
    return bool(re.search(r"[+\-*/=<>&^_]", token))


def _extract_numbers(tokens: Sequence[str]) -> List[str]:
    numbers: List[str] = []
    for token in tokens:
        for match in _NUMBER_PATTERN.finditer(token):
            numbers.append(match.group(0))
    return numbers


def _percentile_tensor(values: torch.Tensor, percentile: float) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    flat = values.reshape(-1)
    k = int(round((percentile / 100.0) * (flat.numel() - 1)))
    k = max(0, min(k, flat.numel() - 1))
    return flat.kthvalue(k + 1).values


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

    def _compute_relation_tensor(
        self,
        prompt_ids: torch.Tensor,
        gen_ids: torch.Tensor,
        prompt_numeric: torch.Tensor,
        gen_numeric: torch.Tensor,
        prompt_operator: torch.Tensor,
        gen_operator: torch.Tensor,
    ) -> torch.Tensor:
        if prompt_ids.numel() == 0 or gen_ids.numel() == 0:
            return torch.zeros(
                (prompt_ids.shape[0], gen_ids.shape[0]),
                device=prompt_ids.device,
                dtype=torch.float32,
            )
        rel = torch.full(
            (prompt_ids.shape[0], gen_ids.shape[0]),
            0.1,
            device=prompt_ids.device,
            dtype=torch.float32,
        )
        eq = prompt_ids[:, None] == gen_ids[None, :]
        rel = torch.where(eq, torch.tensor(1.0, device=rel.device), rel)
        num_mask = prompt_numeric[:, None] & gen_numeric[None, :] & ~eq
        rel = torch.where(num_mask, 0.4, rel)
        op_mask = prompt_operator[:, None] & gen_operator[None, :] & ~eq & ~num_mask
        rel = torch.where(op_mask, 0.6, rel)
        return rel

    def _aggregate_features_tensor(
        self,
        relation: torch.Tensor,
        key_prompt_mask: torch.Tensor,
        logits: Optional[torch.Tensor],
        prev_relation: Optional[torch.Tensor],
        prompt_tokens: Sequence[str],
        gen_tokens: Sequence[str],
        gen_text: Optional[str],
    ) -> AggregationOutput:
        prompt_len = relation.shape[0]
        gen_len = relation.shape[1] if relation.numel() > 0 else 0
        device = relation.device
        if gen_len == 0 or prompt_len == 0:
            return AggregationOutput(
                key_coverage=0.0,
                key_mass_ratio=0.0,
                key_conditioned_confidence=0.0,
                prompt_entropy_mean=0.0,
                prompt_entropy_p90=0.0,
                temporal_delta=0.0,
                hallucinated_number_rate=0.0,
                number_copy_rate=0.0,
                answer_form_valid=0.0,
                token_suspect_scores=[],
                operator_scores={},
                valid_gen_tokens=list(gen_tokens),
            )

        normalized = torch.softmax(relation, dim=0)
        key_mask = key_prompt_mask.to(device=device, dtype=torch.bool) if key_prompt_mask.numel() else torch.zeros(
            (prompt_len,), device=device, dtype=torch.bool
        )
        key_coverage = torch.tensor(0.0, device=device)
        key_mass_ratio = torch.tensor(0.0, device=device)
        alignment = torch.zeros((gen_len,), device=device)
        if key_mask.any():
            key_coverage = normalized[key_mask].max(dim=1).values.mean()
            alignment = normalized[key_mask].sum(dim=0)
            key_mass_ratio = alignment.mean()

        entropy_prompt = -(normalized * torch.log(normalized + 1e-12)).sum(dim=0)
        prompt_entropy_mean = entropy_prompt.mean()
        prompt_entropy_p90 = _percentile_tensor(entropy_prompt, 90.0)

        temporal_delta = torch.tensor(0.0, device=device)
        if prev_relation is not None and prev_relation.numel() > 0:
            prev_norm = torch.softmax(prev_relation.to(device=device), dim=0)
            temporal_delta = torch.abs(prev_norm - normalized).mean()

        key_conditioned_confidence = torch.tensor(0.0, device=device)
        confidence_penalty = torch.zeros((gen_len,), device=device)
        if logits is not None and logits.numel() > 0:
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            entropy_logits = -(probs * log_probs).sum(dim=-1)
            max_entropy = math.log(logits.shape[-1] + 1e-12)
            confidence = 1.0 - (entropy_logits / max_entropy)
            confidence_penalty = entropy_logits / max_entropy
            if key_mask.any():
                aligned = alignment >= 0.5
                if aligned.any():
                    key_conditioned_confidence = confidence[aligned].mean()

        suspect_scores = (1.0 - alignment) + confidence_penalty
        suspect_mean = suspect_scores.mean()
        suspect_p90 = _percentile_tensor(suspect_scores, 90.0)

        prompt_numbers = set(_extract_numbers([str(tok) for tok in prompt_tokens]))
        gen_numbers = _extract_numbers([str(tok) for tok in gen_tokens])
        gen_number_set = set(gen_numbers)
        hallucinated_number_rate = 0.0
        if gen_numbers:
            hallucinated = [num for num in gen_numbers if num not in prompt_numbers]
            hallucinated_number_rate = len(hallucinated) / len(gen_numbers)
        number_copy_rate = 0.0
        if prompt_numbers:
            number_copy_rate = len(prompt_numbers & gen_number_set) / len(prompt_numbers)

        answer_form_valid = 0.0
        answer_text = gen_text or (" ".join([str(tok) for tok in gen_tokens]))
        if answer_text:
            answer_text = answer_text.strip()
            if answer_text:
                answer_tokens = answer_text.split()
                if answer_tokens and _NUMBER_PATTERN.search(answer_tokens[-1]):
                    answer_form_valid = 1.0

        operator_scores = {
            OperatorId.O_NUM.value: float((key_mass_ratio + number_copy_rate - hallucinated_number_rate).item())
            if isinstance(key_mass_ratio, torch.Tensor)
            else float(key_mass_ratio + number_copy_rate - hallucinated_number_rate),
            OperatorId.O_OP.value: float((key_coverage + (1.0 - prompt_entropy_mean)).item())
            if isinstance(key_coverage, torch.Tensor)
            else float(key_coverage + (1.0 - prompt_entropy_mean)),
            OperatorId.O_SCOPE.value: float((1.0 - prompt_entropy_mean).item())
            if isinstance(prompt_entropy_mean, torch.Tensor)
            else float(1.0 - prompt_entropy_mean),
            OperatorId.O_BRANCH.value: float((prompt_entropy_p90 + temporal_delta).item())
            if isinstance(prompt_entropy_p90, torch.Tensor)
            else float(prompt_entropy_p90 + temporal_delta),
            OperatorId.O_FAST.value: float((key_coverage + key_conditioned_confidence).item())
            if isinstance(key_coverage, torch.Tensor)
            else float(key_coverage + key_conditioned_confidence),
        }

        return AggregationOutput(
            key_coverage=float(key_coverage.item()),
            key_mass_ratio=float(key_mass_ratio.item()),
            key_conditioned_confidence=float(key_conditioned_confidence.item()),
            prompt_entropy_mean=float(prompt_entropy_mean.item()),
            prompt_entropy_p90=float(prompt_entropy_p90.item()),
            temporal_delta=float(temporal_delta.item()),
            hallucinated_number_rate=float(hallucinated_number_rate),
            number_copy_rate=float(number_copy_rate),
            answer_form_valid=float(answer_form_valid),
            token_suspect_scores=[float(val) for val in suspect_scores.detach().cpu().tolist()],
            operator_scores=operator_scores,
            valid_gen_tokens=[str(tok) for tok in gen_tokens],
        )

    def evaluate_batch_gpu(
        self,
        prompt_tokens_batch: Sequence[Sequence[object]],
        gen_tokens_batch: Sequence[Sequence[object]],
        prompt_ids_batch: Sequence[torch.Tensor],
        gen_ids_batch: Sequence[torch.Tensor],
        key_prompt_mask_batch: Optional[Sequence[Sequence[bool]]] = None,
        logits_batch: Optional[Sequence[torch.Tensor]] = None,
        prev_relation_batch: Optional[Sequence[Optional[torch.Tensor]]] = None,
        special_token_ids: Optional[Dict[str, int]] = None,
        operator_id_list: Optional[List[str]] = None,
        hidden_vectors: Optional[Sequence[Optional[List[float]]]] = None,
        device: Optional[torch.device] = None,
        return_relation: bool = True,
    ) -> List[RelationalEnergyOutput]:
        batch = len(prompt_tokens_batch)
        if (
            len(gen_tokens_batch) != batch
            or len(prompt_ids_batch) != batch
            or len(gen_ids_batch) != batch
        ):
            raise ValueError("Batch inputs must have the same length")
        if key_prompt_mask_batch is not None and len(key_prompt_mask_batch) != batch:
            raise ValueError("key_prompt_mask_batch length mismatch")
        if logits_batch is not None and len(logits_batch) != batch:
            raise ValueError("logits_batch length mismatch")
        if prev_relation_batch is not None and len(prev_relation_batch) != batch:
            raise ValueError("prev_relation_batch length mismatch")
        if hidden_vectors is not None and len(hidden_vectors) != batch:
            raise ValueError("hidden_vectors length mismatch")

        op_ids = operator_id_list or operator_ids()
        outputs: List[RelationalEnergyOutput] = []
        feature_tensors: List[torch.Tensor] = []
        meta: List[tuple[Optional[List[List[float]]], AggregationOutput, torch.Tensor]] = []

        eos_id = special_token_ids.get("eos_id") if special_token_ids else None
        special_id_list = [sid for sid in (special_token_ids or {}).values() if sid is not None]

        for idx in range(batch):
            prompt_tokens = [str(tok) for tok in prompt_tokens_batch[idx]]
            gen_tokens = [str(tok) for tok in gen_tokens_batch[idx]]
            prompt_ids = prompt_ids_batch[idx].to(device=device)
            gen_ids = gen_ids_batch[idx].to(device=device)
            logits = logits_batch[idx].to(device=device) if logits_batch is not None else None

            if eos_id is not None:
                eos_matches = (gen_ids == eos_id).nonzero(as_tuple=False)
                if eos_matches.numel() > 0:
                    cutoff = int(eos_matches[0].item())
                    gen_ids = gen_ids[:cutoff]
                    gen_tokens = gen_tokens[:cutoff]
                    if logits is not None:
                        logits = logits[:cutoff]

            if special_id_list and gen_ids.numel() > 0:
                keep = torch.ones_like(gen_ids, dtype=torch.bool)
                for sid in special_id_list:
                    keep = keep & (gen_ids != sid)
                if keep.numel() != 0:
                    gen_ids = gen_ids[keep]
                    gen_tokens = [tok for tok, flag in zip(gen_tokens, keep.detach().cpu().tolist()) if flag]
                    if logits is not None:
                        logits = logits[keep]

            prompt_numeric = torch.tensor([_is_numeric(tok) for tok in prompt_tokens], device=device, dtype=torch.bool)
            gen_numeric = torch.tensor([_is_numeric(tok) for tok in gen_tokens], device=device, dtype=torch.bool)
            prompt_operator = torch.tensor([_is_operator(tok) for tok in prompt_tokens], device=device, dtype=torch.bool)
            gen_operator = torch.tensor([_is_operator(tok) for tok in gen_tokens], device=device, dtype=torch.bool)

            relation = self._compute_relation_tensor(
                prompt_ids=prompt_ids,
                gen_ids=gen_ids,
                prompt_numeric=prompt_numeric,
                gen_numeric=gen_numeric,
                prompt_operator=prompt_operator,
                gen_operator=gen_operator,
            )

            key_mask = (
                torch.tensor(key_prompt_mask_batch[idx], device=device, dtype=torch.bool)
                if key_prompt_mask_batch is not None
                else torch.zeros((relation.shape[0],), device=device, dtype=torch.bool)
            )
            prev_relation = prev_relation_batch[idx] if prev_relation_batch is not None else None
            hidden_vector = hidden_vectors[idx] if hidden_vectors is not None else None

            features = self._aggregate_features_tensor(
                relation=relation,
                key_prompt_mask=key_mask,
                logits=logits,
                prev_relation=prev_relation,
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                gen_text=None,
            )
            vector = torch.tensor(features.feature_vector(), device=device, dtype=torch.float32)
            if hidden_vector:
                vector = torch.cat(
                    [
                        vector,
                        torch.tensor(hidden_vector, device=device, dtype=torch.float32),
                    ],
                    dim=0,
                )
            feature_tensors.append(vector)
            relation_payload: Optional[List[List[float]]] = None
            if return_relation:
                relation_payload = relation.detach().cpu().tolist()
            meta.append((relation_payload, features, vector))

        if feature_tensors:
            feature_tensor = torch.stack(feature_tensors, dim=0)
            q_scores = self.forward(feature_tensor)
        else:
            q_scores = torch.empty((0, len(op_ids)), device=device)

        for idx, (relation_payload, features, vector) in enumerate(meta):
            operator_scores = {
                op_id: float(q_scores[idx, j].item()) for j, op_id in enumerate(op_ids)
            }
            outputs.append(
                RelationalEnergyOutput(
                    relation=relation_payload or [],
                    operator_scores=operator_scores,
                    features=features,
                    feature_vector=vector.detach().cpu().tolist(),
                )
            )
        return outputs
