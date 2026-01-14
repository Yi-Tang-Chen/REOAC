"""Aggregate relational energy into operator-facing features."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.operators.definitions import OperatorId


@dataclass
class AggregationOutput:
    key_coverage: float
    key_mass_ratio: float
    key_conditioned_confidence: float
    prompt_entropy_mean: float
    prompt_entropy_p90: float
    temporal_delta: float
    hallucinated_number_rate: float
    number_copy_rate: float
    answer_form_valid: float
    token_suspect_scores: List[float]
    operator_scores: Dict[str, float]
    valid_gen_tokens: List[str]

    def feature_vector(self) -> List[float]:
        suspect_mean = sum(self.token_suspect_scores) / len(self.token_suspect_scores) if self.token_suspect_scores else 0.0
        suspect_p90 = _percentile(self.token_suspect_scores, 90.0)
        return [
            self.key_coverage,
            self.key_mass_ratio,
            self.key_conditioned_confidence,
            self.prompt_entropy_mean,
            self.prompt_entropy_p90,
            self.temporal_delta,
            self.hallucinated_number_rate,
            self.number_copy_rate,
            self.answer_form_valid,
            suspect_mean,
            suspect_p90,
        ]


FEATURE_DIM = 11
_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+\.\d+|\d+|\d+/\d+)%?")


def _softmax(values: List[float]) -> List[float]:
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(val - max_val) for val in values]
    total = sum(exps) or 1.0
    return [val / total for val in exps]


def _entropy(probs: List[float]) -> float:
    total = 0.0
    for prob in probs:
        if prob > 0:
            total -= prob * math.log(prob + 1e-12)
    return total


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    index = int(round((percentile / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[index]


def _token_is_special(token: object, special_ids: Optional[Dict[str, int]], special_tokens: Optional[Sequence[str]]) -> bool:
    if special_ids is not None and isinstance(token, int):
        return token in set(special_ids.values())
    if special_tokens is not None and isinstance(token, str):
        return token in special_tokens
    return False


def _trim_at_eos(tokens: List[object], eos_id: Optional[int], eos_token: Optional[str]) -> List[object]:
    if eos_id is not None:
        for idx, token in enumerate(tokens):
            if token == eos_id:
                return tokens[:idx]
    if eos_token is not None:
        for idx, token in enumerate(tokens):
            if token == eos_token:
                return tokens[:idx]
    return tokens


def _normalize_relation(relation: List[List[float]]) -> List[List[float]]:
    if not relation:
        return []
    prompt_len = len(relation)
    gen_len = len(relation[0]) if relation[0] else 0
    normalized = [[0.0 for _ in range(gen_len)] for _ in range(prompt_len)]
    for j in range(gen_len):
        column = [relation[i][j] for i in range(prompt_len)]
        probs = _softmax(column)
        for i in range(prompt_len):
            normalized[i][j] = probs[i]
    return normalized


def _extract_numbers(tokens: Sequence[str]) -> List[str]:
    numbers: List[str] = []
    for token in tokens:
        for match in _NUMBER_PATTERN.finditer(token):
            numbers.append(match.group(0))
    return numbers


def aggregate_features(
    relation: List[List[float]],
    prompt_tokens: Sequence[object],
    gen_tokens: Sequence[object],
    key_prompt_mask: Optional[Sequence[bool]] = None,
    logits: Optional[List[List[float]]] = None,
    prev_relation: Optional[List[List[float]]] = None,
    special_token_ids: Optional[Dict[str, int]] = None,
    special_tokens: Optional[Sequence[str]] = None,
    gen_text: Optional[str] = None,
) -> AggregationOutput:
    eos_id = special_token_ids.get("eos_id") if special_token_ids else None
    eos_token = "<eos>" if special_tokens and "<eos>" in special_tokens else None

    filtered_gen = _trim_at_eos(list(gen_tokens), eos_id=eos_id, eos_token=eos_token)
    filtered_relation = relation
    if relation and len(relation[0]) != len(filtered_gen):
        filtered_relation = [row[: len(filtered_gen)] for row in relation]

    if special_token_ids or special_tokens:
        keep_indices = [
            idx
            for idx, token in enumerate(filtered_gen)
            if not _token_is_special(token, special_token_ids, special_tokens)
        ]
        filtered_gen = [filtered_gen[idx] for idx in keep_indices]
        if filtered_relation:
            filtered_relation = [
                [row[idx] for idx in keep_indices] for row in filtered_relation
            ]
        if logits and len(logits) >= max(keep_indices, default=-1) + 1:
            logits = [logits[idx] for idx in keep_indices]

    normalized = _normalize_relation(filtered_relation)
    prompt_len = len(filtered_relation)
    gen_len = len(filtered_gen)

    key_mask = list(key_prompt_mask) if key_prompt_mask is not None else [False] * prompt_len
    key_indices = [idx for idx, flag in enumerate(key_mask) if flag]

    key_coverage = 0.0
    if key_indices and gen_len > 0:
        coverage_values = []
        for key_idx in key_indices:
            coverage_values.append(max(normalized[key_idx][j] for j in range(gen_len)))
        key_coverage = sum(coverage_values) / len(coverage_values)

    key_mass_ratio = 0.0
    if key_indices and gen_len > 0:
        mass_values = []
        for j in range(gen_len):
            mass_values.append(sum(normalized[i][j] for i in key_indices))
        key_mass_ratio = sum(mass_values) / len(mass_values)

    key_conditioned_confidence = 0.0
    if logits and gen_len > 0:
        aligned_confidences = []
        for j in range(gen_len):
            alignment = sum(normalized[i][j] for i in key_indices) if key_indices else 0.0
            if alignment < 0.5:
                continue
            probs = _softmax(logits[j])
            aligned_confidences.append(1.0 - (_entropy(probs) / math.log(len(probs) + 1e-12)))
        if aligned_confidences:
            key_conditioned_confidence = sum(aligned_confidences) / len(aligned_confidences)

    entropy_values: List[float] = []
    for j in range(gen_len):
        probs = [normalized[i][j] for i in range(prompt_len)]
        entropy_values.append(_entropy(probs))

    prompt_entropy_mean = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
    prompt_entropy_p90 = _percentile(entropy_values, 90.0)

    temporal_delta = 0.0
    if prev_relation:
        prev_norm = _normalize_relation(prev_relation)
        diffs: List[float] = []
        for i in range(min(len(prev_norm), len(normalized))):
            for j in range(min(len(prev_norm[i]), len(normalized[i]))):
                diffs.append(abs(prev_norm[i][j] - normalized[i][j]))
        temporal_delta = sum(diffs) / len(diffs) if diffs else 0.0

    prompt_str_tokens = [str(tok) for tok in prompt_tokens]
    gen_str_tokens = [str(tok) for tok in filtered_gen]
    prompt_numbers = set(_extract_numbers(prompt_str_tokens))
    gen_numbers = _extract_numbers(gen_str_tokens)
    gen_number_set = set(gen_numbers)

    hallucinated_number_rate = 0.0
    if gen_numbers:
        hallucinated = [num for num in gen_numbers if num not in prompt_numbers]
        hallucinated_number_rate = len(hallucinated) / len(gen_numbers)

    number_copy_rate = 0.0
    if prompt_numbers:
        number_copy_rate = len(prompt_numbers & gen_number_set) / len(prompt_numbers)

    answer_form_valid = 0.0
    answer_text = gen_text or (" ".join(gen_str_tokens))
    if answer_text:
        answer_text = answer_text.strip()
        if _NUMBER_PATTERN.search(answer_text.split()[-1]):
            answer_form_valid = 1.0

    token_suspect_scores: List[float] = []
    max_logit_entropy = 1.0
    if logits and logits[0]:
        max_logit_entropy = math.log(len(logits[0]) + 1e-12)
    for j in range(gen_len):
        alignment = sum(normalized[i][j] for i in key_indices) if key_indices else 0.0
        confidence_penalty = 0.0
        if logits and j < len(logits) and logits[j]:
            probs = _softmax(logits[j])
            confidence_penalty = _entropy(probs) / max_logit_entropy
        token_suspect_scores.append((1.0 - alignment) + confidence_penalty)

    operator_scores = {
        OperatorId.O_NUM.value: key_mass_ratio + number_copy_rate - hallucinated_number_rate,
        OperatorId.O_OP.value: key_coverage + (1.0 - prompt_entropy_mean),
        OperatorId.O_SCOPE.value: 1.0 - prompt_entropy_mean,
        OperatorId.O_BRANCH.value: prompt_entropy_p90 + temporal_delta,
        OperatorId.O_FAST.value: key_coverage + key_conditioned_confidence,
    }

    return AggregationOutput(
        key_coverage=key_coverage,
        key_mass_ratio=key_mass_ratio,
        key_conditioned_confidence=key_conditioned_confidence,
        prompt_entropy_mean=prompt_entropy_mean,
        prompt_entropy_p90=prompt_entropy_p90,
        temporal_delta=temporal_delta,
        hallucinated_number_rate=hallucinated_number_rate,
        number_copy_rate=number_copy_rate,
        answer_form_valid=answer_form_valid,
        token_suspect_scores=token_suspect_scores,
        operator_scores=operator_scores,
        valid_gen_tokens=gen_str_tokens,
    )
