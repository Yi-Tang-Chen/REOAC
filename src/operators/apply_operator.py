"""Apply an operator to the current rollout state."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.operators.definitions import OperatorId, operator_cost
from src.operators.math_heuristics import extract_key_spans, spans_to_token_mask, tokenize_with_offsets
from src.rl.types import OperatorDecision, RolloutState


_DIGIT_PATTERN = re.compile(r"\d")


def _is_numeric_token(token: str) -> bool:
    return bool(_DIGIT_PATTERN.search(token))


def _is_operator_token(token: str) -> bool:
    return bool(re.search(r"[+*/=<>&^_-]", token))


def _select_positions(indices: List[int], limit: Optional[int] = None) -> List[int]:
    if limit is None or limit >= len(indices):
        return indices
    return indices[-limit:]


def _mask_from_positions(length: int, positions: List[int]) -> List[bool]:
    mask = [False] * length
    for pos in positions:
        if 0 <= pos < length:
            mask[pos] = True
    return mask


def apply_operator(
    state: RolloutState,
    operator_id: str,
    operator_params: Optional[Dict[str, object]] = None,
    tokenizer: Optional[object] = None,
) -> OperatorDecision:
    operator_params = operator_params or {}
    operator_id_str = operator_id
    prompt_len = state.prompt_len
    total_len = len(state.tokens)
    gen_indices = list(range(prompt_len, total_len))
    token_strings = state.token_strings or []

    update_positions: List[int] = []
    operator_meta: Dict[str, object] = {}
    branch = False
    new_schedule = None

    if operator_id_str == OperatorId.O_SCOPE.value:
        scope_len = int(operator_params.get("scope_len", 32))
        update_positions = _select_positions(gen_indices, scope_len)
        operator_meta["scope_len"] = scope_len
    elif operator_id_str == OperatorId.O_NUM.value:
        if token_strings:
            numeric_positions = [
                idx
                for idx in gen_indices
                if idx < len(token_strings) and _is_numeric_token(token_strings[idx])
            ]
            max_tokens = int(operator_params.get("max_tokens", 16))
            update_positions = _select_positions(numeric_positions, max_tokens)
            operator_meta["numeric_positions"] = numeric_positions
        else:
            update_positions = _select_positions(gen_indices, int(operator_params.get("max_tokens", 16)))
    elif operator_id_str == OperatorId.O_OP.value:
        if token_strings:
            op_positions = [
                idx
                for idx in gen_indices
                if idx < len(token_strings) and _is_operator_token(token_strings[idx])
            ]
            max_tokens = int(operator_params.get("max_tokens", 16))
            update_positions = _select_positions(op_positions, max_tokens)
            operator_meta["operator_positions"] = op_positions
        else:
            update_positions = _select_positions(gen_indices, int(operator_params.get("max_tokens", 16)))
    elif operator_id_str == OperatorId.O_BRANCH.value:
        scope_len = int(operator_params.get("scope_len", 32))
        update_positions = _select_positions(gen_indices, scope_len)
        branch = True
        operator_meta["branch_depth"] = int(operator_params.get("branch_depth", 1))
    elif operator_id_str == OperatorId.O_FAST.value:
        scope_len = int(operator_params.get("scope_len", 64))
        update_positions = _select_positions(gen_indices, scope_len)
        new_schedule = {"step_scale": float(operator_params.get("step_scale", 1.5))}
    else:
        update_positions = _select_positions(gen_indices, int(operator_params.get("scope_len", 32)))

    if not update_positions and state.metadata.get("prompt"):
        tokenization = tokenize_with_offsets(state.metadata["prompt"], tokenizer=tokenizer)
        spans = extract_key_spans(state.metadata["prompt"])
        mask = spans_to_token_mask(spans, tokenization.offsets)
        operator_meta["prompt_key_tokens"] = sum(mask)

    update_mask = _mask_from_positions(total_len, update_positions)
    cost = operator_cost(operator_id_str, operator_params)

    return OperatorDecision(
        operator_id=operator_id_str,
        params=dict(operator_params),
        update_positions=update_positions,
        update_mask=update_mask,
        branch=branch,
        new_schedule=new_schedule,
        cost=cost,
        meta=operator_meta,
    )
