"""Operator action space definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class OperatorId(str, Enum):
    O_NUM = "O_NUM"
    O_OP = "O_OP"
    O_SCOPE = "O_SCOPE"
    O_BRANCH = "O_BRANCH"
    O_FAST = "O_FAST"


@dataclass
class OperatorSpec:
    operator_id: OperatorId
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    base_cost: float = 1.0


DEFAULT_OPERATOR_SPECS: List[OperatorSpec] = [
    OperatorSpec(
        operator_id=OperatorId.O_NUM,
        description="Focus updates on numeric tokens or number-related spans.",
        params={"max_tokens": 16},
        base_cost=1.2,
    ),
    OperatorSpec(
        operator_id=OperatorId.O_OP,
        description="Focus updates near operators and units.",
        params={"max_tokens": 16},
        base_cost=1.1,
    ),
    OperatorSpec(
        operator_id=OperatorId.O_SCOPE,
        description="Limit updates to a local scope near the tail.",
        params={"scope_len": 32},
        base_cost=1.0,
    ),
    OperatorSpec(
        operator_id=OperatorId.O_BRANCH,
        description="Trigger a counterfactual branch.",
        params={"branch_depth": 1},
        base_cost=2.0,
    ),
    OperatorSpec(
        operator_id=OperatorId.O_FAST,
        description="Aggressive update for faster convergence.",
        params={"scope_len": 64},
        base_cost=1.5,
    ),
]


def _spec_index(specs: Iterable[OperatorSpec]) -> Dict[str, OperatorSpec]:
    return {spec.operator_id.value: spec for spec in specs}


def load_operator_specs(config: Optional[Dict[str, Any]] = None) -> List[OperatorSpec]:
    if not config or "operators" not in config:
        return DEFAULT_OPERATOR_SPECS
    specs: List[OperatorSpec] = []
    for entry in config["operators"]:
        operator_id = OperatorId(entry["id"])
        specs.append(
            OperatorSpec(
                operator_id=operator_id,
                description=entry.get("description", operator_id.value),
                params=dict(entry.get("params", {})),
                base_cost=float(entry.get("base_cost", 1.0)),
            )
        )
    return specs


def operator_cost(operator_id: str, params: Optional[Dict[str, Any]] = None) -> float:
    params = params or {}
    spec_map = _spec_index(DEFAULT_OPERATOR_SPECS)
    spec = spec_map.get(operator_id)
    if spec is None:
        return 1.0
    cost = spec.base_cost
    if operator_id == OperatorId.O_BRANCH.value:
        cost += float(params.get("branch_depth", spec.params.get("branch_depth", 1)))
    if operator_id == OperatorId.O_FAST.value:
        cost += 0.5
    return cost


def operator_ids(specs: Optional[List[OperatorSpec]] = None) -> List[str]:
    specs = specs or DEFAULT_OPERATOR_SPECS
    return [spec.operator_id.value for spec in specs]
