"""Shared rollout data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RolloutState:
    tokens: List[int]
    prompt_len: int
    timestep: int
    token_strings: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def gen_start(self) -> int:
        return self.prompt_len

    @property
    def gen_len(self) -> int:
        return max(0, len(self.tokens) - self.prompt_len)


@dataclass
class OperatorDecision:
    operator_id: str
    params: Dict[str, Any]
    update_positions: List[int]
    update_mask: List[bool]
    branch: bool
    new_schedule: Optional[Dict[str, Any]]
    cost: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorCandidate:
    operator_id: str
    params: Dict[str, Any]
    q_score: float
    cost: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutStep:
    state: RolloutState
    decision: OperatorDecision
    relation: Optional[List[List[float]]]
    operator_scores: Dict[str, float]
    feature_vector: List[float]
    next_tokens: List[int]
    return_value: float = 0.0
    target_tokens: Optional[List[int]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    prompt: str
    prompt_tokens: List[int]
    steps: List[RolloutStep]
    final_tokens: List[int]
    final_text: str
    reward: float
    cost: float
    metrics: Dict[str, Any] = field(default_factory=dict)
