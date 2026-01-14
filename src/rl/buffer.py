"""Rollout buffer for training data."""

from __future__ import annotations

import random
from typing import List, Sequence

from src.rl.types import EpisodeRecord, RolloutStep


class RolloutBuffer:
    def __init__(self, capacity: int = 1000, seed: int = 0) -> None:
        self.capacity = capacity
        self.episodes: List[EpisodeRecord] = []
        self.rng = random.Random(seed)

    def add_episode(self, episode: EpisodeRecord) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity :]

    def sample(self, batch_size: int) -> List[EpisodeRecord]:
        if not self.episodes:
            return []
        if batch_size >= len(self.episodes):
            return list(self.episodes)
        return self.rng.sample(self.episodes, batch_size)

    def all_steps(self) -> List[RolloutStep]:
        steps: List[RolloutStep] = []
        for episode in self.episodes:
            steps.extend(episode.steps)
        return steps

    def __len__(self) -> int:
        return len(self.episodes)

    def summary(self) -> dict:
        return {
            "episodes": len(self.episodes),
            "steps": sum(len(ep.steps) for ep in self.episodes),
        }
