import unittest

from src.actor.operator_policy import OperatorPolicy
from src.backbone.simple_backbone import SimpleBackbone
from src.critic.aggregators import FEATURE_DIM
from src.critic.rel_energy import RelationalEnergyCritic
from src.operators.definitions import operator_ids
import torch

from src.rl.rollout import RolloutEngine


class TestRolloutDeterminism(unittest.TestCase):
    def test_deterministic_rollout(self):
        op_ids = operator_ids()
        backbone = SimpleBackbone({"hidden_size": 8, "vocab_size": 64})
        actor = OperatorPolicy(feature_dim=FEATURE_DIM, num_operators=len(op_ids), hidden_size=16, seed=0)
        critic = RelationalEnergyCritic(feature_dim=FEATURE_DIM, num_operators=len(op_ids))
        engine = RolloutEngine(
            backbone=backbone,
            actor=actor,
            critic=critic,
            config={"max_steps": 3, "gen_len": 6, "selection_mode": "argmax", "seed": 0, "use_hidden": False},
            device=torch.device("cpu"),
        )
        prompt = "If you have 3 apples and get 2 more, how many apples?"
        episode1 = engine.rollout_episode(prompt, task="gsm8k", verifier=lambda text, meta: 0.0)
        episode2 = engine.rollout_episode(prompt, task="gsm8k", verifier=lambda text, meta: 0.0)
        self.assertEqual(episode1.final_text, episode2.final_text)
        self.assertEqual(episode1.cost, episode2.cost)


if __name__ == "__main__":
    unittest.main()
