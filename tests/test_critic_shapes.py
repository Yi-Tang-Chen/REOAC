import unittest

from src.critic.aggregators import FEATURE_DIM
from src.critic.rel_energy import RelationalEnergyCritic
from src.operators.definitions import operator_ids


class TestCriticShapes(unittest.TestCase):
    def test_relation_shape(self):
        op_ids = operator_ids()
        critic = RelationalEnergyCritic(feature_dim=FEATURE_DIM, num_operators=len(op_ids))
        prompt_tokens = ["3", "apples"]
        gen_tokens = ["5", "apples"]
        output = critic.evaluate(
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            key_prompt_mask=[True, False],
            operator_id_list=op_ids,
        )
        self.assertEqual(len(output.relation), len(prompt_tokens))
        self.assertEqual(len(output.relation[0]), len(gen_tokens))
        self.assertIn("O_NUM", output.operator_scores)


if __name__ == "__main__":
    unittest.main()
