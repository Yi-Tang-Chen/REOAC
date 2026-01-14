import unittest

from src.operators.apply_operator import apply_operator
from src.rl.types import RolloutState


class TestOperator(unittest.TestCase):
    def test_num_operator_selects_numeric_tokens(self):
        tokens = list(range(10))
        token_strings = ["Q", ":", "3", "apples", "cost", "3", "each", "total", "9", "."]
        state = RolloutState(tokens=tokens, prompt_len=5, timestep=0, token_strings=token_strings)
        decision = apply_operator(state, "O_NUM", {"max_tokens": 10})
        self.assertIn(5, decision.update_positions)
        self.assertIn(8, decision.update_positions)

    def test_scope_operator_limits_tail(self):
        tokens = list(range(10))
        token_strings = ["Q", ":", "3", "apples", "cost", "3", "each", "total", "9", "."]
        state = RolloutState(tokens=tokens, prompt_len=5, timestep=0, token_strings=token_strings)
        decision = apply_operator(state, "O_SCOPE", {"scope_len": 2})
        self.assertEqual(decision.update_positions, [8, 9])


if __name__ == "__main__":
    unittest.main()
