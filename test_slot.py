"""
Unit tests for the 3x3 Slot Game Evaluator and Memetic Algorithm Solver.

Run with:
    python -m unittest test_slot.py
"""

import unittest
from evaluator import (
    SCALED_MULTIPLIERS,
    TARGET_SCALED_RTP,
    evaluate_grid_scaled,
    evaluate_reels_detailed,
    get_reel_windows,
)
from solver import MemeticSlotSolver


class TestSlotEvaluatorAndSolver(unittest.TestCase):
    def test_circular_reel_windows(self) -> None:
        reel = [0, 1, 2, 3]
        windows = get_reel_windows(reel)
        self.assertEqual(
            windows,
            [
                (0, 1, 2),
                (1, 2, 3),
                (2, 3, 0),
                (3, 0, 1),
            ],
        )

    def test_pattern_4_1_top_left_2x2(self) -> None:
        # Top-Left 2x2 is symbol 2 (scaled multiplier = 20)
        w0 = (2, 2, 0)
        w1 = (2, 2, 1)
        w2 = (3, 4, 0)
        self.assertEqual(evaluate_grid_scaled(w0, w1, w2), SCALED_MULTIPLIERS[2])

    def test_pattern_4_5_full_3x3_exclusive(self) -> None:
        # Full 3x3 of symbol 3 (scaled multiplier = 60) -> pays 5x (300), not 9x
        w0 = (3, 3, 3)
        w1 = (3, 3, 3)
        w2 = (3, 3, 3)
        self.assertEqual(evaluate_grid_scaled(w0, w1, w2), SCALED_MULTIPLIERS[3] * 5)

    def test_multiple_2x2_patterns_stack_when_not_full_3x3(self) -> None:
        # Top-Left 2x2 (4.1) and Top-Right 2x2 (4.2) both match symbol 0 (5 pts each)
        # Bottom row has different symbols so 4.5 does not trigger
        w0 = (0, 0, 1)
        w1 = (0, 0, 2)
        w2 = (0, 0, 3)
        self.assertEqual(evaluate_grid_scaled(w0, w1, w2), SCALED_MULTIPLIERS[0] * 2)

    def test_verified_10x10x10_solution(self) -> None:
        reels = [
            [4, 0, 0, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 4, 4, 0, 0, 0],
            [0, 0, 4, 4, 4, 4, 0, 0, 0, 0],
        ]
        detail = evaluate_reels_detailed(reels)
        self.assertTrue(detail.is_valid)
        self.assertEqual(detail.total_states, 1000)
        self.assertEqual(detail.total_scaled_payout, 19000)
        self.assertAlmostEqual(detail.rtp, 0.95, places=9)
        self.assertGreaterEqual(detail.win_rate, 0.55)

    def test_verified_uneven_10x12x15_all_symbols_solution(self) -> None:
        reels = [
            [4, 4, 4, 4, 2, 3, 2, 2, 1, 1],
            [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
        detail = evaluate_reels_detailed(reels)
        self.assertTrue(detail.is_valid)
        self.assertEqual(detail.total_states, 1800)
        self.assertEqual(detail.total_scaled_payout, 1800 * TARGET_SCALED_RTP)
        self.assertAlmostEqual(detail.rtp, 0.95, places=9)
        self.assertGreaterEqual(detail.win_rate, 0.55)
        present_symbols = set(reels[0]) | set(reels[1]) | set(reels[2])
        self.assertEqual(present_symbols, {0, 1, 2, 3, 4})

    def test_memetic_solver_convergence(self) -> None:
        solver = MemeticSlotSolver(reel_lengths=(10, 10, 10), seed=42)
        result = solver.solve()
        self.assertEqual(result.loss, 0)
        detail = evaluate_reels_detailed(result.reels)
        self.assertTrue(detail.is_valid)


if __name__ == "__main__":
    unittest.main()
