"""
Unit tests for the 3x3 Slot Game Evaluator and Memetic Algorithm Solver.

Run with:
    python -m unittest test_slot.py
"""

import unittest
from evaluator import (
    SCALED_MULTIPLIERS,
    TARGET_SCALED_RTP,
    count_circular_pairs,
    evaluate_grid_scaled,
    evaluate_reels_detailed,
    evaluate_reels_fast,
    get_reel_windows,
    max_circular_run_length,
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
        w0 = (2, 2, 0)
        w1 = (2, 2, 1)
        w2 = (3, 4, 0)
        self.assertEqual(evaluate_grid_scaled(w0, w1, w2), SCALED_MULTIPLIERS[2])

    def test_pattern_4_5_full_3x3_exclusive(self) -> None:
        w0 = (3, 3, 3)
        w1 = (3, 3, 3)
        w2 = (3, 3, 3)
        self.assertEqual(evaluate_grid_scaled(w0, w1, w2), SCALED_MULTIPLIERS[3] * 5)

    def test_multiple_2x2_patterns_stack_when_not_full_3x3(self) -> None:
        w0 = (0, 0, 1)
        w1 = (0, 0, 2)
        w2 = (0, 0, 3)
        self.assertEqual(evaluate_grid_scaled(w0, w1, w2), SCALED_MULTIPLIERS[0] * 2)

    def test_verified_all_5_symbols_winning_40x42x45_solution(self) -> None:
        reels = [
            [
                0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0,
                0, 3, 3, 3, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4,
            ],
            [
                0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0,
                3, 3, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4,
            ],
            [
                0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3,
                0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4,
            ],
        ]
        detail = evaluate_reels_detailed(reels)
        fast_states, fast_wins, fast_score = evaluate_reels_fast(reels)

        self.assertEqual(fast_states, detail.total_states)
        self.assertEqual(fast_wins, detail.winning_states)
        self.assertEqual(fast_score, detail.total_scaled_payout)

        self.assertTrue(detail.is_valid)
        self.assertEqual(detail.total_states, 75600)
        self.assertEqual(detail.total_scaled_payout, 75600 * TARGET_SCALED_RTP)
        self.assertAlmostEqual(detail.rtp, 0.95, places=9)
        self.assertGreaterEqual(detail.win_rate, 0.55)

        # Verify every reel has max_run_length <= 5 and winning pairs for all 5 symbols
        for r in reels:
            self.assertEqual(set(r), {0, 1, 2, 3, 4})
            self.assertLessEqual(max_circular_run_length(r), 5)
            pairs = count_circular_pairs(r)
            for s in range(5):
                self.assertGreaterEqual(pairs[s], 1)

        # Verify all 5 symbols {0, 1, 2, 3, 4} actively contribute to the payout
        for s in range(5):
            self.assertGreater(detail.symbol_scaled_payout_breakdown[s], 0)
        self.assertEqual(
            sum(detail.symbol_scaled_payout_breakdown.values()),
            detail.total_scaled_payout,
        )

    def test_verified_compact_10x10x10_solution(self) -> None:
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

    def test_memetic_solver_convergence(self) -> None:
        solver = MemeticSlotSolver(
            reel_lengths=(10, 10, 10),
            require_all_symbols_winning=False,
            max_run_length=None,
            seed=42,
        )
        result = solver.solve()
        self.assertEqual(result.loss, 0)
        detail = evaluate_reels_detailed(result.reels)
        self.assertTrue(detail.is_valid)


if __name__ == "__main__":
    unittest.main()
