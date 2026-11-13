"""
Entry point for solving and verifying the 3x3 Slot Game reel configuration.

Usage:
    python main.py
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

from evaluator import (
    MULTIPLIERS,
    SCALE_FACTOR,
    TARGET_SCALED_RTP,
    evaluate_reels_detailed,
    max_circular_run_length,
    run_monte_carlo_simulation,
)
from solver import MemeticSlotSolver


def print_verification_report(
    title: str,
    reels: Sequence[Sequence[int]],
    generations: int,
    elapsed_ms: float,
) -> dict:
    """Runs exhaustive and Monte Carlo verification and prints a formatted report."""
    detail = evaluate_reels_detailed(reels)
    mc_rtp, mc_win_rate = run_monte_carlo_simulation(reels, num_spins=500_000, seed=42)

    target_scaled = TARGET_SCALED_RTP * detail.total_states
    expected_return_per_100 = detail.rtp * 100.0
    max_runs = [max_circular_run_length(r) for r in reels]
    symbols_per_reel = [sorted(set(r)) for r in reels]

    print("=" * 76)
    print(f" {title}")
    print("=" * 76)
    print(f"Converged in          : {generations} generation(s) ({elapsed_ms:.2f} ms)")
    print(f"Reel Lengths (L1,L2,L3): {detail.reel_lengths}")
    print(f"Max Run Length / Reel : {max_runs}")
    print(f"Symbols Present / Reel: {symbols_per_reel}")
    print("-" * 76)
    for idx, reel in enumerate(reels, start=1):
        print(f"  Reel {idx} (len={len(reel):2d}) : {list(reel)}")
    print("-" * 76)
    print("[1] Exhaustive State-Space Verification (100% Exact):")
    print(f"  Total Stop States (N)       : {detail.total_states}")
    print(
        f"  Winning States              : {detail.winning_states} / {detail.total_states}"
    )
    print(
        f"  Exact Win Rate              : {detail.win_rate * 100:.2f}% "
        f"(Requirement: >= 55.00%) -> {'PASS' if detail.win_rate >= 0.55 else 'FAIL'}"
    )
    print(
        f"  Total Scaled Payout (x20)   : {detail.total_scaled_payout} / {target_scaled} "
        f"(Exact Integer Match: {detail.total_scaled_payout == target_scaled})"
    )
    print(
        f"  Exact RTP                   : {detail.rtp:.6f} "
        f"(Expected Return: {expected_return_per_100:.2f} per 100 bet) "
        f"-> {'PASS' if detail.is_valid else 'FAIL'}"
    )
    print("-" * 76)
    print("[2] Winning Pattern Hit Breakdown across All States:")
    for pat_name, count in detail.pattern_counts.items():
        pct = (count / detail.total_states) * 100.0
        print(f"  {pat_name:<24}: {count:5d} hits ({pct:6.2f}%)")
    print("-" * 76)
    print("[3] Monte Carlo Simulation (500,000 Random Spins, Bet = 100):")
    print(f"  Simulated RTP               : {mc_rtp:.4f} (Theoretical: 0.9500)")
    print(
        f"  Simulated Win Rate          : {mc_win_rate * 100:.2f}% "
        f"(Theoretical: {detail.win_rate * 100:.2f}%)"
    )
    print("=" * 76)
    print()

    assert detail.is_valid, "Verification failed: Configuration does not meet requirements!"

    return {
        "reel_lengths": list(detail.reel_lengths),
        "max_run_length_per_reel": max_runs,
        "symbols_per_reel": symbols_per_reel,
        "reels": {
            "reel_1": list(reels[0]),
            "reel_2": list(reels[1]),
            "reel_3": list(reels[2]),
        },
        "verification": {
            "total_states": detail.total_states,
            "winning_states": detail.winning_states,
            "win_rate": round(detail.win_rate, 6),
            "total_scaled_payout_x20": detail.total_scaled_payout,
            "target_scaled_payout_x20": target_scaled,
            "exact_rtp": detail.rtp,
            "expected_return_per_100_bet": expected_return_per_100,
            "pattern_counts": detail.pattern_counts,
            "is_valid": detail.is_valid,
        },
    }


def solve_and_verify(
    title: str,
    reel_lengths: Tuple[int, int, int],
    require_all_symbols_per_reel: bool = True,
    max_run_length: Optional[int] = 6,
    seed: int = 42,
) -> dict:
    """Runs the Memetic Algorithm solver for the given reel lengths and verifies the solution."""
    solver = MemeticSlotSolver(
        reel_lengths=reel_lengths,
        require_all_symbols_per_reel=require_all_symbols_per_reel,
        max_run_length=max_run_length,
        seed=seed,
    )
    t0 = time.perf_counter()
    result = solver.solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return print_verification_report(
        title=title,
        reels=result.reels,
        generations=result.generations,
        elapsed_ms=elapsed_ms,
    )


def main() -> None:
    print("Symbol Multipliers:", MULTIPLIERS)
    print(f"Common Denominator Scaling Factor: x{SCALE_FACTOR}\n")

    # Solution 1: Realistic Large Uneven Rotating Drums (24 x 25 x 30)
    # Every reel contains all 5 symbols {0,1,2,3,4} interleaved, max stack run = 6
    sol_large = solve_and_verify(
        title="Primary Solution: Realistic Large Uneven Drums (24 x 25 x 30, All 5 Symbols/Reel)",
        reel_lengths=(24, 25, 30),
        require_all_symbols_per_reel=True,
        max_run_length=6,
        seed=42,
    )

    # Solution 2: Compact Equal-Length Drums (10 x 10 x 10) for closed-form hand verification
    sol_compact = solve_and_verify(
        title="Compact Solution: Equal-Length Reels (10 x 10 x 10, Hand-Verifiable)",
        reel_lengths=(10, 10, 10),
        require_all_symbols_per_reel=False,
        max_run_length=None,
        seed=42,
    )

    output_path = Path(__file__).parent / "solution_reels.json"
    output_data = {
        "symbol_multipliers": {str(k): v for k, v in MULTIPLIERS.items()},
        "primary_realistic_large_drums_24x25x30": sol_large,
        "compact_hand_verifiable_10x10x10": sol_compact,
    }
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"Saved verified reel configurations to: {output_path.name}")


if __name__ == "__main__":
    main()
