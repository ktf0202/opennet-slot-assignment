"""
Exact Evaluator and Monte Carlo Simulator for the 3x3 Slot Game.

Rules Summary:
- Display: 3 columns (reels) x 3 rows (circular drums of lengths L1, L2, L3).
- Symbols & Multipliers: {0: 0.25, 1: 0.55, 2: 1.0, 3: 3.0, 4: 5.0}
- Scaled Integer Multipliers (x20): {0: 5, 1: 11, 2: 20, 3: 60, 4: 100}
- Winning Patterns:
  - 4.1: Top-Left 2x2     (cols 0..1, rows 0..1) -> 1x symbol multiplier
  - 4.2: Top-Right 2x2    (cols 1..2, rows 0..1) -> 1x symbol multiplier
  - 4.3: Bottom-Left 2x2  (cols 0..1, rows 1..2) -> 1x symbol multiplier
  - 4.4: Bottom-Right 2x2 (cols 1..2, rows 1..2) -> 1x symbol multiplier
  - 4.5: Full 3x3         (cols 0..2, rows 0..2) -> 5x symbol multiplier
    (Note: When 4.5 Full 3x3 triggers, only the 5x pattern payout applies;
     the four inner 2x2 patterns are not double-counted.)
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# Common denominator scaling factor: all multipliers and target RTP (0.95)
# become exact integers when multiplied by 20.
SCALE_FACTOR: int = 20

MULTIPLIERS: Dict[int, float] = {
    0: 0.25,
    1: 0.55,
    2: 1.00,
    3: 3.00,
    4: 5.00,
}

SCALED_MULTIPLIERS: Dict[int, int] = {
    0: 5,    # 0.25 * 20
    1: 11,   # 0.55 * 20
    2: 20,   # 1.00 * 20
    3: 60,   # 3.00 * 20
    4: 100,  # 5.00 * 20
}

TARGET_RTP: float = 0.95
TARGET_SCALED_RTP: int = 19  # 0.95 * 20
MIN_WIN_RATE: float = 0.55


@dataclass(frozen=True)
class EvaluationResult:
    """Stores the exact exhaustive evaluation metrics of a 3-reel configuration."""
    reel_lengths: Tuple[int, int, int]
    total_states: int
    winning_states: int
    total_scaled_payout: int
    rtp: float
    win_rate: float
    pattern_counts: Dict[str, int]

    @property
    def is_valid(self) -> bool:
        """Checks exact RTP == 0.95 (using integer arithmetic) and Win Rate >= 55%."""
        exact_rtp_match = (self.total_scaled_payout == TARGET_SCALED_RTP * self.total_states)
        win_rate_ok = (self.winning_states * 100 >= 55 * self.total_states)
        return exact_rtp_match and win_rate_ok


def get_reel_windows(reel: Sequence[int]) -> List[Tuple[int, int, int]]:
    """
    Given a circular reel strip of length L, returns all L visible 3-row windows.
    Window at stop index i is (reel[i], reel[(i+1)%L], reel[(i+2)%L]).
    """
    length = len(reel)
    if length < 3:
        raise ValueError(f"Reel length must be at least 3, got {length}.")
    return [
        (reel[i], reel[(i + 1) % length], reel[(i + 2) % length])
        for i in range(length)
    ]


def max_circular_run_length(reel: Sequence[int]) -> int:
    """Returns the maximum contiguous run length of identical symbols on a circular reel."""
    length = len(reel)
    doubled = list(reel) + list(reel)
    max_run = 1
    cur = 1
    for i in range(1, len(doubled)):
        if doubled[i] == doubled[i - 1]:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 1
    return min(length, max_run)


def _get_window_signatures(reel: Sequence[int]) -> List[Tuple[int, int, int, int]]:
    """
    Compresses all L windows of a reel into equivalence classes `(top_pair, bot_pair, full_triple, count)`.
    Because pattern matching only depends on whether the top 2 rows match (`top_pair`),
    whether the bottom 2 rows match (`bot_pair`), and whether all 3 rows match (`full_triple`),
    there are at most 16 distinct window signatures regardless of reel length L.
    This accelerates exact evaluation on large drums (e.g. 24x25x30 = 18,000 states) by >100x.
    """
    length = len(reel)
    counts: Dict[Tuple[int, int, int], int] = {}
    for i in range(length):
        a = reel[i]
        b = reel[(i + 1) % length]
        c = reel[(i + 2) % length]
        top = a if a == b else -1
        bot = b if b == c else -1
        full = a if (top != -1 and top == bot) else -1
        sig = (top, bot, full)
        counts[sig] = counts.get(sig, 0) + 1
    return [(top, bot, full, cnt) for (top, bot, full), cnt in counts.items()]


def evaluate_grid_scaled(
    w0: Tuple[int, int, int],
    w1: Tuple[int, int, int],
    w2: Tuple[int, int, int],
) -> int:
    """
    Evaluates a single 3x3 grid represented by 3 column windows (w0, w1, w2),
    returning the scaled integer payout (payout * 20).

    - Pattern 4.5 (Full 3x3): all 9 cells identical -> 5 * SCALED_MULTIPLIERS[s]
      (Mutually exclusive: does not stack with 4.1..4.4)
    - Patterns 4.1..4.4 (2x2 corners): each matching 2x2 block adds SCALED_MULTIPLIERS[s]
    """
    mid_top_match = (w1[0] == w1[1])
    mid_bot_match = (w1[1] == w1[2])

    if not (mid_top_match or mid_bot_match):
        return 0

    if mid_top_match and mid_bot_match:
        s = w1[0]
        if (
            w0[0] == s and w0[1] == s and w0[2] == s
            and w2[0] == s and w2[1] == s and w2[2] == s
        ):
            return SCALED_MULTIPLIERS[s] * 5

    score = 0
    if mid_top_match:
        s_top = w1[0]
        if w0[0] == s_top and w0[1] == s_top:
            score += SCALED_MULTIPLIERS[s_top]
        if w2[0] == s_top and w2[1] == s_top:
            score += SCALED_MULTIPLIERS[s_top]

    if mid_bot_match:
        s_bot = w1[1]
        if w0[1] == s_bot and w0[2] == s_bot:
            score += SCALED_MULTIPLIERS[s_bot]
        if w2[1] == s_bot and w2[2] == s_bot:
            score += SCALED_MULTIPLIERS[s_bot]

    return score


def evaluate_reels_fast(reels: Sequence[Sequence[int]]) -> Tuple[int, int, int]:
    """
    Fast exact evaluation over all L1 * L2 * L3 stop positions using window signature compression.
    Returns (total_states, winning_states, total_scaled_payout).
    """
    sig_0 = _get_window_signatures(reels[0])
    sig_1 = _get_window_signatures(reels[1])
    sig_2 = _get_window_signatures(reels[2])

    total_states = len(reels[0]) * len(reels[1]) * len(reels[2])
    winning_states = 0
    total_scaled_payout = 0

    for top1, bot1, full1, cnt1 in sig_1:
        if top1 == -1 and bot1 == -1:
            continue
        for top0, bot0, full0, cnt0 in sig_0:
            for top2, bot2, full2, cnt2 in sig_2:
                ways = cnt0 * cnt1 * cnt2
                if full1 != -1 and full0 == full1 and full2 == full1:
                    winning_states += ways
                    total_scaled_payout += ways * SCALED_MULTIPLIERS[full1] * 5
                    continue

                score = 0
                if top1 != -1:
                    if top0 == top1:
                        score += SCALED_MULTIPLIERS[top1]
                    if top2 == top1:
                        score += SCALED_MULTIPLIERS[top1]
                if bot1 != -1:
                    if bot0 == bot1:
                        score += SCALED_MULTIPLIERS[bot1]
                    if bot2 == bot1:
                        score += SCALED_MULTIPLIERS[bot1]

                if score > 0:
                    winning_states += ways
                    total_scaled_payout += ways * score

    return total_states, winning_states, total_scaled_payout


def evaluate_reels_detailed(reels: Sequence[Sequence[int]]) -> EvaluationResult:
    """
    Detailed exhaustive evaluation over all L1 * L2 * L3 stop positions,
    including hit counts for each of the 5 winning patterns (4.1 - 4.5).
    """
    w_list_0 = get_reel_windows(reels[0])
    w_list_1 = get_reel_windows(reels[1])
    w_list_2 = get_reel_windows(reels[2])

    total_states = len(w_list_0) * len(w_list_1) * len(w_list_2)
    winning_states = 0
    total_scaled_payout = 0
    pattern_counts = {
        "4.1_top_left_2x2": 0,
        "4.2_top_right_2x2": 0,
        "4.3_bottom_left_2x2": 0,
        "4.4_bottom_right_2x2": 0,
        "4.5_full_3x3": 0,
    }

    for w0 in w_list_0:
        for w1 in w_list_1:
            for w2 in w_list_2:
                mid_top_match = (w1[0] == w1[1])
                mid_bot_match = (w1[1] == w1[2])
                if not (mid_top_match or mid_bot_match):
                    continue

                if mid_top_match and mid_bot_match:
                    s = w1[0]
                    if (
                        w0[0] == s and w0[1] == s and w0[2] == s
                        and w2[0] == s and w2[1] == s and w2[2] == s
                    ):
                        pattern_counts["4.5_full_3x3"] += 1
                        winning_states += 1
                        total_scaled_payout += SCALED_MULTIPLIERS[s] * 5
                        continue

                score = 0
                if mid_top_match:
                    s_top = w1[0]
                    if w0[0] == s_top and w0[1] == s_top:
                        pattern_counts["4.1_top_left_2x2"] += 1
                        score += SCALED_MULTIPLIERS[s_top]
                    if w2[0] == s_top and w2[1] == s_top:
                        pattern_counts["4.2_top_right_2x2"] += 1
                        score += SCALED_MULTIPLIERS[s_top]

                if mid_bot_match:
                    s_bot = w1[1]
                    if w0[1] == s_bot and w0[2] == s_bot:
                        pattern_counts["4.3_bottom_left_2x2"] += 1
                        score += SCALED_MULTIPLIERS[s_bot]
                    if w2[1] == s_bot and w2[2] == s_bot:
                        pattern_counts["4.4_bottom_right_2x2"] += 1
                        score += SCALED_MULTIPLIERS[s_bot]

                if score > 0:
                    winning_states += 1
                    total_scaled_payout += score

    rtp = total_scaled_payout / (total_states * SCALE_FACTOR)
    win_rate = winning_states / total_states

    return EvaluationResult(
        reel_lengths=(len(reels[0]), len(reels[1]), len(reels[2])),
        total_states=total_states,
        winning_states=winning_states,
        total_scaled_payout=total_scaled_payout,
        rtp=rtp,
        win_rate=win_rate,
        pattern_counts=pattern_counts,
    )


def run_monte_carlo_simulation(
    reels: Sequence[Sequence[int]],
    num_spins: int = 500_000,
    bet_amount: float = 100.0,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Runs a Monte Carlo simulation of `num_spins` random spins to empirically
    verify RTP and Win Rate.
    Returns (empirical_rtp, empirical_win_rate).
    """
    rng = random.Random(seed)
    w_list_0 = get_reel_windows(reels[0])
    w_list_1 = get_reel_windows(reels[1])
    w_list_2 = get_reel_windows(reels[2])

    wins = 0
    total_return = 0.0

    for _ in range(num_spins):
        w0 = rng.choice(w_list_0)
        w1 = rng.choice(w_list_1)
        w2 = rng.choice(w_list_2)
        scaled_score = evaluate_grid_scaled(w0, w1, w2)
        if scaled_score > 0:
            wins += 1
            total_return += bet_amount * (scaled_score / SCALE_FACTOR)

    total_bet = bet_amount * num_spins
    return total_return / total_bet, wins / num_spins
