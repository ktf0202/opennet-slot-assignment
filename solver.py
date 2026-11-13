"""
Memetic Algorithm Solver (Genetic Algorithm + Local Search) for the 3x3 Slot Game.

Why Memetic Algorithm + Integer Scaling (x20)?
1. Common-Denominator Integer Scaling (x20):
   Symbol multipliers {0.25, 0.55, 1, 3, 5} and target RTP (0.95) all share a
   common denominator of 20. Multiplying by 20 converts the exact equality
   constraint `RTP == 0.95` into an exact integer sum constraint:
       total_scaled_payout == 19 * (L1 * L2 * L3)
   This completely eliminates IEEE-754 floating-point precision errors.

2. Genetic Algorithm (Population + Selection + Crossover + Mutation):
   Explores the non-linear combinatorial space of 3 circular reels globally,
   rapidly evolving reel strips that satisfy `Win Rate >= 55%` and bring RTP
   into the neighborhood of 0.95.

3. Memetic Local Search (Single-Cell & Adjacent-Pair Hill Climbing):
   Pure GA struggles with the last few integer points to hit an exact equality
   constraint (`total_scaled_payout == 19 * N`). Because winning patterns require
   vertically adjacent identical symbols, our Memetic Local Search explores both
   single-cell flips and adjacent-pair replacements `(pos, pos+1) = (s, s)` on
   the elite chromosomes, converging to `loss == 0` in milliseconds.
"""

from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from evaluator import (
    SCALE_FACTOR,
    TARGET_SCALED_RTP,
    MIN_WIN_RATE,
    evaluate_reels_fast,
)

Chromosome = List[List[int]]  # [reel_0, reel_1, reel_2]


@dataclass
class SolverHistoryStep:
    generation: int
    best_loss: int
    best_rtp: float
    best_win_rate: float


@dataclass
class SolverResult:
    reels: Chromosome
    generations: int
    loss: int
    total_states: int
    winning_states: int
    total_scaled_payout: int
    history: List[SolverHistoryStep]


class MemeticSlotSolver:
    """
    Memetic Algorithm (GA + Local Search) solver for finding valid 3x3 slot reels.
    """

    def __init__(
        self,
        reel_lengths: Tuple[int, int, int] = (10, 10, 12),
        population_size: int = 50,
        elite_size: int = 4,
        tournament_k: int = 4,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.30,
        max_generations: int = 300,
        stagnation_limit: int = 8,
        require_all_symbols: bool = False,
        seed: Optional[int] = 42,
    ) -> None:
        self.reel_lengths = reel_lengths
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.require_all_symbols = require_all_symbols
        self.rng = random.Random(seed)

        self.total_states = reel_lengths[0] * reel_lengths[1] * reel_lengths[2]
        self.target_scaled_score = TARGET_SCALED_RTP * self.total_states
        self.min_winning_states = math.ceil(MIN_WIN_RATE * self.total_states)

    def compute_loss(self, reels: Sequence[Sequence[int]]) -> Tuple[int, int, int]:
        """
        Computes integer loss for a chromosome.
        Returns (loss, winning_states, total_scaled_payout).

        loss == 0 iff:
          1. winning_states >= min_winning_states (Win Rate >= 55%)
          2. total_scaled_payout == target_scaled_score (Exact RTP == 0.95)
          3. All 5 symbols {0, 1, 2, 3, 4} appear if require_all_symbols is True
        """
        _, wins, scaled_score = evaluate_reels_fast(reels)
        win_shortfall = max(0, self.min_winning_states - wins)
        score_diff = abs(scaled_score - self.target_scaled_score)

        missing_count = 0
        if self.require_all_symbols:
            present = set(reels[0]) | set(reels[1]) | set(reels[2])
            missing_count = 5 - len(present)

        loss = win_shortfall * 10000 + score_diff * 10 + missing_count
        return loss, wins, scaled_score

    def _create_heuristic_reel(self, length: int) -> List[int]:
        """
        Initializes a single reel with short runs (pairs/triples) of symbols.
        Because all 2x2 and 3x3 winning patterns require vertically adjacent
        identical symbols, seeding reels with consecutive pairs of low-multiplier
        symbols (0 and 1) alongside higher symbols (2, 3, 4) helps the initial
        population quickly reach high win rates while keeping RTP near 0.95.
        """
        symbol_pool = [0, 0, 0, 0, 0, 1, 1, 2, 3, 4]
        reel: List[int] = []
        while len(reel) < length:
            sym = self.rng.choice(symbol_pool)
            run_len = self.rng.choice([1, 2, 3, 4])
            for _ in range(run_len):
                if len(reel) < length:
                    reel.append(sym)
        return reel

    def init_population(self) -> List[Chromosome]:
        """Generates the initial population of 3-reel chromosomes."""
        return [
            [self._create_heuristic_reel(L) for L in self.reel_lengths]
            for _ in range(self.population_size)
        ]

    def tournament_selection(
        self,
        population: Sequence[Chromosome],
        losses: Sequence[int],
    ) -> Chromosome:
        """Selects a parent chromosome using k-way tournament selection."""
        indices = self.rng.sample(range(len(population)), self.tournament_k)
        best_idx = min(indices, key=lambda idx: losses[idx])
        return copy.deepcopy(population[best_idx])

    def crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Performs crossover between two parent chromosomes.
        Combines reel-level swap and single-point slice crossover within reels.
        """
        if self.rng.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        child1: Chromosome = []
        child2: Chromosome = []

        for r_idx in range(3):
            r1 = parent1[r_idx]
            r2 = parent2[r_idx]
            length = len(r1)

            if self.rng.random() < 0.5:
                child1.append(r2[:])
                child2.append(r1[:])
            else:
                cut = self.rng.randint(1, length - 1)
                child1.append(r1[:cut] + r2[cut:])
                child2.append(r2[:cut] + r1[cut:])

        return child1, child2

    def mutate(self, chromosome: Chromosome) -> None:
        """
        Applies genetic mutation in-place:
        - Point mutation: changes a random position to a new symbol (0..4).
        - Pair mutation: writes two adjacent identical symbols to help form 2x2 blocks.
        """
        for r_idx, length in enumerate(self.reel_lengths):
            if self.rng.random() < self.mutation_rate:
                pos = self.rng.randrange(length)
                new_sym = self.rng.randrange(5)
                chromosome[r_idx][pos] = new_sym
                if self.rng.random() < 0.5:
                    chromosome[r_idx][(pos + 1) % length] = new_sym

    def local_search(
        self,
        chromosome: Chromosome,
        max_steps: int = 30,
    ) -> Tuple[Chromosome, int, int, int]:
        """
        Memetic Local Search (First-Improvement Hill Climbing):
        Explores both:
          1. Single-cell symbol flips: `reel[pos] = s`
          2. Adjacent-pair replacements: `reel[pos] = reel[pos+1] = s`
        Whenever a move strictly reduces `loss`, it updates the chromosome
        immediately and continues until `loss == 0` or a local minimum is reached.
        """
        best_chrom = copy.deepcopy(chromosome)
        best_loss, best_wins, best_score = self.compute_loss(best_chrom)

        if best_loss == 0:
            return best_chrom, best_loss, best_wins, best_score

        for _ in range(max_steps):
            improved = False
            positions = [
                (r_idx, pos)
                for r_idx, length in enumerate(self.reel_lengths)
                for pos in range(length)
            ]
            self.rng.shuffle(positions)

            for r_idx, pos in positions:
                length = self.reel_lengths[r_idx]
                next_pos = (pos + 1) % length
                orig_sym0 = best_chrom[r_idx][pos]
                orig_sym1 = best_chrom[r_idx][next_pos]

                # 1. Single-cell flip
                for cand_sym in range(5):
                    if cand_sym == orig_sym0:
                        continue
                    best_chrom[r_idx][pos] = cand_sym
                    cand_loss, cand_wins, cand_score = self.compute_loss(best_chrom)

                    if cand_loss < best_loss:
                        best_loss = cand_loss
                        best_wins = cand_wins
                        best_score = cand_score
                        improved = True
                        if best_loss == 0:
                            return best_chrom, best_loss, best_wins, best_score
                        break
                    else:
                        best_chrom[r_idx][pos] = orig_sym0

                if improved:
                    continue

                # 2. Adjacent-pair replacement: set (pos, pos+1) = (cand_sym, cand_sym)
                for cand_sym in range(5):
                    if cand_sym == orig_sym0 and cand_sym == orig_sym1:
                        continue
                    best_chrom[r_idx][pos] = cand_sym
                    best_chrom[r_idx][next_pos] = cand_sym
                    cand_loss, cand_wins, cand_score = self.compute_loss(best_chrom)

                    if cand_loss < best_loss:
                        best_loss = cand_loss
                        best_wins = cand_wins
                        best_score = cand_score
                        improved = True
                        if best_loss == 0:
                            return best_chrom, best_loss, best_wins, best_score
                        break
                    else:
                        best_chrom[r_idx][pos] = orig_sym0
                        best_chrom[r_idx][next_pos] = orig_sym1

            if not improved:
                break

        return best_chrom, best_loss, best_wins, best_score

    def solve(self) -> SolverResult:
        """
        Executes the Memetic Algorithm (GA + Local Search + Stagnation Restart)
        until an exact solution (`loss == 0`, meaning `RTP == 0.95` and
        `Win Rate >= 55%`) is found.
        """
        population = self.init_population()
        history: List[SolverHistoryStep] = []
        stagnation_counter = 0
        previous_best_loss = float("inf")

        for gen in range(1, self.max_generations + 1):
            # 1. Evaluate current population
            evals = [self.compute_loss(chrom) for chrom in population]
            ranked = sorted(
                zip(population, evals),
                key=lambda item: item[1][0],
            )

            population = [item[0] for item in ranked]
            losses = [item[1][0] for item in ranked]

            # 2. Apply Memetic Local Search to the top 4 elite chromosomes
            for elite_idx in range(min(4, len(population))):
                refined_chrom, refined_loss, refined_wins, refined_score = (
                    self.local_search(population[elite_idx], max_steps=30)
                )
                if refined_loss <= losses[elite_idx]:
                    population[elite_idx] = refined_chrom
                    losses[elite_idx] = refined_loss
                    evals[elite_idx] = (refined_loss, refined_wins, refined_score)

            # Re-sort after local search refinement
            ranked = sorted(
                zip(population, evals),
                key=lambda item: item[1][0],
            )
            population = [item[0] for item in ranked]
            evals = [item[1] for item in ranked]
            losses = [item[0] for item in evals]

            best_chrom = population[0]
            best_loss, best_wins, best_score = evals[0]
            best_rtp = best_score / (self.total_states * SCALE_FACTOR)
            best_wr = best_wins / self.total_states

            history.append(
                SolverHistoryStep(
                    generation=gen,
                    best_loss=best_loss,
                    best_rtp=best_rtp,
                    best_win_rate=best_wr,
                )
            )

            # Check termination criterion: Exact RTP == 0.95 and Win Rate >= 55%
            if best_loss == 0:
                return SolverResult(
                    reels=best_chrom,
                    generations=gen,
                    loss=best_loss,
                    total_states=self.total_states,
                    winning_states=best_wins,
                    total_scaled_payout=best_score,
                    history=history,
                )

            # Track stagnation and trigger population restart if stuck
            if best_loss < previous_best_loss:
                previous_best_loss = best_loss
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= self.stagnation_limit:
                population = self.init_population()
                stagnation_counter = 0
                previous_best_loss = float("inf")
                continue

            # 3. Build next generation via Elitism + Tournament Selection + Crossover + Mutation
            next_population: List[Chromosome] = [
                copy.deepcopy(population[i]) for i in range(self.elite_size)
            ]

            while len(next_population) < self.population_size:
                p1 = self.tournament_selection(population, losses)
                p2 = self.tournament_selection(population, losses)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                next_population.append(c1)
                if len(next_population) < self.population_size:
                    next_population.append(c2)

            population = next_population

        best_loss, best_wins, best_score = self.compute_loss(population[0])
        return SolverResult(
            reels=population[0],
            generations=self.max_generations,
            loss=best_loss,
            total_states=self.total_states,
            winning_states=best_wins,
            total_scaled_payout=best_score,
            history=history,
        )
