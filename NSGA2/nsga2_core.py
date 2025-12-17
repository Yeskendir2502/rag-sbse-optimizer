"""
NSGA-II core implementation (minimization form).

    eval_fn(chromosome) -> tuple[float, float]

    f1 = -ndcg@10    (maximize ndcg)
    f2 = latency_ms  (minimize latency)

NSGA-II will minimize both f1 and f2.
"""

from __future__ import annotations

import math
import random
import multiprocessing.pool
from typing import Callable, List, Tuple

from NSGA2.representation import (
    Chromosome,
    random_chromosome,
    mutate_chromosome,
    crossover_chromosomes,
)


Fitness = Tuple[float, float]  # for 2 objectives


def dominates(f_i: Fitness, f_j: Fitness) -> bool:
    """
    Return True if f_i Pareto-dominates f_j for minimization.
    i dominates j if:
    - i is no worse in all objectives
    - i is strictly better in at least one objective
    """
    return (
        all(a <= b for a, b in zip(f_i, f_j))
        and any(a < b for a, b in zip(f_i, f_j))
    )


def fast_non_dominated_sort(fitnesses: List[Fitness]) -> List[List[int]]:
    """
    Deb's fast non-dominated sorting.
    Returns a list of fronts, each front is a list of indices.
    Front 0 = Pareto-optimal set in the current population.
    """
    n_solutions = len(fitnesses)
    S = [set() for _ in range(n_solutions)]  # set of solutions dominated by i
    n = [0 for _ in range(n_solutions)]      # number of solutions that dominate i
    fronts: List[List[int]] = [[]]

    # Compare all pairs
    for p, fp in enumerate(fitnesses):
        for q, fq in enumerate(fitnesses):
            if p == q:
                continue
            if dominates(fp, fq):
                S[p].add(q)
            elif dominates(fq, fp):
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # last front is empty, remove it
    fronts.pop()
    return fronts


def crowding_distance(front_indices: List[int], fitnesses: List[Fitness]) -> dict:
    """
    Compute crowding distance for each solution in a front.
    Returns dict: index -> distance.
    """
    if not front_indices:
        return {}

    n_obj = len(fitnesses[0])
    distance = {i: 0.0 for i in front_indices}

    for m in range(n_obj):
        # Sort by objective m
        sorted_front = sorted(front_indices, key=lambda i: fitnesses[i][m])
        f_min = fitnesses[sorted_front[0]][m]
        f_max = fitnesses[sorted_front[-1]][m]

        # Boundary points get infinite distance
        distance[sorted_front[0]] = math.inf
        distance[sorted_front[-1]] = math.inf

        if f_max == f_min:
            # All points have the same value for this objective
            continue

        for k in range(1, len(sorted_front) - 1):
            prev_f = fitnesses[sorted_front[k - 1]][m]
            next_f = fitnesses[sorted_front[k + 1]][m]
            distance[sorted_front[k]] += (next_f - prev_f) / (f_max - f_min)

    return distance


def crowded_better(
    i: int,
    j: int,
    rank: dict,
    crowd_dist: dict,
) -> bool:
    """
    Crowded comparison operator: True if i is better than j.
    Lower rank is better. For same rank, larger crowding distance is better.
    """
    if rank[i] < rank[j]:
        return True
    if rank[i] > rank[j]:
        return False
    return crowd_dist[i] > crowd_dist[j]


def tournament_select(
    population: List[Chromosome],
    fitnesses: List[Fitness],
    rank: dict,
    crowd_dist: dict,
) -> Chromosome:
    """
    Binary tournament selection using crowded comparison operator.
    """
    i = random.randrange(len(population))
    j = random.randrange(len(population))
    if crowded_better(i, j, rank, crowd_dist):
        return population[i]
    else:
        return population[j]


def run_nsga2(
    eval_fn: Callable[[Chromosome], Fitness],
    pop_size: int = 40,
    n_generations: int = 20,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.1,
    seed: int | None = None,
    workers: int | None = None,
):
    """
    Main NSGA-II loop.
    Returns:
        pareto_front_chromosomes, pareto_front_fitnesses,
        final_population, final_fitnesses
    """
    if seed is not None:
        random.seed(seed)

    # Initialize random population
    population: List[Chromosome] = [
        random_chromosome() for _ in range(pop_size)
    ]
    if workers and workers > 1:
        with multiprocessing.pool.Pool(processes=workers) as pool:
            fitnesses = pool.map(eval_fn, population)
    else:
        fitnesses = [eval_fn(ind) for ind in population]

    for gen in range(n_generations):
        #  Rank & crowding for current population 
        fronts = fast_non_dominated_sort(fitnesses)
        rank = {}
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
        crowd_dist = {}
        for front in fronts:
            crowd_dist.update(crowding_distance(front, fitnesses))

        # Create offspring via selection, crossover, mutation
        offspring: List[Chromosome] = []
        while len(offspring) < pop_size:
            parent1 = tournament_select(population, fitnesses, rank, crowd_dist)
            parent2 = tournament_select(population, fitnesses, rank, crowd_dist)

            if random.random() < crossover_prob:
                child1, child2 = crossover_chromosomes(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = mutate_chromosome(child1, mutation_prob)
            child2 = mutate_chromosome(child2, mutation_prob)

            offspring.append(child1)
            if len(offspring) < pop_size:
                offspring.append(child2)

        if workers and workers > 1:
            with multiprocessing.pool.Pool(processes=workers) as pool:
                offspring_fitnesses = pool.map(eval_fn, offspring)
        else:
            offspring_fitnesses = [eval_fn(ind) for ind in offspring]

        #  Combine parents & offspring (elitism) 
        combined = population + offspring
        combined_fitnesses = fitnesses + offspring_fitnesses

        # Non-dominated sorting on combined population
        fronts = fast_non_dominated_sort(combined_fitnesses)

        new_population: List[Chromosome] = []
        new_fitnesses: List[Fitness] = []

        for front in fronts:
            if len(new_population) + len(front) > pop_size:
                # Need to select only a part of this front,
                # based on crowding distance.
                cd = crowding_distance(front, combined_fitnesses)
                # Sort descending by crowding distance
                front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                remaining = pop_size - len(new_population)
                chosen = front_sorted[:remaining]
            else:
                chosen = front

            for idx in chosen:
                new_population.append(combined[idx])
                new_fitnesses.append(combined_fitnesses[idx])

            if len(new_population) == pop_size:
                break

        population, fitnesses = new_population, new_fitnesses

        # Optional: print generation summary
        best_front = fronts[0]
        best_vals = [combined_fitnesses[i] for i in best_front]
        print(
            f"[Gen {gen + 1}/{n_generations}] "
            f"| Pareto front size: {len(best_front)} "
            f"| Example fitness (first): {best_vals[0]}"
        )

    # Final Pareto front
    final_fronts = fast_non_dominated_sort(fitnesses)
    pareto_indices = final_fronts[0]
    pareto_chromosomes = [population[i] for i in pareto_indices]
    pareto_fitnesses = [fitnesses[i] for i in pareto_indices]

    return pareto_chromosomes, pareto_fitnesses, population, fitnesses
