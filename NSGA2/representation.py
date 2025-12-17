"""
Chromosome representation and genetic operators for the RAG config space.

Chromosome = list[int], each int is an index into CONFIG_SPACE[param]
decode_chromosome -> config dict where each value is a 1-element list,
"""

import random
from typing import List, Dict, Any

from NSGA2.config_space import CONFIG_SPACE, GENE_ORDER


Chromosome = List[int]


def random_chromosome() -> Chromosome:
    """
    Sample a random chromosome using the CONFIG_SPACE & GENE_ORDER.
    """
    genes: Chromosome = []
    for param in GENE_ORDER:
        n_options = len(CONFIG_SPACE[param])
        genes.append(random.randrange(n_options))
    return genes


def decode_chromosome(chromosome: Chromosome) -> Dict[str, list]:
    """
    Convert a chromosome (indexes) into a config dict with values as single-element lists.
    """
    if len(chromosome) != len(GENE_ORDER):
        raise ValueError(
            f"Chromosome length {len(chromosome)} "
            f"does not match gene order length {len(GENE_ORDER)}"
        )

    config: Dict[str, list] = {}

    for gene_idx, param in enumerate(GENE_ORDER):
        idx = chromosome[gene_idx]
        options = CONFIG_SPACE[param]

        if not (0 <= idx < len(options)):
            raise ValueError(
                f"Gene index {idx} out of range for parameter '{param}'. "
                f"Expected in [0, {len(options) - 1}]"
            )

    
        config[param] = [options[idx]]

    return config


def mutate_chromosome(
    chromosome: Chromosome,
    mutation_prob: float = 0.1,
) -> Chromosome:
    """
    Simple mutation: for each gene, with probability mutation_prob, change it to a different valid option index.
    """
    new_chrom = chromosome[:]  # copy
    for i, param in enumerate(GENE_ORDER):
        if random.random() < mutation_prob:
            n_options = len(CONFIG_SPACE[param])
            current = new_chrom[i]

            if n_options == 1:
                # nothing to mutate, only one option
                continue

            choices = list(range(n_options))
            choices.remove(current)
            new_chrom[i] = random.choice(choices)
    return new_chrom


def crossover_chromosomes(
    parent1: Chromosome,
    parent2: Chromosome,
) -> (Chromosome, Chromosome):
    """
    One-point crossover. Returns two children.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    length = len(parent1)
    if length == 1:
        # nothing to cross over
        return parent1[:], parent2[:]

    # choose crossover point in [1, length-1]
    point = random.randrange(1, length)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


if __name__ == "__main__":
    # Small self-test
    random.seed(42)
    c = random_chromosome()
    print("Random chromosome:", c)
    print("Decoded config:", decode_chromosome(c))

    c_mut = mutate_chromosome(c, mutation_prob=0.5)
    print("Mutated chromosome:", c_mut)
    print("Mutated config:", decode_chromosome(c_mut))

    c1 = random_chromosome()
    c2 = random_chromosome()
    ch1, ch2 = crossover_chromosomes(c1, c2)
    print("Parent1:", c1)
    print("Parent2:", c2)
    print("Child1:", ch1)
    print("Child2:", ch2)
