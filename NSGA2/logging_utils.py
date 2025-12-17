import csv
from typing import List, Tuple

from representation import Chromosome, decode_chromosome
from nsga2_core import Fitness


def save_pareto_to_csv(
    filename: str,
    pareto_chroms: List[Chromosome],
    pareto_fits: List[Fitness],
) -> None:
    """
    Save Pareto front to CSV with both fitness values and decoded configs.
    """
    if not pareto_chroms:
        print("No Pareto solutions to save.")
        return

    example_cfg = decode_chromosome(pareto_chroms[0])
    config_keys = list(example_cfg.keys())

    fieldnames = ["f1", "f2"] + config_keys

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for chrom, fit in zip(pareto_chroms, pareto_fits):
            cfg = decode_chromosome(chrom)
            row = {"f1": fit[0], "f2": fit[1]}
            for k in config_keys:
                # cfg[k] is a 1-element list, store the actual value
                row[k] = cfg[k][0]
            writer.writerow(row)

    print(f"Saved Pareto front ({len(pareto_chroms)} configs) to {filename}")
