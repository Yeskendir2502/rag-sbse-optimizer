from rag_runner import set_seeds, run_nsga, DATASETS, MY_SEED


def main():
    set_seeds(MY_SEED)
    for ds in DATASETS:
        print(f"=== DATASET {ds} :: nsga2 ===")
        run_nsga(ds)
    print("NSGA-II done.")


if __name__ == "__main__":
    main()

