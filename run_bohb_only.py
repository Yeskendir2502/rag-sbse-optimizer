from rag_runner import set_seeds, run_bohb, DATASETS, MY_SEED


def main():
    set_seeds(MY_SEED)
    for ds in DATASETS:
        print(f"=== DATASET {ds} :: bohb ===")
        run_bohb(ds)
    print("BOHB done.")


if __name__ == "__main__":
    main()

