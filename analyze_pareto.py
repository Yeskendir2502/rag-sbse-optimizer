import json
from pathlib import Path


def summarize_pareto(path: Path):
    data = json.loads(path.read_text())
    fitness = data.get("fitness", [])
    if not fitness:
        return None
    best_idx = min(range(len(fitness)), key=lambda i: fitness[i][0])  # f1 = -ndcg
    f1, f2 = fitness[best_idx]
    ndcg = -f1
    latency_ms = f2
    return {
        "file": str(path),
        "pareto_size": len(fitness),
        "best_ndcg": ndcg,
        "latency_ms": latency_ms,
    }


def summarize_bohb(path: Path):
    """
    Summaries for BOHB result files.
    Files are JSON arrays of entries with keys: trial_id, config, nDCG, latency.
    We report:
      - trials: number of entries
      - best_score: max of ndcg - 0.0005 * latency (the objective used)
      - ndcg/latency for that best-score entry
      - best_ndcg_overall: highest ndcg seen (and its latency) for quick reference
    """
    try:
        entries = json.loads(path.read_text())
    except Exception:
        return None

    if not entries:
        return None

    def score(e):
        # Same objective used in ConfigOptimizer.bohb_objective
        return e.get("nDCG", 0.0) - 0.0005 * e.get("latency", 0.0)

    best_score_idx = max(range(len(entries)), key=lambda i: score(entries[i]))
    best_score_entry = entries[best_score_idx]
    best_ndcg_idx = max(range(len(entries)), key=lambda i: entries[i].get("nDCG", 0.0))
    best_ndcg_entry = entries[best_ndcg_idx]

    return {
        "file": str(path),
        "trials": len(entries),
        "best_score": score(best_score_entry),
        "best_score_ndcg": best_score_entry.get("nDCG", 0.0),
        "best_score_latency": best_score_entry.get("latency", 0.0),
        "best_ndcg": best_ndcg_entry.get("nDCG", 0.0),
        "best_ndcg_latency": best_ndcg_entry.get("latency", 0.0),
    }


def main():
    artifacts = Path("artifacts")
    reports = []
    for p in artifacts.glob("nsga2_*.json"):
        r = summarize_pareto(p)
        if r:
            reports.append(r)

    bohb_dir = Path("GA_BOHB") / "optimizer" / "results"
    bohb_reports = []
    for p in bohb_dir.glob("BOHB_results*.json"):
        r = summarize_bohb(p)
        if r:
            bohb_reports.append(r)

    if reports:
        for r in reports:
            print(
                f"{Path(r['file']).name}: pareto_size={r['pareto_size']} "
                f"best_ndcg={r['best_ndcg']:.4f} latency_ms={r['latency_ms']:.2f}"
            )
    else:
        print("No Pareto files found in artifacts/nsga2_*.json")

    if bohb_reports:
        print()  # spacer
        for r in bohb_reports:
            print(
                f"{Path(r['file']).name}: trials={r['trials']} "
                f"best_score={r['best_score']:.4f} "
                f"(ndcg={r['best_score_ndcg']:.4f}, latency_ms={r['best_score_latency']:.2f}) "
                f"best_ndcg={r['best_ndcg']:.4f} "
                f"(latency_ms={r['best_ndcg_latency']:.2f})"
            )
    else:
        print("No BOHB result files found in GA_BOHB/optimizer/results/BOHB_results*.json")


if __name__ == "__main__":
    main()

