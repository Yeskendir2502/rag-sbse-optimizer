"""
Metrics computation module for RAG pipeline evaluation.
Computes nDCG@10 and compares with BEIR benchmark baselines.
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Any


def dcg_at_k(relevances: List[float], k: int = 10) -> float:
    # compute discounted cumulative gain at cutoff k
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))


def ndcg_at_k(qrels: Dict[str, Dict[str, int]], 
              results: Dict[str, Dict[str, float]], 
              k: int = 10) -> float:
    """
    Compute nDCG@k over all queries.
    
    qrels: ground truth {qid: {docid: relevance}}
    results: retrieved {qid: {docid: score}}
    """
    ndcgs = []
    for qid, rels in qrels.items():
        retrieved = results.get(qid, {})
        # sort by score descending
        ranked = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)[:k]
        gains = [rels.get(doc_id, 0) for doc_id, _ in ranked]
        ideal = sorted(rels.values(), reverse=True)[:k]
        ideal_dcg = dcg_at_k(ideal, k)
        ndcg = dcg_at_k(gains, k) / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcgs.append(ndcg)
    return sum(ndcgs) / max(1, len(ndcgs))


# BEIR baseline results for comparison (from official leaderboard)
BEIR_BASELINES = {
    "fiqa": {
        "BM25": 0.236,
        "DPR": 0.295,
        "ANCE": 0.295,
        "TAS-B": 0.300,
        "ColBERT": 0.317,
    },
    "scifact": {
        "BM25": 0.665,
        "DPR": 0.642,
        "ANCE": 0.672,
        "TAS-B": 0.707,
        "ColBERT": 0.671,
    },
}


def compare_with_beir(dataset: str, our_ndcg: float) -> Dict[str, Any]:
    # compare our results with beir baselines
    baselines = BEIR_BASELINES.get(dataset, {})
    comparison = {}
    for method, score in baselines.items():
        improvement = ((our_ndcg - score) / score) * 100 if score > 0 else 0
        comparison[method] = {
            "baseline_ndcg": score,
            "our_ndcg": our_ndcg,
            "improvement_pct": round(improvement, 2),
            "better": our_ndcg > score,
        }
    return comparison


def load_nsga2_results(path: Path) -> Dict[str, Any]:
    # load nsga2 pareto front from json
    data = json.loads(path.read_text())
    fitness = data.get("fitness", [])
    if not fitness:
        return None
    # f1 = -ndcg, so we negate to get actual ndcg
    best_idx = min(range(len(fitness)), key=lambda i: fitness[i][0])
    f1, f2 = fitness[best_idx]
    return {
        "pareto_size": len(fitness),
        "best_ndcg": -f1,
        "latency_ms": f2,
        "all_points": [(-f[0], f[1]) for f in fitness],
    }


def load_bohb_results(path: Path) -> Dict[str, Any]:
    # load bohb trial results from json
    try:
        entries = json.loads(path.read_text())
    except Exception:
        return None
    if not entries:
        return None
    best_idx = max(range(len(entries)), key=lambda i: entries[i].get("nDCG", 0.0))
    best = entries[best_idx]
    return {
        "trials": len(entries),
        "best_ndcg": best.get("nDCG", 0.0),
        "latency_ms": best.get("latency", 0.0),
        "best_config": best.get("config", {}),
    }


def generate_latex_table(results: Dict[str, Dict[str, Any]]) -> str:
    # generate latex table for report
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparison with BEIR Baselines (nDCG@10)}",
        r"\begin{tabular}{l|cc|cc}",
        r"\hline",
        r"Method & \multicolumn{2}{c|}{FiQA} & \multicolumn{2}{c}{SciFact} \\",
        r" & nDCG & $\Delta$\% & nDCG & $\Delta$\% \\",
        r"\hline",
    ]
    
    for method in ["BM25", "DPR", "ANCE", "TAS-B", "ColBERT"]:
        fiqa_base = BEIR_BASELINES["fiqa"].get(method, 0)
        scifact_base = BEIR_BASELINES["scifact"].get(method, 0)
        lines.append(f"{method} & {fiqa_base:.3f} & - & {scifact_base:.3f} & - \\\\")
    
    lines.append(r"\hline")
    
    # add our results
    if "fiqa" in results and "scifact" in results:
        fiqa_ndcg = results["fiqa"]["best_ndcg"]
        scifact_ndcg = results["scifact"]["best_ndcg"]
        fiqa_imp = ((fiqa_ndcg - 0.317) / 0.317) * 100  # vs ColBERT
        scifact_imp = ((scifact_ndcg - 0.707) / 0.707) * 100  # vs TAS-B
        lines.append(f"Ours (NSGA-II) & {fiqa_ndcg:.3f} & {fiqa_imp:+.1f} & {scifact_ndcg:.3f} & {scifact_imp:+.1f} \\\\")
    
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    """Generate comparison report."""
    artifacts = Path("artifacts")
    
    print("=" * 60)
    print("RAG Pipeline Optimization Results vs BEIR Baselines")
    print("=" * 60)
    
    results = {}
    
    for dataset in ["fiqa", "scifact"]:
        print(f"\n### {dataset.upper()} ###\n")
        
        # load nsga2 results
        nsga_path = artifacts / f"nsga2_{dataset}.json"
        if nsga_path.exists():
            nsga = load_nsga2_results(nsga_path)
            if nsga:
                results[dataset] = nsga
                print(f"NSGA-II: nDCG@10 = {nsga['best_ndcg']:.4f}, Latency = {nsga['latency_ms']:.1f}ms")
                print(f"  Pareto front size: {nsga['pareto_size']}")
                
                # compare with baselines
                comparison = compare_with_beir(dataset, nsga["best_ndcg"])
                print("\n  Comparison with BEIR baselines:")
                for method, cmp in comparison.items():
                    status = "✓ BETTER" if cmp["better"] else "✗ WORSE"
                    print(f"    vs {method}: {cmp['baseline_ndcg']:.3f} -> {cmp['our_ndcg']:.4f} ({cmp['improvement_pct']:+.1f}%) {status}")
        
        # load bohb results
        bohb_path = Path("GA_BOHB/optimizer/results") / f"BOHB_results_{dataset}.json"
        if bohb_path.exists():
            bohb = load_bohb_results(bohb_path)
            if bohb:
                print(f"\nBOHB: nDCG@10 = {bohb['best_ndcg']:.4f}, Latency = {bohb['latency_ms']:.1f}ms")
                print(f"  Total trials: {bohb['trials']}")
    
    # generate latex table
    if results:
        print("\n" + "=" * 60)
        print("LaTeX Table for Report:")
        print("=" * 60)
        print(generate_latex_table(results))


if __name__ == "__main__":
    main()
