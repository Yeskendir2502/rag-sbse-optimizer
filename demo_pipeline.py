import argparse
import json
import time
from pathlib import Path

import numpy as np  # type: ignore

from rag_pipeline.pipeline import (
    RAGEvaluator,
    _split_text,
    _ndcg_at_k,
    _build_bm25,
    SentenceTransformer,
    _normalize,
)


def load_config(path: str | None) -> dict:
    if path is None:
        # Fallback demo config (same as config_demo.json)
        return {
            "splitter_type": "paragraph",
            "chunk_size": 256,
            "chunk_overlap": 0,
            "embedding_model": "bge-base",
            "normalize_embeddings": True,
            "index_type": "HNSW",
            "hnsw_M": 24,
            "hnsw_efSearch": 100,
            "top_k": 10,
            "hybrid_weight": 0.5,
        }
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def subset_dataset(
    corpus: dict,
    queries: dict,
    qrels: dict,
    ratio: float,
) -> tuple[dict, dict, dict]:
    """Take ~ratio of docs and keep only queries with at least one relevant doc."""
    n_docs = len(corpus)
    n_keep = max(1, int(n_docs * ratio))
    doc_ids = sorted(corpus.keys())[:n_keep]
    sub_corpus = {d: corpus[d] for d in doc_ids}

    sub_qrels = {}
    sub_queries = {}
    for qid, rels in qrels.items():
        filtered = {did: rel for did, rel in rels.items() if did in sub_corpus}
        if filtered:
            sub_qrels[qid] = filtered
            sub_queries[qid] = queries[qid]

    if not sub_qrels:
        print("WARNING: subset produced no qrels; falling back to full dataset.")
        return corpus, queries, qrels

    print(
        f"  Subset: {len(sub_corpus)} docs ({n_keep}/{n_docs}), "
        f"{len(sub_queries)} queries with non-empty qrels"
    )
    return sub_corpus, sub_queries, sub_qrels


def main():
    parser = argparse.ArgumentParser(description="Step-by-step RAG pipeline demo (real BEIR, subset)")
    parser.add_argument("--dataset", choices=["fiqa", "scifact"], default="fiqa")
    parser.add_argument("--config-file", type=str, default=None, help="Path to JSON config")
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=0.01,
        help="Fraction of documents to keep for the demo (e.g., 0.01 = 1%)",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)

    print("=== Demo RAG pipeline (real BEIR, subset) ===")
    print(f"Dataset: {args.dataset}")
    print(f"Subset ratio: {args.subset_ratio:.4f}")
    print("Config:")
    print(json.dumps(config, indent=2))
    print()

    # Real evaluator (no dummy)
    evaluator = RAGEvaluator(dataset=args.dataset, use_dummy=False)

    t0 = time.time()

    # --- Step 1: load dataset ---
    print("Step 1: Loading full BEIR dataset...")
    t_load0 = time.time()
    corpus, queries, qrels = evaluator._load_dataset()
    t_load1 = time.time()
    print(f"  Loaded {len(corpus)} documents, {len(queries)} queries")
    some_doc_id = next(iter(corpus.keys()))
    some_query_id = next(iter(queries.keys()))
    print(f"  Example doc id: {some_doc_id}, title: {corpus[some_doc_id].get('title', '')[:60]}...")
    print(f"  Example query id: {some_query_id}, text: {queries[some_query_id][:60]}...")
    print(f"  Load time: {(t_load1 - t_load0)*1000:.1f} ms\n")

    # --- Step 1b: subset to ~1% for the demo ---
    print("Step 1b: Subsetting corpus and queries for the demo...")
    sub_corpus, sub_queries, sub_qrels = subset_dataset(
        corpus, queries, qrels, ratio=args.subset_ratio
    )
    print()

    # --- Step 2: chunk corpus ---
    print("Step 2: Chunking corpus (on subset)...")
    t_chunk0 = time.time()
    chunk_texts: list[str] = []
    chunk_doc_ids: list[str] = []
    for doc_id, doc in sub_corpus.items():
        text = f"{doc.get('title', '')}. {doc.get('text', '')}".strip()
        chunks = _split_text(
            text=text,
            splitter=config["splitter_type"],
            chunk_size=int(config["chunk_size"]),
            overlap=int(config["chunk_overlap"]),
        )
        for c in chunks:
            chunk_texts.append(c)
            chunk_doc_ids.append(doc_id)
    t_chunk1 = time.time()
    print(f"  Produced {len(chunk_texts)} chunks from {len(sub_corpus)} docs")
    if chunk_texts:
        print(f"  Example chunk: {chunk_texts[0][:80]}...")
    print(f"  Chunking time: {(t_chunk1 - t_chunk0)*1000:.1f} ms\n")

    # --- Step 3: build embeddings ---
    print("Step 3: Building dense embeddings for chunks...")
    t_emb0 = time.time()
    embeddings = evaluator._build_embeddings(chunk_texts, config)
    t_emb1 = time.time()
    print(f"  Embeddings shape: {getattr(embeddings, 'shape', 'unknown')}")
    print(f"  Embedding time: {(t_emb1 - t_emb0)*1000:.1f} ms\n")

    # --- Step 4: build index & encode queries ---
    print("Step 4: Building FAISS index and encoding queries...")
    t_lat0 = time.time()  # start of latency window (query->results)

    # Build index
    t_idx0 = time.time()
    index = evaluator._build_index(embeddings, config)
    t_idx1 = time.time()
    print(f"  Index built with type={config['index_type']} in {(t_idx1 - t_idx0)*1000:.1f} ms")

    # Encode queries with same encoder family
    model_name = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "bge-base": "BAAI/bge-base-en",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
    }.get(config["embedding_model"], config["embedding_model"])
    st_model = SentenceTransformer(model_name)
    t_q0 = time.time()
    query_ids = list(sub_queries.keys())
    query_texts = [sub_queries[qid] for qid in query_ids]
    query_vecs = st_model.encode(
        query_texts,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
    ).astype("float32")
    if config.get("normalize_embeddings", False):
        query_vecs = _normalize(query_vecs)
    t_q1 = time.time()
    print(f"  Encoded {len(query_ids)} queries in {(t_q1 - t_q0)*1000:.1f} ms\n")

    # --- Step 5: dense (and hybrid) search ---
    print("Step 5: Running dense (and optional hybrid) search...")
    top_k = int(config.get("top_k", 10))
    top_k = max(1, min(top_k, len(chunk_texts)))

    t_search0 = time.time()
    dense_scores, dense_idxs = evaluator._dense_search(index, query_vecs, top_k)
    t_search1 = time.time()

    hybrid_weight = float(config.get("hybrid_weight", 0.0))
    bm25_scores = None
    if hybrid_weight > 0.0:
        print(f"  Hybrid mode: adding BM25 scores with weight = {hybrid_weight}")
        t_bm0 = time.time()
        scorer = _build_bm25(chunk_texts)
        bm25_scores = [scorer(q) for q in query_texts]
        t_bm1 = time.time()
        print(f"  BM25 scoring time: {(t_bm1 - t_bm0)*1000:.1f} ms")
    else:
        print("  Dense-only mode (no BM25).")

    print(f"  Dense search time: {(t_search1 - t_search0)*1000:.1f} ms\n")

    # --- Step 6: aggregate and compute metrics ---
    print("Step 6: Aggregating results and computing nDCG@10...")
    t_agg0 = time.time()
    results = evaluator._aggregate_results(
        query_ids=query_ids,
        bm25_scores=bm25_scores,
        dense_scores=dense_scores,
        dense_idxs=dense_idxs,
        chunk_doc_ids=chunk_doc_ids,
        hybrid_weight=hybrid_weight,
    )
    t_agg1 = time.time()

    t_ndcg0 = time.time()
    ndcg = _ndcg_at_k(sub_qrels, results, k=10)
    t_ndcg1 = time.time()

    t_lat1 = time.time()
    total_pipeline_time = time.time() - t0
    latency_ms = (t_lat1 - t_lat0) * 1000.0

    print(f"  Example retrieved docs for first query:")
    first_qid = query_ids[0]
    ranked_docs = sorted(results[first_qid].items(), key=lambda x: x[1], reverse=True)[:5]
    for doc_id, score in ranked_docs:
        print(f"    {doc_id}: score={score:.4f}")
    print(f"  Aggregation time: {(t_agg1 - t_agg0)*1000:.1f} ms")
    print(f"  nDCG@10: {ndcg:.4f} (computed in {(t_ndcg1 - t_ndcg0)*1000:.1f} ms)")
    print(f"  Total pipeline wall-clock: {total_pipeline_time*1000:.1f} ms")
    print(f"  Measured latency_ms (query->results): {latency_ms:.1f} ms\n")

    # --- Final: fitness vector for NSGA-II ---
    f1 = -ndcg
    f2 = latency_ms
    print("Final metrics returned to optimizers:")
    print(f"  ndcg = {ndcg:.4f}")
    print(f"  latency_ms = {latency_ms:.1f}")
    print()
    print("Fitness vector for NSGA-II (what mutation/selection sees):")
    print(f"  (-ndcg, latency_ms) = ({f1:.4f}, {f2:.1f})")


if __name__ == "__main__":
    main()
