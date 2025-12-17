from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Optional heavy deps; imported lazily where needed.
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - handled in runtime checks
    faiss = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - handled in runtime checks
    torch = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - handled in runtime checks
    np = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - handled in runtime checks
    SentenceTransformer = None  # type: ignore

# Config space used across optimizers
from GA_BOHB.config_space import CONFIG_SPACE


# Keys that change chunking/index artifacts (caches use these only).
CHUNK_KEYS = ["splitter_type", "chunk_size", "chunk_overlap"]
EMBED_KEYS = ["embedding_model", "normalize_embeddings"]
INDEX_KEYS = ["index_type", "hnsw_M", "hnsw_efSearch"]


def _require(module: Any, name: str, extra: str = ""):
    """Guarded import helper."""
    if module is None:
        hint = f" Install {name} first."
        if extra:
            hint = f"{hint} {extra}"
        raise ImportError(f"{name} is required but not installed.{hint}")


def _normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _has_cuda() -> bool:
    return torch is not None and torch.cuda.is_available()


def _hash_from_config(config: Dict[str, Any], keys: List[str]) -> str:
    payload = {k: config[k] for k in keys if k in config}
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _tokenize(text: str) -> List[str]:
    return [tok for tok in text.lower().split() if tok.strip()]


def _split_text(text: str, splitter: str, chunk_size: int, overlap: int) -> List[str]:
    if splitter == "paragraph":
        units = text.split("\n\n")
    elif splitter == "sentence":
        units = text.replace("\n", " ").split(". ")
    else:  # token
        units = text.split()

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    if splitter == "token":
        for i in range(0, len(units), step):
            chunk_tokens = units[i : i + chunk_size]
            if chunk_tokens:
                chunks.append(" ".join(chunk_tokens))
    else:
        buf: List[str] = []
        for unit in units:
            if not unit.strip():
                continue
            buf.append(unit.strip())
            joined = ". ".join(buf)
            if len(joined.split()) >= chunk_size:
                chunks.append(joined)
                if overlap > 0:
                    buf = buf[-1:]
                else:
                    buf = []
        if buf:
            chunks.append(". ".join(buf))
    return chunks


def _build_bm25(docs: List[str]):
    # Lightweight BM25 implementation (Okapi) with defaults.
    tokenized = [_tokenize(d) for d in docs]
    doc_freq: Dict[str, int] = {}
    for doc in tokenized:
        for term in set(doc):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    avgdl = sum(len(d) for d in tokenized) / max(1, len(tokenized))
    N = len(tokenized)
    k1, b = 1.5, 0.75

    def score(query: str) -> List[float]:
        q_tokens = _tokenize(query)
        scores = []
        for doc in tokenized:
            dl = len(doc)
            s = 0.0
            for term in q_tokens:
                f = doc.count(term)
                df = doc_freq.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                denom = f + k1 * (1 - b + b * dl / avgdl)
                s += idf * f * (k1 + 1) / denom
            scores.append(s)
        return scores

    return score


def _ndcg_at_k(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]], k: int = 10) -> float:
    """Compute nDCG@k over all queries."""
    def dcg(scores: List[float]) -> float:
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(scores))

    ndcgs: List[float] = []
    for qid, rels in qrels.items():
        retrieved = results.get(qid, {})
        # Sort retrieved doc IDs by score desc
        ranked = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)[:k]
        gains = [rels.get(doc_id, 0) for doc_id, _ in ranked]
        ideal = sorted(rels.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal)
        ndcg = dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcgs.append(ndcg)
    return sum(ndcgs) / max(1, len(ndcgs))


@dataclass
class CachedArtifacts:
    chunk_texts: List[str]
    chunk_doc_ids: List[str]
    embeddings: Any
    index: Any
    cache_hit: bool


class PipelineCache:
    def __init__(self, base_dir: str | Path = "artifacts/cache"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, dataset: str, config: Dict[str, Any], use_dummy: bool) -> Path:
        key = _hash_from_config(config, CHUNK_KEYS + EMBED_KEYS + INDEX_KEYS)
        mode = "dummy" if use_dummy else "real"
        return self.base_dir / dataset / f"{key}_{mode}"

    def load(self, dataset: str, config: Dict[str, Any], use_dummy: bool) -> CachedArtifacts | None:
        path = self._cache_path(dataset, config, use_dummy)
        meta_path = path / "meta.json"
        if not meta_path.exists():
            return None
        try:
            _require(np, "numpy")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected_hash = f"{_hash_from_config(config, CHUNK_KEYS + EMBED_KEYS + INDEX_KEYS)}_{'dummy' if use_dummy else 'real'}"
            if meta.get("config_hash") != expected_hash:
                return None
            chunk_path = path / "chunks.json"
            emb_path = path / "embeddings.npy"
            with open(chunk_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            embeddings = np.load(emb_path)
            index = None
            index_path = path / "index.faiss"
            if index_path.exists():
                try:
                    _require(faiss, "faiss-cpu")
                    index = faiss.read_index(str(index_path))
                except Exception:
                    index = None
            return CachedArtifacts(
                chunk_texts=chunks["texts"],
                chunk_doc_ids=chunks["doc_ids"],
                embeddings=embeddings,
                index=index,
                cache_hit=True,
            )
        except Exception:
            return None

    def save(
        self,
        dataset: str,
        config: Dict[str, Any],
        chunk_texts: List[str],
        chunk_doc_ids: List[str],
        embeddings,
        index,
        use_dummy: bool,
    ) -> None:
        path = self._cache_path(dataset, config, use_dummy)
        path.mkdir(parents=True, exist_ok=True)
        _require(np, "numpy")
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump({"texts": chunk_texts, "doc_ids": chunk_doc_ids}, f)
        np.save(path / "embeddings.npy", embeddings)
        if index is not None:
            _require(faiss, "faiss-cpu")
            faiss.write_index(index, str(path / "index.faiss"))
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config_hash": f"{_hash_from_config(config, CHUNK_KEYS + EMBED_KEYS + INDEX_KEYS)}_{'dummy' if use_dummy else 'real'}",
                    "created_at": time.time(),
                },
                f,
                indent=2,
            )


class RAGEvaluator:
    def __init__(
        self,
        dataset: str,
        cache_dir: str | Path = "artifacts/cache",
        use_dummy: bool = True,
    ):
        self.dataset = dataset.lower()
        self.cache = PipelineCache(cache_dir)
        self.use_dummy = use_dummy

        if not use_dummy:
            # Ensure heavy deps before running real evaluation.
            _require(np, "numpy", "pip install numpy")
            _require(faiss, "faiss-cpu", "pip install faiss-cpu")
            _require(SentenceTransformer, "sentence-transformers", "pip install sentence-transformers")

    def _load_dataset(self):
        if self.use_dummy:
            corpus = {
                "d1": {"title": "Economy update", "text": "Stocks rise as markets rally on strong earnings."},
                "d2": {"title": "Science news", "text": "New study confirms water on Mars in recent past."},
                "d3": {"title": "Finance tips", "text": "Diversify investments to manage risk effectively."},
            }
            queries = {
                "q1": "stock market earnings",
                "q2": "investments risk management",
            }
            qrels = {
                "q1": {"d1": 3},
                "q2": {"d3": 2},
            }
            return corpus, queries, qrels

        try:
            from beir.datasets.data_loader import GenericDataLoader  # type: ignore
            from beir import util as beir_util  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime check
            raise ImportError("beir is required for real evaluation. pip install beir") from exc

        data_dir = Path("data") / self.dataset
        if not data_dir.exists():
            url = beir_util.download_and_unzip(self.dataset, str(Path("data")))
            data_dir = Path(url)

        loader = GenericDataLoader(str(data_dir))
        corpus, queries, qrels = loader.load(split="test")
        return corpus, queries, qrels

    def _build_embeddings(self, texts: List[str], config: Dict[str, Any]):
        _require(np, "numpy")
        if self.use_dummy:
            rng = np.random.default_rng(seed=42)
            vecs = rng.normal(size=(len(texts), 64)).astype("float32")
        else:
            device = "cuda" if _has_cuda() else "cpu"
            model_name = {
                "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
                "bge-base": "BAAI/bge-base-en",
                "mpnet": "sentence-transformers/all-mpnet-base-v2",
            }.get(config["embedding_model"], config["embedding_model"])
            model = SentenceTransformer(model_name, device=device)
            vecs = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False).astype("float32")

        if config.get("normalize_embeddings", False):
            vecs = _normalize(vecs)
        return vecs

    def _build_index(self, embeddings, config: Dict[str, Any]):
        _require(faiss, "faiss-cpu")
        dim = embeddings.shape[1]
        metric = faiss.METRIC_INNER_PRODUCT
        index_type = config["index_type"].lower()

        if index_type == "flat":
            index = faiss.IndexFlatIP(dim)
        elif index_type == "hnsw":
            M = int(config.get("hnsw_M", 24))
            index = faiss.IndexHNSWFlat(dim, M, metric)
            index.hnsw.efSearch = int(config.get("hnsw_efSearch", 100))
        elif index_type == "ivf":
            nlist = max(8, int(math.sqrt(len(embeddings))))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric)
            index.train(embeddings)
        else:
            raise ValueError(f"Unsupported index_type '{config['index_type']}'")

        index.add(embeddings)
        # GPU acceleration if available
        if _has_cuda() and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                pass
        return index

    def _dense_search(self, index, query_vecs, top_k: int):
        scores, idxs = index.search(query_vecs, top_k)
        return scores, idxs

    def _aggregate_results(
        self,
        query_ids: List[str],
        bm25_scores: List[List[float]] | None,
        dense_scores: Any,
        dense_idxs: Any,
        chunk_doc_ids: List[str],
        hybrid_weight: float,
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for q_pos, qid in enumerate(query_ids):
            doc_scores: Dict[str, float] = {}

            if dense_scores is not None:
                for score, idx in zip(dense_scores[q_pos], dense_idxs[q_pos]):
                    if idx < 0 or idx >= len(chunk_doc_ids):
                        continue
                    doc_id = chunk_doc_ids[idx]
                    doc_scores[doc_id] = max(doc_scores.get(doc_id, -1e9), float(score))

            if bm25_scores is not None:
                for idx, s in enumerate(bm25_scores[q_pos]):
                    doc_id = chunk_doc_ids[idx]
                    combined = hybrid_weight * s + (1 - hybrid_weight) * doc_scores.get(doc_id, 0.0)
                    doc_scores[doc_id] = combined

            results[qid] = doc_scores
        return results

    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure config keys exist (fallback to defaults).
        for key in CONFIG_SPACE:
            if key not in config:
                raise ValueError(f"Missing config key '{key}'")

        t0 = time.time()
        corpus, queries, qrels = self._load_dataset()
        t_after_load = time.time()
        chunk_sec = 0.0
        embed_sec = 0.0
        index_sec = 0.0

        cache_hit = False
        cached = self.cache.load(self.dataset, config, self.use_dummy)
        if cached:
            chunk_texts = cached.chunk_texts
            chunk_doc_ids = cached.chunk_doc_ids
            embeddings = cached.embeddings
            index = cached.index
            cache_hit = True
            if index is None and not self.use_dummy:
                t_start_index = time.time()
                index = self._build_index(embeddings, config)
                index_sec = time.time() - t_start_index
        else:
            # Chunk corpus
            t_start_chunk = time.time()
            chunk_texts: List[str] = []
            chunk_doc_ids: List[str] = []
            for doc_id, doc in corpus.items():
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
            chunk_sec = time.time() - t_start_chunk
            if not chunk_texts:
                raise ValueError("No chunks produced; check chunk_size and splitter settings.")
            t_start_embed = time.time()
            embeddings = self._build_embeddings(chunk_texts, config)
            embed_sec = time.time() - t_start_embed
            index = None
            if not self.use_dummy:
                t_start_index = time.time()
                index = self._build_index(embeddings, config)
                index_sec = time.time() - t_start_index
            self.cache.save(self.dataset, config, chunk_texts, chunk_doc_ids, embeddings, index, self.use_dummy)

        start = time.time()
        # Encode queries
        if self.use_dummy:
            _require(np, "numpy")
            rng = np.random.default_rng(seed=123)
            query_vecs = rng.normal(size=(len(queries), embeddings.shape[1])).astype("float32")
        else:
            model_name = {
                "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
                "bge-base": "BAAI/bge-base-en",
                "mpnet": "sentence-transformers/all-mpnet-base-v2",
            }.get(config["embedding_model"], config["embedding_model"])
            model = SentenceTransformer(model_name)
            query_vecs = model.encode(
                list(queries.values()),
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=False,
            ).astype("float32")
            if config.get("normalize_embeddings", False):
                query_vecs = _normalize(query_vecs)

        query_sec = time.time() - start

        top_k = int(config.get("top_k", 10))
        top_k = max(1, min(top_k, len(chunk_texts)))
        if self.use_dummy:
            sims = query_vecs @ embeddings.T
            top_idxs = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
            dense_idxs = top_idxs
            dense_scores = np.take_along_axis(sims, top_idxs, axis=1)
        else:
            dense_scores, dense_idxs = self._dense_search(index, query_vecs, top_k)

        search_sec = time.time() - start

        bm25_scores = None
        hybrid_weight = float(config.get("hybrid_weight", 0.0))
        if hybrid_weight > 0.0:
            t_bm = time.time()
            scorer = _build_bm25(chunk_texts)
            bm25_scores = [scorer(q) for q in queries.values()]
            search_sec += time.time() - t_bm

        t_agg = time.time()
        results = self._aggregate_results(
            query_ids=list(queries.keys()),
            bm25_scores=bm25_scores,
            dense_scores=dense_scores,
            dense_idxs=dense_idxs,
            chunk_doc_ids=chunk_doc_ids,
            hybrid_weight=hybrid_weight,
        )
        aggregate_sec = time.time() - t_agg
        t_ndcg = time.time()
        ndcg = _ndcg_at_k(qrels, results, k=10)
        ndcg_sec = time.time() - t_ndcg
        total_sec = time.time() - t0
        latency_ms = (time.time() - start) * 1000.0

        return {
            "ndcg": ndcg,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "n_chunks": len(chunk_texts),
            "chunk_sec": chunk_sec,
            "embed_sec": embed_sec,
            "index_sec": index_sec,
            "query_sec": query_sec,
            "search_sec": search_sec,
            "aggregate_sec": aggregate_sec,
            "ndcg_sec": ndcg_sec,
            "total_sec": total_sec,
        }


def evaluate_config(
    config: Dict[str, Any],
    dataset: str = "fiqa",
    use_dummy: bool = True,
    cache_dir: str | Path = "artifacts/cache",
    return_details: bool = False,
) -> Tuple[float, float] | Dict[str, Any]:
    """
    Public entry point used by optimizers.

    Returns (ndcg, latency_ms) by default to preserve compatibility.
    When return_details=True, returns a dict with additional metadata.
    """
    evaluator = RAGEvaluator(dataset=dataset, cache_dir=cache_dir, use_dummy=use_dummy)
    result = evaluator.run(config)
    if return_details:
        return result
    return result["ndcg"], result["latency_ms"]

