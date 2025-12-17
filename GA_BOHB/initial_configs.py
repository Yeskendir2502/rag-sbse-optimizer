from .config_space import CONFIG_SPACE

FIRST_CONFIG = {
    "splitter_type": "sentence",
    "chunk_size": 384,
    "chunk_overlap": 30,
    "embedding_model": "all-MiniLM-L6-v2",
    "normalize_embeddings": True,
    "index_type": "HNSW",
    "hnsw_M": 24,
    "hnsw_efSearch": 100,
    "top_k": 10,
    "hybrid_weight": 0.5,
}

INITIAL_CONFIGS = [
    FIRST_CONFIG,
    {
        "splitter_type": "token",
        "chunk_size": 256,
        "chunk_overlap": 0,
        "embedding_model": "all-MiniLM-L6-v2",
        "normalize_embeddings": True,
        "index_type": "HNSW",
        "hnsw_M": 16,
        "hnsw_efSearch": 50,
        "top_k": 5,
        "hybrid_weight": 0.0,
    },
    {
        "splitter_type": "sentence",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "mpnet",
        "normalize_embeddings": False,
        "index_type": "Flat",
        "hnsw_M": 32,
        "hnsw_efSearch": 150,
        "top_k": 20,
        "hybrid_weight": 0.8,
    },
]
