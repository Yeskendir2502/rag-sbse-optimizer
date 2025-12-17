CONFIG_SPACE = {
    "splitter_type": ["token", "sentence", "paragraph"],
    "chunk_size": [256, 384, 512],
    "chunk_overlap": [0, 30, 50],
    "embedding_model": ["all-MiniLM-L6-v2", "bge-base", "mpnet"],
    "normalize_embeddings": [True, False],
    "index_type": ["HNSW", "IVF", "Flat"],
    "hnsw_M": [16, 24, 32],
    "hnsw_efSearch": [50, 100, 150],
    "top_k": [5, 10, 20, 50],
    "hybrid_weight": [0.0, 0.5, 0.8, 1.0],
}
