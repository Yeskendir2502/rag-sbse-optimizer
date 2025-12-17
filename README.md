# RAG Pipeline Hyperparameter Optimization using SBSE

**CS 454: AI-based Software Engineering - Final Project**

A search-based software engineering approach to optimize Retrieval-Augmented Generation (RAG) pipeline configurations using multi-objective evolutionary algorithms.

## Team Members
- **Yeskendir** - Pipeline architecture, system integration, deployment
- **Yerzhan** - BOHB (Bayesian Optimization with HyperBand) implementation
- **Dilnaz** - NSGA-II multi-objective optimization implementation
- **Alikhan** - Metrics evaluation and BEIR benchmarking

## Problem Statement
RAG systems involve numerous hyperparameters (chunking strategy, embedding model, index type, retrieval depth, hybrid weights) that significantly impact both retrieval quality (nDCG@10) and inference latency. Manual tuning is time-consuming and suboptimal. We formulate this as a multi-objective optimization problem:
- **Maximize**: nDCG@10 (retrieval accuracy)
- **Minimize**: Query latency (milliseconds)

## Project Structure
```
rag-sbse-optimizer/
├── rag_pipeline/           # Core RAG evaluation pipeline
│   ├── __init__.py
│   └── pipeline.py         # RAGEvaluator, chunking, embedding, search
├── NSGA2/                  # Multi-objective genetic algorithm
│   ├── __init__.py
│   ├── config_space.py     # Hyperparameter search space
│   ├── nsga2_core.py       # NSGA-II implementation
│   ├── representation.py   # Chromosome encoding/decoding
│   └── logging_utils.py    # Pareto front logging
├── GA_BOHB/                # Bayesian optimization module
│   ├── config_space.py
│   ├── initial_configs.py
│   └── optimizer/
│       ├── optimizer.py    # ConfigOptimizer with TPE sampler
│       ├── evaluator.py    # Config evaluation wrapper
│       ├── logger.py       # Trial result logging
│       └── utils.py        # Random config generation
├── evaluation/             # Metrics and analysis
│   └── compute_metrics.py  # nDCG@10, BEIR comparison
├── rag_runner.py           # Main experiment runner
├── run_nsga_only.py        # NSGA-II standalone runner
├── run_bohb_only.py        # BOHB standalone runner
├── analyze_pareto.py       # Results analysis script
├── demo_pipeline.py        # Interactive demo script
├── artifacts/              # Output results
│   ├── nsga2_fiqa.json     # NSGA-II Pareto front (FiQA)
│   ├── nsga2_scifact.json  # NSGA-II Pareto front (SciFact)
│   └── cache/              # Embedding/index cache
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
git clone https://github.com/Yeskendir2502/rag-sbse-optimizer.git
cd rag-sbse-optimizer

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Dependencies
```
beir>=2.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
optuna>=3.0.0
numpy>=1.24.0
tqdm>=4.65.0
pytest>=7.0.0
```

## Usage

### Quick Demo
Run a step-by-step demonstration of the RAG pipeline:
```bash
python demo_pipeline.py --dataset fiqa --config-file config_demo.json --subset-ratio 0.01
```

### Run Full Optimization

**NSGA-II (Multi-objective)**:
```bash
python run_nsga_only.py
```

**BOHB (Single-objective with latency penalty)**:
```bash
python run_bohb_only.py
```

**Both algorithms**:
```bash
python rag_runner.py
```

### Analyze Results
```bash
python analyze_pareto.py
```

Example output:
```
nsga2_fiqa.json: pareto_size=5 best_ndcg=0.4860 latency_ms=4271.44
nsga2_scifact.json: pareto_size=5 best_ndcg=0.7168 latency_ms=3756.32

BOHB_results_fiqa.json: trials=15 best_score=-1.1896 (ndcg=0.3644, latency_ms=3107.95)
BOHB_results_scifact.json: trials=15 best_score=-1.0929 (ndcg=0.7092, latency_ms=3604.17)
```

## Configuration Space
| Parameter | Values | Description |
|-----------|--------|-------------|
| splitter_type | token, sentence, paragraph | Document chunking strategy |
| chunk_size | 256, 384, 512 | Tokens per chunk |
| chunk_overlap | 0, 30, 50 | Overlap between chunks |
| embedding_model | MiniLM, BGE-base, MPNet | Sentence transformer model |
| normalize_embeddings | True, False | L2 normalization |
| index_type | HNSW, IVF, Flat | FAISS index type |
| hnsw_M | 16, 24, 32 | HNSW graph connectivity |
| hnsw_efSearch | 50, 100, 150 | HNSW search depth |
| top_k | 5, 10, 20, 50 | Retrieved documents |
| hybrid_weight | 0.0, 0.5, 0.8, 1.0 | BM25-dense interpolation |

## Results Summary

### NSGA-II Pareto Front (2 generations, pop=30)
| Dataset | Pareto Size | Best nDCG@10 | Latency (ms) |
|---------|-------------|--------------|--------------|
| FiQA | 5 | 0.4860 | 4271 |
| SciFact | 5 | 0.7168 | 3756 |

### BOHB Best Configurations (15 trials each)
| Dataset | Best nDCG@10 | Latency (ms) | Best Score |
|---------|--------------|--------------|------------|
| FiQA | 0.4988 | 4221 | -1.19 |
| SciFact | 0.7168 | 4197 | -1.09 |
