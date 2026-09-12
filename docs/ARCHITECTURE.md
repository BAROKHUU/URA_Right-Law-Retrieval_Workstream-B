# Architecture mapping to the supplied diagram

## 1. Indexing layer

- Preprocess / unit representation: `corpus.py`, `text.py`
- Hierarchical units and fixed-length chunks: `corpus.py`
- Sparse BM25 index: `indexing/sparse.py`
- Dense encoder and index: `embedding.py`, `indexing/dense.py`
- Index lifecycle / manifest: `indexing/manager.py`

## 2. Retrieval layer

- Query normalization / static expansion / metadata filters: `query.py`, `pipeline.py`
- Sparse retrieval: BM25
- Dense retrieval: FAISS or NumPy backend
- Hybrid candidate fusion: `fusion.py`

## 3. Ranking layer

- Candidate set: `pipeline.py`
- Optional one/multiple CrossEncoder rerankers: `rerank.py`
- Final ranking, deduplication, top-K: `pipeline.py`

## 4. API layer

- Request validation / retrieval / response formatting: `api.py`

## 5. Evaluation layer

- Benchmark loading, configurable metrics, per-query results: `evaluation.py`, `experiment.py`

## Experiment runner

- Single hypothesis: `scripts/run_experiment.py`
- Multiple hypotheses: `scripts/run_suite.py`
- Comparison CSV: `scripts/compare_experiments.py`

The major components are deliberately behind configuration switches/interfaces so a new hypothesis normally requires a YAML override rather than pipeline changes.
