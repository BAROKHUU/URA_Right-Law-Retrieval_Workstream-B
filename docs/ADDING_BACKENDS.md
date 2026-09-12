# Adding a new dense index backend

1. Implement `BaseDenseIndex` in `src/legal_retrieval/indexing/dense.py` or another imported module.
2. Implement `build`, `search`, `save`, and `load`.
3. Register it with `register_dense_backend("my_backend", MyBackend)`.
4. Set `indexing.dense.backend: my_backend` in YAML.

This keeps the pipeline independent from FAISS and is the intended extension point for Qdrant, Milvus, Elasticsearch vector search, ScaNN, or custom ANN systems later.
