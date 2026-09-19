# Legal Retrieval Experiments

A YAML-driven framework for evaluating legal-text retrieval configurations. The current scope is the core research workflow:

- hierarchical legal units and fixed-length chunks;
- BM25, dense, and hybrid retrieval;
- Sentence Transformers embedding models;
- RRF and weighted fusion;
- optional cross-encoder reranking;
- single-query inspection;
- multi-query evaluation and experiment comparison;
- immutable, content-addressed index artifacts.

For the complete H1-H9 matrix, commands, temporal semantics, benchmark setup,
and result interpretation, use the canonical
[`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md). This README covers only
installation and framework concepts.

## 1. Setup

### Linux/macOS

```bash
cd /path/to/legal-retrieval-experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[all]"
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[all]"
```

For a BM25-only environment:

```bash
python -m pip install -e .
```

Verify the installation:

```bash
python -m pip check
pytest -q
```

Dense retrieval and reranking require Sentence Transformers, PyTorch, and FAISS. Model weights are downloaded from Hugging Face the first time a model is used.

## Reproducible Docker environment

Development installs may continue to use `pip install -e ".[all]"`. For a fixed experiment environment, use the included `Dockerfile` and `requirements.lock`:

```bash
docker build -t legal-retrieval:locked .
docker run --rm -it legal-retrieval:locked \
  python scripts/run_experiment.py --config configs/examples/h1_sparse_hierarchical.yaml
```

The container sets `LEGAL_RETRIEVAL_PROJECT_ROOT=/app`. Outside Docker, the framework first checks this environment variable and otherwise searches upward for the repository markers. It no longer assumes that the package always lives exactly two parent directories below the project root.

## 2. Repository layout

```text
legal-retrieval-experiments/
├── configs/
│   ├── base.yaml                 # Shared defaults
│   ├── hypothesis_template.yaml  # Template for a new experiment config
│   ├── examples/                 # H1, H2, H3, ... configurations
│   └── suites/                   # Lists of configs to run together
├── data/
│   ├── enriched/                 # Input legal corpus; never modified by the pipeline
│   ├── benchmarks/               # Query sets and relevance labels
│   └── README.md
├── src/legal_retrieval/
│   ├── config.py                 # YAML inheritance and validation
│   ├── corpus.py                 # Corpus loading and record construction
│   ├── embedding.py              # Dense encoder
│   ├── query.py                  # Query normalization, expansion, and filters
│   ├── pipeline.py               # Retrieval, fusion, reranking, and final ranking
│   ├── evaluation.py             # Benchmark loading and metrics
│   ├── experiment.py             # Experiment execution and metadata
│   └── indexing/                 # BM25, FAISS/NumPy, and index management
├── scripts/
│   ├── build_index.py            # Build or reuse an index
│   ├── retrieve.py               # Run one query with one config
│   ├── run_experiment.py         # Run a complete query set
│   ├── run_suite.py              # Run multiple configs
│   └── compare_experiments.py    # Export run metrics to CSV
├── artifacts/
│   ├── indexes/                  # Immutable indexes grouped by SHA-256 signature
│   ├── runs/                     # Per-run outputs
│   └── reports/                  # Cross-experiment comparison tables
├── tests/
├── docs/
└── pyproject.toml
```

## 3. Configuration workflow

`configs/base.yaml` contains the shared defaults. Each experiment config should extend a base or another experiment and override only the variables being tested.

Create a new config:

```bash
cp configs/hypothesis_template.yaml configs/examples/h6_multilingual_e5.yaml
```

Example dense configuration:

```yaml
extends: ../base.yaml

hypothesis:
  id: H6
  description: "Evaluate multilingual-e5-large on hierarchical legal units."
  variables:
    - retrieval.method
    - indexing.dense.model_name
    - indexing.dense.query_prefix
    - indexing.dense.document_prefix
  notes: "Keep representation and candidate depth fixed."

indexing:
  output_dir: artifacts/indexes/h6_multilingual_e5
  sparse:
    enabled: false
  dense:
    enabled: true
    backend: faiss
    index_type: FlatIP
    model_name: intfloat/multilingual-e5-large
    # Pin a Hugging Face commit hash for final reproducible runs.
    revision: null
    device: auto
    batch_size: 8
    normalize_embeddings: true
    query_prefix: "query: "
    document_prefix: "passage: "

retrieval:
  method: dense
  dense_top_k: 50
  candidate_top_k: 50

fusion:
  method: none

ranking:
  final_top_k: 10
```

Configuration guidelines:

- Give every experiment a stable, unique `hypothesis.id`.
- `hypothesis.variables` is an allow-list of scientific config paths that may differ from the config named by the experiment's immediate `extends`.
- The loader automatically computes `Declared changes` and `Actual changes`; an undeclared actual change raises an error before indexing/retrieval starts.
- `hypothesis.*` and `indexing.output_dir` are bookkeeping fields and are excluded from scientific diffing.
- A declaration may authorize a subtree, e.g. `reranking.models` covers `reranking.models[0].model_name`, `.revision`, `.weight`, etc.
- Keep unrelated settings fixed for a controlled comparison.
- For publication/final runs, set `runtime.reproducibility_mode: final` and pin every enabled Hugging Face model `revision` to its full 40-character commit SHA.
- Do not edit `artifacts/` while an index is being built.

The main configuration groups are:

| Section | Purpose |
|---|---|
| `corpus` | Input files and included legal-unit types |
| `representation` | Hierarchical records, fixed chunks, and indexed fields |
| `query_processing` | Normalization, static expansion, and metadata filters |
| `indexing.sparse` | BM25 settings |
| `indexing.dense` | Backend, embedding model, device, and batch size |
| `retrieval` | Retrieval method and candidate depths |
| `fusion` | RRF or weighted fusion |
| `reranking` | Optional cross-encoder rerankers |
| `ranking` | Final top-k and deduplication |
| `evaluation` | Query set, relevance matching, metrics, and k values |

See `configs/base.yaml` for the complete set of options.

## 4. Run a single query

Use `retrieve.py` for quick inspection. The corresponding content-addressed
index is built automatically when missing.

```bash
python scripts/retrieve.py \
  --config configs/examples/h1_sparse_hierarchical.yaml \
  --query "điều kiện cấp giấy chứng nhận quyền sử dụng đất" \
  --top-k 10 \
  > artifacts/reports/h1_sparse_hierarchical.json
```

See the [experiment guide](docs/EXPERIMENT_GUIDE.md) for every H1-H9 config,
metadata filters, temporal query explanations, suites, and heavy-model notes.

## 5. Run multiple queries

Store query sets under `data/benchmarks/` as JSONL: one JSON object per line.

### Query-only file

Use this format for retrieval inspection before relevance labels are available:

```jsonl
{"query_id":"q001","query":"điều kiện cấp giấy chứng nhận quyền sử dụng đất","filters":{}}
{"query_id":"q002","query":"thủ tục đăng ký biến động đất đai","filters":{}}
{"query_id":"q003","query":"thẩm quyền thu hồi đất","filters":{"doc_type":"law"}}
```

### Labeled query file

Add `relevant_unit_ids` when ground truth is available:

```jsonl
{"query_id":"q001","query":"điều kiện cấp giấy chứng nhận quyền sử dụng đất","relevant_unit_ids":["legal-unit-id-1","legal-unit-id-2"],"filters":{}}
{"query_id":"q002","query":"thủ tục đăng ký biến động đất đai","relevant_unit_ids":["legal-unit-id-3"],"filters":{}}
```

Keep `query_id` unique and stable across experiments. Version a query set instead of silently changing it, for example:

```text
data/benchmarks/
├── smoke_queries_v1.jsonl
├── dev_queries_v1.jsonl
├── dev_queries_v2.jsonl
└── test_queries_v1.jsonl
```

Recommended use:

- `smoke`: a small set for checking that a configuration runs correctly;
- `dev`: parameter selection and error analysis;
- `test`: final evaluation only.

Point the experiment config to the query set:

```yaml
evaluation:
  benchmark_path: data/benchmarks/dev_queries_v1.jsonl
  query_field: query
  query_id_field: query_id
  relevant_ids_field: relevant_unit_ids
  filters_field: filters
  match_by: source_unit
  metrics: [hit_rate, precision, recall, mrr, ndcg, map]
  k_values: [1, 3, 5, 10, 20]
  save_per_query: true
```

Run all queries using the selected configuration:

```bash
python scripts/run_experiment.py \
  --config configs/examples/h3_hybrid_rrf.yaml
```

Query-only rows are retrieved and saved without metrics. Rows with non-empty `relevant_unit_ids` are also evaluated.

## 6. Inspect experiment outputs

Each invocation of `run_experiment.py` creates a unique directory:

```text
artifacts/runs/<hypothesis-id>/<timestamp-and-uuid>/
├── resolved_config.yaml
├── run_metadata.json
├── run_summary.json
├── metrics.json
└── per_query.jsonl
```

Use the files as follows:

| File | Use |
|---|---|
| `per_query.jsonl` | Inspect ranked results and per-query metrics |
| `metrics.json` | Compare aggregate retrieval quality |
| `run_summary.json` | Check status, query counts, index signature, and duration |
| `resolved_config.yaml` | Recover the exact effective configuration |
| `run_metadata.json` | Check seed, Python, packages, platform, and Git revision |

`per_query.jsonl` is the primary error-analysis file. Each row contains the input query, filters, relevance labels, ranked results, result text, metadata, and metrics when labels exist.

## 7. Run and compare multiple configurations

Create a suite such as `configs/suites/my_suite.yaml`:

```yaml
continue_on_error: false
configs:
  - ../examples/h1_sparse_hierarchical.yaml
  - ../examples/h2_dense_hierarchical.yaml
  - ../examples/h3_hybrid_rrf.yaml
```

All listed configs should point to the same benchmark for a controlled comparison.

Run the suite:

```bash
python scripts/run_suite.py \
  --suite configs/suites/my_suite.yaml
```

Export run summaries to CSV:

```bash
python scripts/compare_experiments.py \
  --runs-dir artifacts/runs \
  --output artifacts/reports/dev_queries_v1_comparison.csv
```

## 8. Index storage and reuse

`indexing.output_dir` is an artifact namespace. The actual immutable index is stored under its SHA-256 signature:

```text
artifacts/indexes/h3_hybrid_rrf/
└── <sha256-signature>/
    ├── manifest.json
    ├── resolved_index_config.yaml
    ├── records.jsonl
    ├── sparse/
    │   └── bm25.pkl
    └── dense/
        ├── index.faiss
        └── ids.json
```

The signature covers corpus content, representation, sparse settings, and dense settings. Therefore:

- changing `model_name` or `revision` creates a new index;
- changing chunk size or overlap creates a new index;
- changing the corpus, prefixes, or normalization creates a new index;
- identical index configurations intentionally reuse the same artifact;
- completed indexes are never overwritten by a different configuration.

Artifacts created before content-addressed storage may still exist directly under `artifacts/indexes/<name>/`. The current code uses signature directories and does not automatically delete legacy artifacts.

## 9. Practical recommendations

- Start with H1 to validate the corpus and query format.
- Run one query with `retrieve.py` before launching a complete query set.
- Use a small smoke query set before a development or test set.
- Reduce `indexing.dense.batch_size` if dense indexing runs out of memory.
- Reduce `reranking.top_n` and `reranking.batch_size` for faster reranking tests.
- Pin Hugging Face model revisions for final experiments.
- Compare configurations on the same versioned query set.
- Use `per_query.jsonl` for error analysis, not only aggregate metrics.
