# Running All H1-H9 Experiments

This is the canonical guide for selecting, running, and comparing every
hypothesis. Run all commands from the repository root.

All documentation prose is in English. Query strings remain in Vietnamese
because the corpus and the retrieval task are Vietnamese-language legal data.

## 1. Hypothesis matrix

| ID | Representation | Retrieval/ranking | Temporal | Research question |
|---|---|---|---|---|
| H1 | Independent Article/Clause/Point units | BM25 | No | Sparse baseline over legal units |
| H2 | Independent Article/Clause/Point units | BGE-M3 + FAISS | No | Does dense retrieval outperform BM25? |
| H3 | Independent Article/Clause/Point units | BM25 + BGE-M3 + RRF | No | Does hybrid retrieval improve recall? |
| H4 | Same as H3 | Hybrid + BGE reranker | No | Does reranking improve result ordering? |
| H5 | 220-word chunks with 40-word overlap | Hybrid + RRF | No | Fixed chunks versus legal units |
| H6 | 220-word chunks with 40-word overlap | BM25 | No | Chunking comparison under the same BM25 retriever |
| H7 | Independent Clause/Point children | BM25 | Yes | Child baseline without parent context |
| H8 | Child + Article heading | BM25 | Yes | Does a parent heading help retrieve the correct child? |
| H9 | Article parent + contextual child | Parent BM25 → Child BM25 + RRF | Yes | Does parent-first retrieval narrow child search effectively? |

The corresponding configs are under `configs/examples/`. H1/H6 and H7/H8/H9
are lightweight BM25 comparison groups. H2-H5 download Hugging Face models and
may require substantial time, RAM, and disk space. A GPU is optional but
recommended.

## 2. Environment setup

### Local Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[all]"
python -m pip check
pytest -q
```

For BM25-only experiments:

```bash
python -m pip install -e .
```

### Docker

```bash
docker build -t legal-retrieval:locked .
```

When using `--rm`, mount `artifacts` so generated indexes and results survive
after the container exits:

```bash
docker run --rm \
  -v "$PWD/artifacts:/app/artifacts" \
  legal-retrieval:locked \
  python scripts/run_experiment.py \
  --config configs/examples/h1_sparse_hierarchical.yaml
```

## 3. Data and parent-child contract

`data/enriched/` is immutable input; the pipeline never writes back to it.

- H1-H6 read legal units or derive fixed-length chunks from them.
- H7-H9 join Articles and Children with
  `(doc_id, article, effective_from, effective_to)` rather than parsing
  `unit_id`.
- Generated Children contain `parent_id`, `article_id`, `parent_heading`, and
  `retrieval_role=child`.
- H9 Parent records contain `retrieval_role=parent`.
- Builds fail by default when a Child is orphaned or date metadata is invalid.

H7 deliberately excludes `title_chain`, because that field already contains
the Article heading and would leak parent context into the baseline. H8 adds
only a length-capped Article heading. H9 retrieves Parents first, searches only
the Children of top-ranked Parents, and combines parent/child ranks with RRF.
All H7-H9 variants return Children, so they share the same child-level relevance
labels and evaluation semantics.

## 4. Temporal semantics for H7-H9

Temporal eligibility is filtered before retrieval using `effective_from` and
`effective_to`:

| Query example | Interpretation |
|---|---|
| No time expression | Valid at the pinned `temporal.reference_date` |
| `năm 2022` | Validity interval overlaps calendar year 2022 |
| `tại thời điểm 01/06/2022` | Valid on that exact date |
| `trước năm 2020` | `effective_from < 2020-01-01` |
| `từ năm 2020 đến năm 2022` | Validity interval overlaps the requested range |

`Luật Đất đai 2024` is not automatically interpreted as a temporal query unless
it includes a temporal cue such as `năm`, `trước`, `sau`, `từ`, `đến`, or
`tại thời điểm`.

BM25 and NumPy dense retrieval apply the allow-list before ranking. The current
FAISS adapter searches all vector IDs and then filters exactly. This is correct
for the current corpus but is not optimized for very large deployments.
Temporal accuracy depends on metadata supplied by Workstream A. Partial expiry,
promulgation dates, and amendment/supersession chains require additional
upstream metadata.

## 5. Single-query smoke tests

The retrieval script automatically builds or reuses the correct
content-addressed index.

```bash
python scripts/retrieve.py \
  --config configs/examples/h1_sparse_hierarchical.yaml \
  --query "điều kiện cấp giấy chứng nhận quyền sử dụng đất" \
  --top-k 10 \
  > artifacts/reports/h1_smoke.json
```

Inspect H9 together with its parsed temporal constraint:

```bash
python scripts/retrieve.py \
  --config configs/examples/h9_parent_to_child_temporal.yaml \
  --query "năm 2022 ngư dân được hỗ trợ như thế nào" \
  --top-k 10 \
  --explain-query \
  > artifacts/reports/h9_temporal_smoke.json
```

Metadata filters can be repeated:

```bash
python scripts/retrieve.py \
  --config configs/examples/h1_sparse_hierarchical.yaml \
  --query "đăng ký đất đai" \
  --top-k 10 \
  --filter doc_type=law \
  --filter unit_type=article
```

## 6. Build indexes and run suites

Build or reuse one index without running a query:

```bash
python scripts/build_index.py \
  --config configs/examples/h8_contextual_child_temporal.yaml
```

Run H1-H3:

```bash
python scripts/run_suite.py --suite configs/suites/example_suite.yaml
```

Run the lightweight BM25 H7-H9 group:

```bash
python scripts/run_suite.py \
  --suite configs/suites/parent_child_temporal_suite.yaml
```

Run all H1-H9 experiments (H2-H5 may be resource-intensive):

```bash
python scripts/run_suite.py \
  --suite configs/suites/all_hypotheses.yaml
```

When `evaluation.benchmark_path` is `null`, a suite only builds or reuses its
indexes and records `index_ready_no_benchmark`. This is not an evaluation
result.

## 7. Run a labeled benchmark

Benchmark JSONL example:

```jsonl
{"query_id":"q001","query":"điều kiện cấp giấy chứng nhận quyền sử dụng đất","relevant_unit_ids":["legal-unit-id-1"],"filters":{}}
{"query_id":"q002","query":"năm 2022 ngư dân được hỗ trợ thế nào","relevant_unit_ids":["legal-unit-id-2"],"filters":{}}
```

Set one shared benchmark for the full campaign in `configs/base.yaml`:

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

Then run one hypothesis or the full suite:

```bash
python scripts/run_experiment.py \
  --config configs/examples/h7_child_temporal.yaml

python scripts/run_suite.py \
  --suite configs/suites/all_hypotheses.yaml
```

Do not change the benchmark between hypotheses. For H7-H9, keep
`temporal.reference_date` fixed and inspect `query_context.temporal` in the
output.

## 8. Results and comparison

Each run is stored under:

```text
artifacts/runs/<hypothesis-id>/<run-id>/
├── resolved_config.yaml
├── run_metadata.json
├── run_summary.json
├── metrics.json
└── per_query.jsonl
```

Export one comparison table:

```bash
python scripts/compare_experiments.py \
  --runs-dir artifacts/runs \
  --output artifacts/reports/experiment_comparison.csv
```

Inspect files in this order:

1. `run_summary.json`: status, duration, and index signature.
2. `metrics.json`: aggregate metrics.
3. `per_query.jsonl`: error analysis, results, and temporal interpretation.
4. `resolved_config.yaml`: exact effective experimental variables.
5. `run_metadata.json`: seed, packages, Python, platform, and Git revision.

## 9. Valid-experiment checklist

- Use the same corpus snapshot and benchmark version.
- Change only variables declared under `hypothesis.variables`.
- Pin `temporal.reference_date` for H7-H9.
- Pin full Hugging Face commit SHAs and enable
  `runtime.reproducibility_mode: final` for final dense/reranker runs.
- Do not evaluate from one query or only the first result.
- Compare aggregate metrics and inspect `per_query.jsonl`.
- Record latency, RAM/VRAM, and build time alongside retrieval metrics.
