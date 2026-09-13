from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import json
import logging
import platform
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
import yaml

from .evaluation import load_benchmark, evaluate_query, aggregate_metrics
from .io_utils import write_jsonl
from .paths import project_root
from .pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _package_versions() -> dict[str, str]:
    names = ("legal-retrieval-experiments", "numpy", "PyYAML", "sentence-transformers", "torch", "faiss-cpu")
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_experiment(cfg: dict[str, Any]) -> Path:
    started = time.perf_counter()
    seed = int(cfg.get("project", {}).get("seed", 42))
    _seed_everything(seed)
    hypothesis = cfg.get("hypothesis", {})
    name = str(hypothesis.get("id") or "experiment")
    run_dir = project_root() / "artifacts" / "runs" / name / _run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(clean_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    hypothesis_diff = cfg.get("_hypothesis_diff", {})
    logger.info("Hypothesis baseline: %s", cfg.get("_baseline_config_path"))
    logger.info("Declared changes = %s", hypothesis_diff.get("declared_changes", []))
    logger.info("Actual changes = %s", hypothesis_diff.get("actual_changes", []))

    run_metadata = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "package_versions": _package_versions(),
        "reproducibility_mode": cfg.get("runtime", {}).get("reproducibility_mode", "development"),
        "config_path": cfg.get("_config_path"),
        "baseline_config_path": cfg.get("_baseline_config_path"),
        "hypothesis_diff": hypothesis_diff,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pipeline = RetrievalPipeline(cfg)
    benchmark = load_benchmark(cfg)
    if not benchmark:
        status = {
            "status": "index_ready_no_benchmark",
            "message": "No evaluation.benchmark_path configured; index was built/loaded successfully.",
            "index_dir": str(pipeline.manager.index_dir),
            "index_signature": pipeline.manager.index_signature,
            "duration_seconds": time.perf_counter() - started,
        }
        (run_dir / "run_summary.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(status["message"])
        return run_dir

    max_k = max([int(x) for x in cfg["evaluation"].get("k_values", [10])] + [int(cfg["ranking"].get("final_top_k", 10))])
    rows = []
    metric_rows = []
    labeled_count = 0
    for item in benchmark:
        hits = pipeline.retrieve(item.query, top_k=max_k, filters=item.filters)
        row = {
            "query_id": item.query_id,
            "query": item.query,
            "filters": item.filters,
            "relevant_unit_ids": item.relevant_ids,
            "results": [h.to_dict(include_text=True) for h in hits],
        }
        if item.relevant_ids:
            metrics = evaluate_query(hits, item.relevant_ids, cfg)
            row["metrics"] = metrics
            metric_rows.append(metrics)
            labeled_count += 1
        rows.append(row)

    if cfg["evaluation"].get("save_per_query", True):
        write_jsonl(run_dir / "per_query.jsonl", rows)
    metrics = aggregate_metrics(metric_rows)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": "completed",
        "query_count": len(benchmark),
        "labeled_query_count": labeled_count,
        "metrics": metrics,
        "index_dir": str(pipeline.manager.index_dir),
        "index_signature": pipeline.manager.index_signature,
        "duration_seconds": time.perf_counter() - started,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir
