from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Collect metrics from experiment run folders into one CSV.")
    p.add_argument("--runs-dir", default="artifacts/runs")
    p.add_argument("--output", default="artifacts/reports/experiment_comparison.csv")
    args = p.parse_args()
    root = Path(args.runs_dir)
    rows = []
    for summary_path in root.glob("*/*/run_summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg_path = summary_path.parent / "resolved_config.yaml"
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        runtime = summary.get("runtime", {})
        latency = runtime.get("query_latency", {})
        resources = summary.get("resources", {})
        gpu_devices = resources.get("gpu_devices", [])
        row = {
            "hypothesis_id": cfg.get("hypothesis", {}).get("id", summary_path.parents[1].name),
            "description": cfg.get("hypothesis", {}).get("description", ""),
            "run_id": summary_path.parent.name,
            "status": summary.get("status"),
            "query_count": summary.get("query_count", 0),
            "labeled_query_count": summary.get("labeled_query_count", 0),
            "total_seconds": runtime.get("total_seconds", summary.get("duration_seconds")),
            "pipeline_setup_seconds": runtime.get("pipeline_setup_seconds"),
            "index_operation": runtime.get("index_operation"),
            "index_operation_seconds": runtime.get("index_operation_seconds"),
            "index_size_bytes": summary.get("index_size_bytes"),
            "query_mean_seconds": latency.get("mean_seconds"),
            "query_p95_seconds": latency.get("p95_seconds"),
            "first_query_seconds": latency.get("first_query_seconds"),
            "warm_query_mean_seconds": latency.get("warm_mean_seconds"),
            "throughput_queries_per_second": runtime.get("throughput_queries_per_second"),
            "peak_process_ram_bytes": resources.get("peak_process_ram_bytes"),
            "peak_gpu_allocated_bytes": max(
                (device.get("peak_allocated_bytes", 0) for device in gpu_devices),
                default=0,
            ),
            "peak_gpu_reserved_bytes": max(
                (device.get("peak_reserved_bytes", 0) for device in gpu_devices),
                default=0,
            ),
        }
        row.update(summary.get("metrics", {}))
        rows.append(row)
    if not rows:
        print("No completed experiment summaries found.")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {out}")


if __name__ == "__main__":
    main()
