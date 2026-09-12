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
        row = {
            "hypothesis_id": cfg.get("hypothesis", {}).get("id", summary_path.parents[1].name),
            "description": cfg.get("hypothesis", {}).get("description", ""),
            "run_id": summary_path.parent.name,
            "status": summary.get("status"),
            "query_count": summary.get("query_count", 0),
            "labeled_query_count": summary.get("labeled_query_count", 0),
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
