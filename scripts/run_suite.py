from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import traceback
from legal_retrieval.config import load_config
from legal_retrieval.experiment import run_experiment
from legal_retrieval.logging_utils import configure_logging


def main():
    p = argparse.ArgumentParser(description="Run a suite of hypothesis config files sequentially.")
    p.add_argument("--suite", required=True)
    args = p.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    configs = suite.get("configs", [])
    continue_on_error = bool(suite.get("continue_on_error", False))
    if not configs:
        raise ValueError("Suite has no configs")
    for entry in configs:
        cfg_path = (suite_path.parent / entry).resolve()
        try:
            cfg = load_config(cfg_path)
            configure_logging(cfg.get("runtime", {}).get("log_level", "INFO"))
            run_dir = run_experiment(cfg)
            print(f"{cfg.get('hypothesis', {}).get('id')}: {run_dir}")
        except Exception:
            print(f"FAILED: {cfg_path}")
            traceback.print_exc()
            if not continue_on_error:
                raise


if __name__ == "__main__":
    main()
