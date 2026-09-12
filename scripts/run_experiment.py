from __future__ import annotations
import argparse
from legal_retrieval.config import load_config
from legal_retrieval.experiment import run_experiment
from legal_retrieval.logging_utils import configure_logging


def main():
    p = argparse.ArgumentParser(description="Run one config-driven retrieval hypothesis.")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("runtime", {}).get("log_level", "INFO"))
    run_dir = run_experiment(cfg)
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
