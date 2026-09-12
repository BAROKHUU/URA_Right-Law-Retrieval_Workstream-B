from __future__ import annotations
import argparse
from legal_retrieval.config import load_config
from legal_retrieval.indexing import IndexManager
from legal_retrieval.logging_utils import configure_logging


def main():
    p = argparse.ArgumentParser(description="Build sparse/dense indexes from Workstream A enriched data.")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--force", action="store_true",
        help="Re-run the build command; an existing immutable signature is still reused, never overwritten.",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.force:
        cfg.setdefault("runtime", {})["force_rebuild"] = True
    configure_logging(cfg.get("runtime", {}).get("log_level", "INFO"))
    manager = IndexManager(cfg)
    manager.build() if args.force or manager.needs_build() else print(f"Index already up to date: {manager.index_dir}")


if __name__ == "__main__":
    main()
