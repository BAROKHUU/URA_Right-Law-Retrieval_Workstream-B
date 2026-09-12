from __future__ import annotations
import argparse
from legal_retrieval.api import create_app
from legal_retrieval.config import load_config
from legal_retrieval.logging_utils import configure_logging


def main():
    p = argparse.ArgumentParser(description="Serve the configured retrieval pipeline with FastAPI.")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("runtime", {}).get("log_level", "INFO"))
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Run `pip install -e \".[api]\"` first.") from exc
    app = create_app(cfg)
    uvicorn.run(app, host=cfg["api"].get("host", "127.0.0.1"), port=int(cfg["api"].get("port", 8000)))


if __name__ == "__main__":
    main()
