from __future__ import annotations
import argparse
import json
from legal_retrieval.config import load_config
from legal_retrieval.logging_utils import configure_logging
from legal_retrieval.pipeline import RetrievalPipeline


def parse_filter(values: list[str]) -> dict:
    out = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Filter must be key=value: {item}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def main():
    p = argparse.ArgumentParser(description="Retrieve legal units for one query.")
    p.add_argument("--config", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--filter", action="append", default=[], help="Metadata filter key=value; repeatable.")
    p.add_argument(
        "--explain-query",
        action="store_true",
        help="Include parsed temporal constraints and eligible-record count in the JSON output.",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.get("runtime", {}).get("log_level", "INFO"))
    pipeline = RetrievalPipeline(cfg)
    hits = pipeline.retrieve(args.query, top_k=args.top_k, filters=parse_filter(args.filter))
    results = [h.to_dict(include_text=True) for h in hits]
    payload = {
        "query": args.query,
        "query_context": pipeline.last_query_context,
        "results": results,
    } if args.explain_query else results
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
