from __future__ import annotations

from typing import Any

from .pipeline import RetrievalPipeline


def create_app(cfg: dict[str, Any]):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError("API dependencies missing. Run `pip install -e \".[api]\"`.") from exc

    pipeline = RetrievalPipeline(cfg)
    app = FastAPI(title="Legal Retrieval API", version="0.1.0")

    class RetrieveRequest(BaseModel):
        query: str = Field(min_length=1)
        top_k: int | None = Field(default=None, ge=1, le=1000)
        filters: dict[str, Any] = Field(default_factory=dict)

    @app.get("/health")
    def health():
        return {"status": "ok", "index_dir": str(pipeline.manager.index_dir)}

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        try:
            hits = pipeline.retrieve(req.query, top_k=req.top_k, filters=req.filters)
            return {
                "query": req.query,
                "query_context": pipeline.last_query_context,
                "count": len(hits),
                "results": [h.to_dict(include_text=True) for h in hits],
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
