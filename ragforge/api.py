"""FastAPI application shape.

FastAPI is optional. Importing ``ragforge.api`` must not require it: the
package's primary use is as a library and its tests run on the standard library
alone. ``create_app`` imports FastAPI inside the function and raises a readable
error if it is not installed.

Run it with::

    pip install fastapi uvicorn
    uvicorn ragforge.api:app_factory --factory

The routes are deliberately thin. Validation, retrieval and answer composition
live in the library, so the same code paths are exercised by the tests.
"""

from __future__ import annotations

import json

from .evaluate import evaluate
from .providers import complete_with_fallback, provider_status
from .retriever import Retriever, load_eval_labels


def create_app(retriever: Retriever | None = None, *, labels: list[dict] | None = None):
    """Build the FastAPI app. Raises RuntimeError when FastAPI is missing."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as exc:  # pragma: no cover - exercised only with fastapi installed
        raise RuntimeError(
            "The HTTP API needs FastAPI. Install it with `pip install fastapi uvicorn`, "
            "or use the retriever directly."
        ) from exc

    live = {"retriever": retriever or Retriever.from_corpus_file()}
    labels = labels if labels is not None else load_eval_labels()

    app = FastAPI(
        title="RAGForge",
        version="1.0.0",
        description="Citation-first retrieval over a local corpus. No API key required.",
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "retrieval": "bm25",
            "corpus": live["retriever"].stats()["index"],
            "providers": provider_status(),
        }

    @app.get("/documents")
    def documents() -> dict:
        return {"documents": live["retriever"].manifest()}

    @app.post("/query")
    def query(body: dict) -> dict:
        question = str(body.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=422, detail="question is required")
        k = int(body.get("k") or 4)
        if not 1 <= k <= 20:
            raise HTTPException(status_code=422, detail="k must be between 1 and 20")
        return live["retriever"].ask(question, k=k)

    @app.post("/query/stream")
    def query_stream(body: dict):
        """Server-sent events, one passage per event.

        The extractive composer already has the answer before the first byte is
        written; streaming exists so a UI can render citations progressively
        instead of waiting for the whole JSON body.
        """
        answer = live["retriever"].ask(str(body.get("question") or ""), k=int(body.get("k") or 4))

        def events():
            yield f"event: answer\ndata: {json.dumps({'answer': answer['answer']})}\n\n"
            for citation in answer["citations"]:
                yield f"event: citation\ndata: {json.dumps(citation)}\n\n"
            yield f"event: done\ndata: {json.dumps({'coverage': answer['coverage']})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/ingest")
    def ingest(body: dict) -> dict:
        documents = body.get("documents")
        if not isinstance(documents, list) or not documents:
            raise HTTPException(status_code=422, detail="documents must be a non-empty array")
        try:
            rebuilt = Retriever(documents, **live["retriever"].options)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        live["retriever"] = rebuilt
        app.state.retriever = rebuilt
        return {"accepted": len(documents), "index": rebuilt.stats()["index"]}

    @app.get("/eval")
    def eval_route(k: int = 5) -> dict:
        report = evaluate(live["retriever"], labels, k=max(1, min(k, 20)))
        return {key: report[key] for key in ("k", "queries", "hit_rate", "mrr", "recall")} | {
            "misses": [row["question"] for row in report["misses"]]
        }

    @app.post("/answer/prose")
    def prose(body: dict) -> dict:
        """Optional: rewrite the extractive answer with a configured provider.

        Without a configured key this returns the deterministic mock text and
        ``degraded: true``. It never fabricates an API response.
        """
        from .answer import build_prompt

        question = str(body.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=422, detail="question is required")
        answer = live["retriever"].ask(question, k=int(body.get("k") or 4))
        prompt = build_prompt(question, answer)
        return {"answer": answer, "generation": complete_with_fallback(prompt, provider=str(body.get("provider") or "mock"))}

    return app


def app_factory():
    """Entry point for ``uvicorn ragforge.api:app_factory --factory``."""
    return create_app()
