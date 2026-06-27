"""The object most callers need: documents in, cited answers out."""

from __future__ import annotations

import json
from pathlib import Path

from .answer import compose_answer
from .bm25 import BM25Index
from .chunk import DEFAULT_OVERLAP, DEFAULT_SIZE, chunk_documents, chunk_stats
from .ingest import load_documents


def default_corpus_path() -> Path:
    """The bundled 20-document Mesa corpus.

    Resolved relative to the repository rather than the installed package: the
    corpus is demo data, not package data, so ``pip install .`` does not copy it.
    Set ``RAGFORGE_CORPUS`` to point the CLI somewhere else.
    """
    root = Path(__file__).resolve().parent.parent
    return root / "corpus" / "documents.json"


class Retriever:
    """Chunk, index and answer. One object so the CLI and API share a path."""

    def __init__(
        self,
        documents: list[dict],
        *,
        size: int = DEFAULT_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        strategy: str = "heading",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("Retriever needs at least one document")
        self.documents = documents
        self.options = {"size": size, "overlap": overlap, "strategy": strategy, "k1": k1, "b": b}
        self.chunks = chunk_documents(documents, size=size, overlap=overlap, strategy=strategy)
        self.index = BM25Index.from_chunks(self.chunks, k1=k1, b=b)

    @classmethod
    def from_corpus_file(cls, path: str | Path | None = None, **options) -> "Retriever":
        source = Path(path) if path else default_corpus_path()
        return cls(load_documents(source), **options)

    # -- queries --------------------------------------------------------

    def search(self, question: str, *, k: int = 4) -> list[dict]:
        return self.index.search(question, k=k)

    def search_documents(self, question: str, *, k: int = 5) -> list[dict]:
        return self.index.search_documents(question, k=k)

    def ask(self, question: str, *, k: int = 4) -> dict:
        hits = self.search(question, k=k)
        result = compose_answer(question, hits)
        result["retrieval"] = hits
        return result

    def document(self, doc_id: str) -> dict | None:
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None

    # -- introspection --------------------------------------------------

    def stats(self) -> dict:
        return {
            "index": self.index.stats(),
            "chunks": chunk_stats(self.chunks),
            "options": self.options,
        }

    def manifest(self) -> list[dict]:
        return [
            {
                "id": doc["id"],
                "title": doc["title"],
                "kind": doc.get("kind", "document"),
                "updated": doc.get("updated"),
                "chunks": sum(1 for chunk in self.chunks if chunk["doc_id"] == doc["id"]),
            }
            for doc in self.documents
        ]


def load_eval_labels(path: str | Path | None = None) -> list[dict]:
    source = Path(path) if path else default_corpus_path().parent / "eval.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{source} must contain a non-empty JSON array")
    return data
