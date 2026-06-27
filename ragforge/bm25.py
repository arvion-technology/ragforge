"""Okapi BM25 over the chunk index.

Why BM25 and not embeddings: the kit has to run with no API key, no model
download and no vector database, and BM25 is strong on the kind of named,
technical text a product documentation corpus contains. The class is small
enough to audit, and the embedding adapters in ``providers.py`` can be layered
on later without changing the answer or evaluation interfaces.

The parameters are the usual ``k1=1.5`` and ``b=0.75``. ``b`` matters here: a
long reference page should not outrank a short paragraph that answers the
question, so document length is normalised rather than purely rewarded.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .text import tokenize

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Index:
    """An in-memory inverted index with per-term score attribution."""

    def __init__(self, *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.chunks: list[dict] = []
        self._tokens: list[list[str]] = []
        self._freqs: list[Counter] = []
        self._df: Counter = Counter()
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        self._built = False

    # -- construction ---------------------------------------------------

    def add(self, chunks: list[dict]) -> "BM25Index":
        if self._built:
            raise RuntimeError("cannot add chunks after build(); create a new index")
        for chunk in chunks:
            # The heading is part of the searchable text, counted twice: it is
            # a human-written topic label, so a section headed "Rate limits"
            # should outrank a passing mention of the same words in another
            # document. (Document titles reach the index through the preamble
            # heading, so they are not counted separately.) Counting headings
            # in the term frequency rather than adding a second scoring field
            # keeps the formula the one everyone knows.
            heading = tokenize(chunk.get("heading", ""))
            tokens = heading + heading + tokenize(chunk.get("text", ""))
            self.chunks.append(chunk)
            self._tokens.append(tokens)
            self._freqs.append(Counter(tokens))
            for term in set(tokens):
                self._df[term] += 1
        return self

    def build(self) -> "BM25Index":
        total = sum(len(tokens) for tokens in self._tokens)
        self._avgdl = total / len(self._tokens) if self._tokens else 0.0
        n = len(self._tokens)
        for term, df in self._df.items():
            # IDF floor is 0 rather than a negative value: a term in every
            # chunk is uninformative, not evidence against the chunk.
            self._idf[term] = max(0.0, math.log(1 + (n - df + 0.5) / (df + 0.5)))
        self._built = True
        return self

    @classmethod
    def from_chunks(cls, chunks: list[dict], **options) -> "BM25Index":
        return cls(**options).add(chunks).build()

    # -- scoring --------------------------------------------------------

    def _term_score(self, term: str, term_freq: int, length: int) -> float:
        if not term_freq:
            return 0.0
        idf = self._idf.get(term, 0.0)
        norm = self.k1 * (1 - self.b + self.b * (length / self._avgdl if self._avgdl else 0.0))
        return idf * (term_freq * (self.k1 + 1)) / (term_freq + norm)

    def score(self, query: str) -> list[dict]:
        """Score every chunk and return results ordered by score, then chunk id.

        The deterministic tie-break matters for evaluation: without it, two
        chunks with identical scores could swap between runs and change MRR.
        """
        if not self._built:
            raise RuntimeError("call build() before search()")
        terms = list(dict.fromkeys(tokenize(query)))
        results = []
        for i, chunk in enumerate(self.chunks):
            freqs = self._freqs[i]
            length = len(self._tokens[i])
            contributions = {}
            total = 0.0
            for term in terms:
                value = self._term_score(term, freqs.get(term, 0), length)
                if value:
                    contributions[term] = round(value, 6)
                    total += value
            results.append(
                {
                    "chunk_id": chunk["id"],
                    "doc_id": chunk["doc_id"],
                    "title": chunk.get("title", ""),
                    "heading": chunk.get("heading", ""),
                    "kind": chunk.get("kind", ""),
                    "text": chunk.get("text", ""),
                    "score": round(total, 6),
                    "terms": contributions,
                }
            )
        results.sort(key=lambda row: (-row["score"], row["chunk_id"]))
        return results

    def search(self, query: str, *, k: int = 5, min_score: float = 0.0) -> list[dict]:
        if k <= 0:
            raise ValueError("k must be positive")
        hits = [row for row in self.score(query) if row["score"] > min_score]
        return hits[:k]

    def search_documents(self, query: str, *, k: int = 5) -> list[dict]:
        """Collapse chunk hits to documents, keeping the best chunk per document.

        Retrieval is chunk-level but evaluation is document-level: a citation
        points at a document, and counting three chunks of the same document as
        three hits would flatter the numbers.
        """
        best: dict[str, dict] = {}
        for hit in self.score(query):
            if hit["score"] <= 0:
                continue
            current = best.get(hit["doc_id"])
            if current is None or hit["score"] > current["score"]:
                best[hit["doc_id"]] = {**hit, "chunks_hit": current["chunks_hit"] + 1 if current else 1}
            else:
                current["chunks_hit"] += 1
        ranked = sorted(best.values(), key=lambda row: (-row["score"], row["doc_id"]))
        return ranked[:k]

    # -- introspection --------------------------------------------------

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "documents": len({chunk["doc_id"] for chunk in self.chunks}),
            "terms": len(self._df),
            "avgdl": round(self._avgdl, 2),
            "k1": self.k1,
            "b": self.b,
        }

    def term_document_frequency(self) -> dict[str, int]:
        return dict(self._df)
