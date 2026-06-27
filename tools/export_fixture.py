#!/usr/bin/env python3
"""Regenerate the cross-language BM25 parity fixture.

The browser demo runs a JavaScript port of the retriever. That port is only
useful if it produces the same numbers as the Python package, so both sides are
tested against one committed fixture:

    python3 tools/export_fixture.py        # after changing text.py/chunk.py/bm25.py

The fixture pins chunk ids, document order and scores rounded to six decimals.
Rounding is the point: the two languages share IEEE-754 doubles but not
necessarily the same libm, and a difference in the last bit is not a bug.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragforge.retriever import Retriever  # noqa: E402

QUERIES = [
    "webhook signature timestamp tolerance",
    "cursor pagination limit",
    "refund window payment method",
    "SAML signed assertions",
    "recovery time objective restore",
    "rate limit headers burst",
    "SCIM deprovision revoke sessions",
    "export signed URL expiry",
]


def main() -> int:
    retriever = Retriever.from_corpus_file(ROOT / "corpus" / "documents.json")
    payload = {
        "corpus": "corpus/documents.json",
        "options": retriever.options,
        "index": {
            "chunks": retriever.index.stats()["chunks"],
            "documents": retriever.index.stats()["documents"],
            "terms": retriever.index.stats()["terms"],
            "avgdl": retriever.index.stats()["avgdl"],
        },
        "queries": [
            {
                "query": query,
                "results": [
                    {"chunk_id": hit["chunk_id"], "score": round(hit["score"], 6)}
                    for hit in retriever.search(query, k=5)
                ],
            }
            for query in QUERIES
        ],
    }
    target = ROOT / "test" / "fixtures" / "bm25-parity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)} ({len(payload['queries'])} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
