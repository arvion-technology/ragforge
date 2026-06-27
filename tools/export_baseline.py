#!/usr/bin/env python3
"""Regenerate the evaluation baseline shown next to the live numbers in the demo.

    python3 tools/export_baseline.py

The demo computes hit rate, MRR and recall in the browser with the JavaScript
port. The Python numbers are committed here so the UI can show both and flag a
drift, which is a more honest comparison than a hard-coded "expected" value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragforge.evaluate import evaluate, evaluate_at  # noqa: E402
from ragforge.retriever import Retriever, load_eval_labels  # noqa: E402


def main() -> int:
    retriever = Retriever.from_corpus_file(ROOT / "corpus" / "documents.json")
    labels = load_eval_labels(ROOT / "corpus" / "eval.json")
    report = evaluate_at(retriever, labels, cutoffs=(1, 3, 5, 10))
    misses = evaluate(retriever, labels, k=1)["misses"]
    payload = {
        "index": retriever.index.stats(),
        "labels": len(labels),
        "rows": report["rows"],
        "misses_at_1": [
            {
                "id": row["id"],
                "question": row["question"],
                "relevant": row["relevant"],
                "retrieved": row["retrieved"],
            }
            for row in misses
        ],
    }
    target = ROOT / "corpus" / "baseline.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}: " + ", ".join(f"hit@{r['k']}={r['hit_rate']}" for r in payload["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
