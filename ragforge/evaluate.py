"""Retrieval evaluation: hit rate, MRR, recall and citation precision.

The labels in ``corpus/eval.json`` were written by reading each document and
writing the question a support engineer would actually ask. They are a small,
hand-checked set, 30 questions over 20 documents, so the numbers are a smoke
test for retrieval changes, not a benchmark. The demo computes the same metrics
in the browser from the same labels; ``tests/test_parity.py`` checks that the
two implementations agree.
"""

from __future__ import annotations

from .text import tokenize


def hit_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> int:
    """1 if any relevant document is in the top k, else 0."""
    return int(bool(set(ranked_doc_ids[:k]) & relevant))


def reciprocal_rank(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first relevant document, 0 when none is in the top k."""
    for position, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def recall_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_doc_ids[:k]) & relevant) / len(relevant)


def citation_precision(citations: list[dict], relevant: set[str]) -> float:
    """Share of cited chunks that point at a labelled-relevant document.

    On an extractive answer this catches the failure mode where the answer
    sentence is right but the citation attached to it is from an unrelated
    chunk.
    """
    if not citations:
        return 0.0
    return sum(1 for citation in citations if citation["doc_id"] in relevant) / len(citations)


def evaluate(index, labeled: list[dict], *, k: int = 5) -> dict:
    """Run every labelled question through the index at one cutoff."""
    per_query = []
    for item in labeled:
        question = item["question"]
        relevant = set(item.get("relevant") or [])
        hits = index.search_documents(question, k=k)
        ranked = [hit["doc_id"] for hit in hits]
        per_query.append(
            {
                "id": item.get("id"),
                "question": question,
                "relevant": sorted(relevant),
                "retrieved": ranked,
                "hit": hit_at_k(ranked, relevant, k),
                "rr": round(reciprocal_rank(ranked, relevant, k), 4),
                "recall": round(recall_at_k(ranked, relevant, k), 4),
                "top_score": round(hits[0].get("score", 0.0), 4) if hits else 0.0,
            }
        )

    count = len(per_query) or 1
    return {
        "k": k,
        "queries": len(per_query),
        "hit_rate": round(sum(row["hit"] for row in per_query) / count, 4),
        "mrr": round(sum(row["rr"] for row in per_query) / count, 4),
        "recall": round(sum(row["recall"] for row in per_query) / count, 4),
        "misses": [row for row in per_query if not row["hit"]],
        "results": per_query,
    }


def evaluate_at(index, labeled: list[dict], *, cutoffs: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Metrics at several cutoffs, the shape the demo's evaluation tab plots.

    ``hit_rate`` rises with k and is therefore useless on its own; showing the
    curve makes it obvious that k=1 is the honest number for a chat answer with
    one citation.
    """
    table = [evaluate(index, labeled, k=k) for k in cutoffs]
    return {
        "cutoffs": list(cutoffs),
        "rows": [
            {
                "k": row["k"],
                "hit_rate": row["hit_rate"],
                "mrr": row["mrr"],
                "recall": row["recall"],
                "queries": row["queries"],
            }
            for row in table
        ],
        "detail": table[-1],
    }


def answer_quality(answer: dict, relevant: set[str]) -> dict:
    """Score one composed answer: did it retrieve, cite and cover the question?"""
    query_terms = set(tokenize(answer.get("question", "")))
    cited_terms: set[str] = set()
    for citation in answer.get("citations", []):
        cited_terms |= set(tokenize(citation.get("quote", "")))
    return {
        "grounded": bool(answer.get("grounded")),
        "citations": len(answer.get("citations", [])),
        "citation_precision": round(citation_precision(answer.get("citations", []), relevant), 4),
        "coverage": answer.get("coverage", 0.0),
        "question_terms_covered": len(query_terms & cited_terms),
        "question_terms": len(query_terms),
    }
