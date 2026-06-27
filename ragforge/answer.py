"""Compose an extractive, citation-first answer from retrieved chunks.

There is no language model in this path. The answer is built from the best
matching sentence in each retrieved chunk, numbered, and every sentence keeps a
pointer back to the chunk it came from. That has two consequences worth stating
plainly: the wording is the corpus's wording (so it cannot hallucinate details),
and the answer only reads well when retrieval found the right passage.

When a language model is available — see ``providers.py`` — the prompt is the
same citations, and the model is instructed to answer only from them. The
citation list stays the contract between retrieval and generation.
"""

from __future__ import annotations

import re

from .text import sentence_split, tokenize

# Extractive QA has one advantage over a model: the question's shape is visible
# before any sentence is scored. "How long" and "how many" questions are
# answered by a sentence containing a number and a unit, so that sentence gets a
# bonus over one that merely repeats the topic words. Without this, "How long is
# the signature tolerance?" quotes "Every webhook delivery is signed." — true,
# cited, and useless.
_NUMBER = r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty|forty|fifty|sixty|ninety|hundred)"
_UNIT = r"(?:second|minute|hour|day|week|month|year|business\s+day)"
_DURATION_SHAPE = re.compile(rf"\b{_NUMBER}\s+(?:business\s+)?{_UNIT}s?\b", re.IGNORECASE)
_COUNT_SHAPE = re.compile(r"\b\d+\b")
_DURATION_QUESTION = re.compile(r"\bhow (?:long|soon|much time)\b|\btolerance\b|\bwindow\b|\bretention\b", re.IGNORECASE)
_COUNT_QUESTION = re.compile(r"\bhow many\b|\bmaximum number\b|\bhighest number\b", re.IGNORECASE)


def _shape_bonus(question: str, sentence: str) -> float:
    if _DURATION_QUESTION.search(question) and _DURATION_SHAPE.search(sentence):
        return 1.5
    if _COUNT_QUESTION.search(question) and _COUNT_SHAPE.search(sentence):
        return 1.5
    return 0.0


def _sentence_score(sentence: str, query_terms: set[str], question: str) -> float:
    overlap = sum(1 for term in set(tokenize(sentence)) if term in query_terms)
    return overlap + _shape_bonus(question, sentence)


def compose_answer(
    question: str,
    hits: list[dict],
    *,
    max_citations: int = 3,
    min_overlap: int = 1,
) -> dict:
    """Return ``{answer, citations, coverage, grounded, ...}``.

    ``hits`` is the chunk-level output of ``BM25Index.search``. Only the first
    ``max_citations`` chunks are considered, so the answer cannot balloon into a
    summary that hides which passage supports which claim.
    """
    query_terms = set(tokenize(question))
    citations: list[dict] = []
    quotes: list[str] = []
    # A chunk that scored a third of the best hit is usually a passing mention.
    # Quoting it produces the "here are three sources, two irrelevant" answer
    # that makes citation-first UIs untrustworthy, so weak hits are dropped
    # rather than padded in to look thorough.
    floor = (hits[0].get("score", 0.0) * 0.4) if hits else 0.0

    for hit in hits[:max_citations]:
        if hit.get("score", 0.0) < floor:
            continue
        sentences = sentence_split(hit.get("text", ""))
        best, best_score = None, 0.0
        for sentence in sentences:
            score = _sentence_score(sentence, query_terms, question)
            if score > best_score:
                best, best_score = sentence, score
        if best is None or best_score < min_overlap:
            continue
        marker = len(citations) + 1
        citations.append(
            {
                "marker": marker,
                "chunk_id": hit["chunk_id"],
                "doc_id": hit["doc_id"],
                "title": hit.get("title", ""),
                "heading": hit.get("heading", ""),
                "quote": best,
                "score": hit.get("score", 0.0),
            }
        )
        quotes.append(f"{best} [{marker}]")

    covered = set()
    for citation in citations:
        covered |= set(tokenize(citation["quote"])) & query_terms
    coverage = len(covered) / len(query_terms) if query_terms else 0.0

    if quotes:
        return {
            "question": question,
            "answer": " ".join(quotes),
            "citations": citations,
            "coverage": round(coverage, 4),
            "query_terms": sorted(query_terms),
            "grounded": True,
            "method": "extractive",
        }

    # Retrieval returned nothing relevant. Saying so is part of the contract:
    # a RAG demo that always produces an answer is hiding its failure state.
    fallback = ""
    if hits:
        sentences = sentence_split(hits[0].get("text", ""))
        fallback = sentences[0] if sentences else hits[0].get("text", "")
    return {
        "question": question,
        "answer": "No passage in the corpus matched this question. Try different words, or add the source document to the corpus.",
        "citations": [],
        "coverage": 0.0,
        "query_terms": sorted(query_terms),
        "grounded": False,
        "nearest": fallback,
        "method": "extractive",
    }


def format_answer(result: dict) -> str:
    """Plain-text rendering for the CLI."""
    lines = [result["answer"], ""]
    if result["citations"]:
        lines.append("Sources")
        for citation in result["citations"]:
            lines.append(f"  [{citation['marker']}] {citation['title']} / {citation['heading']} ({citation['chunk_id']}, score {citation['score']:.3f})")
    else:
        lines.append("Sources: none")
    lines.append(f"coverage: {result['coverage']:.0%} of question terms appear in the cited text")
    return "\n".join(lines)


def build_prompt(question: str, result: dict) -> str:
    """The prompt a provider adapter receives. Citations are explicit so a
    model that invents a source is visibly wrong against the citation list."""
    context = "\n\n".join(f"[{c['marker']}] {c['title']} / {c['heading']}\n{c['quote']}" for c in result["citations"])
    if not context:
        context = "(no passages were retrieved)"
    return (
        "Answer the question using only the numbered passages. "
        "Cite with [n]. If the passages do not contain the answer, say so.\n\n"
        f"Question: {question}\n\nPassages:\n{context}"
    )
