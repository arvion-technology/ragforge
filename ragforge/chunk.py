"""Split documents into retrieval-sized chunks.

Two strategies are supported and both are pure functions of the input, so the
browser demo, the tests and the CLI produce identical chunks for the same
corpus.

``heading`` (default)
    Pack whole Markdown sections into windows of at most ``size`` tokens. A
    section keeps its heading in the chunk metadata and its body in the chunk
    text. Sections are never split unless one section alone exceeds the window,
    in which case it is sentence-windowed. Keeping a section intact matters for
    answers: a "Verification" paragraph usually only makes sense next to the
    sentence that introduced it.

``fixed``
    Sliding word windows. Useful for text with no structure (logs, transcripts),
    where sentence detection is noise rather than signal.

Chunks are dictionaries rather than classes so they serialise straight to JSON
for the API response and the parity fixture.
"""

from __future__ import annotations

import re

from .text import sentence_split, tokenize

DEFAULT_SIZE = 120
DEFAULT_OVERLAP = 30

_HEADING = re.compile(r"(?m)^##\s+(.+?)\s*$")


def split_sections(doc: dict) -> list[tuple[str, str]]:
    """Return ``(heading, body)`` pairs. Text before the first H2 belongs to the
    document title, which is how the corpus files are written."""
    text = (doc.get("text") or "").strip()
    if not text:
        return []
    title = (doc.get("title") or doc.get("id") or "document").strip()
    pieces = _HEADING.split(text)
    sections: list[tuple[str, str]] = []

    preamble = pieces[0].strip()
    if preamble:
        sections.append((title, preamble))

    # re.split with one capture group yields [preamble, h, body, h, body, ...]
    for i in range(1, len(pieces), 2):
        heading = pieces[i].strip()
        body = pieces[i + 1].strip() if i + 1 < len(pieces) else ""
        if body:
            sections.append((heading, body))
        elif heading and not sections:
            # A heading with no body is still a legitimate (short) section.
            sections.append((heading, ""))
    return sections


def _count(text: str) -> int:
    return len(tokenize(text, keep_stopwords=True))


def _window_sections(sections: list[tuple[str, str]], size: int, overlap: int) -> list[list[tuple[str, str]]]:
    """Pack sections into windows of at most ``size`` tokens.

    Overlap is carried as whole trailing sections, so it only applies when the
    trailing sections are short enough to fit in the overlap budget. That is
    intentional: duplicating half a section would make the same sentence appear
    in two citations with two different chunk ids.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    windows: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_tokens = 0

    for heading, body in sections:
        unit_tokens = _count(heading) + _count(body)
        if current and current_tokens + unit_tokens > size:
            windows.append(current)
            tail: list[tuple[str, str]] = []
            tail_tokens = 0
            for previous in reversed(current):
                previous_tokens = _count(previous[0]) + _count(previous[1])
                if previous_tokens > overlap or tail_tokens + previous_tokens > overlap:
                    break
                tail.insert(0, previous)
                tail_tokens += previous_tokens
            current = tail
            current_tokens = tail_tokens
        current.append((heading, body))
        current_tokens += unit_tokens

    if current:
        windows.append(current)
    return windows


def _sentence_windows(body: str, size: int, overlap: int) -> list[str]:
    """Fallback for a single section longer than the window: pack sentences and
    carry trailing sentences as overlap."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    windows: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentence_split(body):
        tokens = _count(sentence)
        if current and current_tokens + tokens > size:
            windows.append(current)
            tail: list[str] = []
            tail_tokens = 0
            for previous in reversed(current):
                previous_tokens = _count(previous)
                if previous_tokens > overlap or tail_tokens + previous_tokens > overlap:
                    break
                tail.insert(0, previous)
                tail_tokens += previous_tokens
            current = tail
            current_tokens = tail_tokens
        current.append(sentence)
        current_tokens += tokens
    if current:
        windows.append(current)
    return [" ".join(window).strip() for window in windows]


def chunk_document(
    doc: dict,
    *,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = "heading",
) -> list[dict]:
    """Chunk one document. ``doc`` needs ``id`` and ``text``; ``title`` and
    ``kind`` are copied onto every chunk so a citation is self-describing."""
    doc_id = doc.get("id") or "doc"
    title = doc.get("title") or doc_id
    kind = doc.get("kind") or "document"
    pieces: list[dict] = []

    if strategy == "fixed":
        words = (doc.get("text") or "").split()
        step = max(1, size - overlap)
        for start in range(0, len(words), step):
            body = " ".join(words[start : start + size]).strip()
            if not body:
                break
            pieces.append({"headings": [title], "text": body})
            if start + size >= len(words):
                break
    elif strategy == "heading":
        for window in _window_sections(split_sections(doc), size, overlap):
            headings = [heading for heading, _ in window]
            bodies = [body for _, body in window if body]
            text = "\n\n".join(bodies).strip()
            if not text:
                continue
            if _count(text) > size:
                # A single oversized section: sentence-window it, reusing the
                # heading so every produced chunk is still labelled.
                for part in _sentence_windows("\n\n".join(bodies), size, overlap):
                    pieces.append({"headings": headings, "text": part})
            else:
                pieces.append({"headings": headings, "text": text})
    else:
        raise ValueError(f"unknown chunk strategy: {strategy}")

    out = []
    for ordinal, piece in enumerate(pieces):
        out.append(
            {
                "id": f"{doc_id}::{ordinal}",
                "doc_id": doc_id,
                "title": title,
                "kind": kind,
                "heading": piece["headings"][0],
                "headings": piece["headings"],
                "ordinal": ordinal,
                "text": piece["text"],
            }
        )
    return out


def chunk_documents(documents: list[dict], **options) -> list[dict]:
    chunks: list[dict] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, **options))
    return chunks


def chunk_stats(chunks: list[dict]) -> dict:
    """Small summary used by the CLI and the demo's corpus panel."""
    tokens = [len(tokenize(chunk["text"])) for chunk in chunks]
    docs = {chunk["doc_id"] for chunk in chunks}
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "tokens": sum(tokens),
        "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
        "max_tokens": max(tokens) if tokens else 0,
    }
