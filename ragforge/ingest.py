"""Loaders that turn files on disk into the document dicts the retriever expects.

The pipeline boundary is intentionally narrow: every loader produces
``{"id", "title", "kind", "text"}`` and nothing downstream knows what file
format it came from. That keeps chunking, retrieval and evaluation independent
of ingestion, and makes the JSON corpus used by the demo just another loader.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

from .text import sentence_split

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".json"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article", "header", "footer"}


class IngestError(ValueError):
    """Raised when a file cannot be turned into a document."""


class _HtmlText(HTMLParser):
    """Extract readable text from HTML without a parser dependency.

    Script and style contents are dropped rather than escaped: they are the
    most common source of junk terms in a crawled corpus, and a retrieval
    system that indexes JavaScript cannot be trusted to answer a question
    about the page.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ARG002 - parser API
        if tag in ("script", "style"):
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
        return "\n\n".join(line for line in lines if line)


def parse_html(html: str) -> tuple[str, str]:
    """Return ``(title, text)`` for an HTML fragment or page."""
    parser = _HtmlText()
    parser.feed(html)
    parser.close()
    return parser.title.strip(), parser.text()


def parse_markdown(text: str, *, fallback_title: str) -> tuple[str, str]:
    """Pull the first H1 out as the title and keep the rest as Markdown.

    Keeping the H2 headings is what makes heading-aware chunking work, so the
    H1 is removed but nothing else is rewritten.
    """
    title = fallback_title
    body_lines: list[str] = []
    found_title = False
    for line in text.splitlines():
        if not found_title and re.match(r"^#\s+\S", line):
            title = line[2:].strip()
            found_title = True
            continue
        body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return title, body


def _title_from_name(path: Path) -> str:
    return re.sub(r"[-_]+", " ", path.stem).strip().title() or path.stem


def document_from_text(text: str, *, doc_id: str, title: str, kind: str = "document") -> dict:
    body = (text or "").strip()
    if not body:
        raise IngestError(f"document {doc_id!r} has no text")
    return {"id": doc_id, "title": title.strip() or doc_id, "kind": kind, "text": body}


def load_file(path: str | Path) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(f"unsupported file type: {path.suffix or path.name}")
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".html", ".htm"):
        title, text = parse_html(raw)
        return document_from_text(text, doc_id=path.stem, title=title or _title_from_name(path), kind="html")
    if suffix in (".md", ".markdown"):
        title, text = parse_markdown(raw, fallback_title=_title_from_name(path))
        return document_from_text(text, doc_id=path.stem, title=title, kind="markdown")
    return document_from_text(raw, doc_id=path.stem, title=_title_from_name(path), kind="text")


def normalize_document(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise IngestError("a document must be a JSON object")
    doc_id = str(raw.get("id") or "").strip()
    text = str(raw.get("text") or "").strip()
    if not doc_id:
        raise IngestError("document is missing an id")
    if not text:
        raise IngestError(f"document {doc_id!r} has no text")
    return {
        "id": doc_id,
        "title": str(raw.get("title") or doc_id).strip(),
        "kind": str(raw.get("kind") or "document").strip(),
        "updated": raw.get("updated"),
        "text": text,
    }


def load_json_corpus(raw: str) -> list[dict]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IngestError(f"invalid JSON corpus: {exc.msg} at line {exc.lineno}") from exc
    if isinstance(parsed, dict) and "documents" in parsed:
        parsed = parsed["documents"]
    if not isinstance(parsed, list):
        raise IngestError("JSON corpus must be a list of documents or {\"documents\": [...]}")
    documents = [normalize_document(item) for item in parsed]
    if not documents:
        raise IngestError("JSON corpus is empty")
    return documents


def load_documents(source: str | Path) -> list[dict]:
    """Load a JSON corpus, a single text file, or every supported file in a
    directory (recursively, in sorted order so ids and chunk ordinals are
    stable across runs)."""
    path = Path(source)
    if not path.exists():
        raise IngestError(f"no such file or directory: {path}")
    if path.is_file():
        if path.suffix.lower() == ".json":
            return load_json_corpus(path.read_text(encoding="utf-8"))
        return [load_file(path)]

    documents = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        if file.name.startswith((".", "_")):
            continue
        if file.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if file.suffix.lower() == ".json":
            documents.extend(load_json_corpus(file.read_text(encoding="utf-8")))
        else:
            documents.append(load_file(file))
    if not documents:
        raise IngestError(f"no supported documents found under {path}")
    return documents


def preview(document: dict, *, sentences: int = 2) -> str:
    """First few sentences, used by the review UI and the CLI listing."""
    return " ".join(sentence_split(document["text"])[:sentences])
