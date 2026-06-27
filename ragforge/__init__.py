"""RAGForge — a RAG starter kit that runs without an API key.

The pieces are importable independently::

    from ragforge import Retriever, load_documents, evaluate

    retriever = Retriever.from_corpus_file("corpus/documents.json")
    answer = retriever.ask("How are webhook signatures verified?")
    print(answer["answer"])

Retrieval is Okapi BM25 over heading-aware chunks. Answers are extractive and
cited. Evaluation is hit rate, MRR and recall over hand-labelled questions.
Nothing in that path calls the network.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .answer import build_prompt, compose_answer, format_answer
from .bm25 import BM25Index
from .chunk import chunk_document, chunk_documents, chunk_stats
from .evaluate import answer_quality, evaluate, evaluate_at, hit_at_k, reciprocal_rank
from .ingest import IngestError, load_documents, load_file, parse_html, parse_markdown
from .providers import (
    AnthropicProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    ProviderUnavailable,
    get_provider,
    provider_status,
)
from .retriever import Retriever, load_eval_labels
from .text import STOPWORDS, sentence_split, stem, tokenize

__all__ = [
    "AnthropicProvider",
    "BM25Index",
    "IngestError",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "ProviderUnavailable",
    "Retriever",
    "STOPWORDS",
    "answer_quality",
    "build_prompt",
    "chunk_document",
    "chunk_documents",
    "chunk_stats",
    "compose_answer",
    "evaluate",
    "evaluate_at",
    "format_answer",
    "get_provider",
    "hit_at_k",
    "load_documents",
    "load_eval_labels",
    "load_file",
    "parse_html",
    "parse_markdown",
    "provider_status",
    "reciprocal_rank",
    "sentence_split",
    "stem",
    "tokenize",
]
