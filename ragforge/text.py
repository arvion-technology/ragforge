"""Tokenisation shared by the retriever, the answer composer and the evaluator.

Retrieval quality depends more on the tokeniser than on the ranking function:
if ``policies`` and ``policy`` do not collapse to the same term, a question
about one will miss a passage about the other. The rules below are deliberately
small and readable instead of a full Porter stemmer, every rule is one line,
and a reader can predict its output. The JavaScript port in ``src/retrieval.js``
implements the same rules and is checked against this module by
``tests/test_parity.py``.
"""

from __future__ import annotations

import re

# Function words carry no retrieval signal on their own and inflate document
# length, which in turn distorts the BM25 length normalisation. Kept as a set
# rather than a list because membership is the only operation performed.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have how i if in into is it its may might more most no not of on or our
    should so than that the their them then there these they this to was we
    were what when where which who why will with would you your
    """.split()
)

# Hyphenated and dotted identifiers such as ``x-ratelimit-limit`` or
# ``mesa-signature`` are one term, not three: they are names of things in the
# corpus, and splitting them makes the parts match everything.
_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def stem(word: str) -> str:
    """Fold simple plurals onto their singular form.

    Only plural endings are handled. Verb endings (``-ing``, ``-ed``) are not,
    because stripping them on a word such as ``during`` produces a term that
    exists in no document, and the corpus already uses a consistent form.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase, split, stem, drop stopwords. Order is preserved."""
    if not text:
        return []
    raw = _TOKEN.findall(text.lower())
    if keep_stopwords:
        return [stem(token) for token in raw]
    return [stem(token) for token in raw if token not in STOPWORDS]


def unique_terms(text: str) -> list[str]:
    """Sorted unique terms, used for coverage reporting and eval diagnostics."""
    return sorted(set(tokenize(text)))


def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    """Adjacent pairs, used by the relevance explanation panel."""
    return list(zip(tokens, tokens[1:]))


def sentence_split(text: str) -> list[str]:
    """Split prose into sentences without a NLP dependency.

    Paragraph breaks and ``##`` headings are hard boundaries. A full stop
    followed by whitespace and a capital letter is a boundary, except for the
    common abbreviations this documentation set actually uses. Getting this
    wrong only affects quote selection, never the BM25 ranking.
    """
    if not text:
        return []
    protected = text
    for abbreviation in ("approx.", "e.g.", "i.e.", "vs.", "etc.", "No."):
        protected = protected.replace(abbreviation, abbreviation.replace(".", "\u2024"))
    parts: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n(?=#{1,3}\s)", protected):
        paragraph = paragraph.strip().lstrip("#").strip()
        if not paragraph:
            continue
        # A lookbehind keeps the terminator with the sentence.
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", paragraph):
            sentence = sentence.replace("\u2024", ".").strip()
            if sentence:
                parts.append(sentence)
    return parts
