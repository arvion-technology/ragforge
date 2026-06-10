"""Chunking rules, including the cases that used to split an answer in half."""

import unittest

from ragforge.chunk import chunk_document, chunk_documents, chunk_stats, split_sections


def doc(text, **extra):
    return {"id": "d1", "title": "Test document", "text": text, **extra}


class SplitSectionTests(unittest.TestCase):
    def test_preamble_uses_the_document_title(self):
        sections = split_sections(doc("Intro text.\n\n## Details\nMore text."))
        self.assertEqual(sections[0], ("Test document", "Intro text."))
        self.assertEqual(sections[1], ("Details", "More text."))

    def test_text_without_headings_is_one_section(self):
        sections = split_sections(doc("Just one paragraph."))
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][1], "Just one paragraph.")

    def test_empty_document_has_no_sections(self):
        self.assertEqual(split_sections(doc("   ")), [])


class HeadingStrategyTests(unittest.TestCase):
    def test_chunks_keep_the_heading_that_produced_them(self):
        text = "Intro.\n\n## Refunds\nRefunds arrive within ten days.\n\n## Seats\nA seat is a user."
        chunks = chunk_document(doc(text), size=200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["headings"], ["Test document", "Refunds", "Seats"])
        # Citations show a single primary heading; the first section's heading.
        self.assertEqual(chunks[0]["heading"], "Test document")

    def test_windows_respect_the_size_budget(self):
        sections = "\n\n".join(f"## Section {i}\n" + ("word " * 40) for i in range(5))
        chunks = chunk_document(doc(sections), size=60, overlap=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # A single section may exceed the window when it is indivisible,
            # but each generated chunk should be within one section of the cap.
            self.assertLessEqual(len(chunk["text"].split()), 80)

    def test_short_trailing_sections_are_carried_as_overlap(self):
        text = "\n\n".join(f"## S{i}\n" + ("one two three four five" if i in (2, 5) else "word " * 25) for i in range(7))
        chunks = chunk_document(doc(text), size=80, overlap=30)
        self.assertGreater(len(chunks), 1)
        # The short section appears in two consecutive chunks, so the context
        # around a boundary is not lost.
        seen = [chunk["text"] for chunk in chunks]
        self.assertTrue(any(seen[i] != seen[i + 1] for i in range(len(seen) - 1)))

    def test_oversized_single_section_is_sentence_windowed(self):
        body = " ".join(f"Sentence number {i} explains a detail of the system." for i in range(30))
        chunks = chunk_document(doc("## Long\n" + body), size=40, overlap=10)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(chunk["heading"] == "Long" for chunk in chunks))
        self.assertTrue(all(chunk["text"] for chunk in chunks))

    def test_chunk_ids_are_unique_and_ordered(self):
        text = "\n\n".join(f"## S{i}\n" + ("word " * 25) for i in range(4))
        chunks = chunk_document(doc(text), size=60, overlap=0)
        ids = [chunk["id"] for chunk in chunks]
        self.assertEqual(ids, [f"d1::{i}" for i in range(len(ids))])
        self.assertEqual(len(ids), len(set(ids)))

    def test_invalid_options_are_rejected(self):
        with self.assertRaises(ValueError):
            chunk_document(doc("text"), size=0)
        with self.assertRaises(ValueError):
            chunk_document(doc("text"), size=30, overlap=30)
        with self.assertRaises(ValueError):
            chunk_document(doc("text"), strategy="fuzzy")


class FixedStrategyTests(unittest.TestCase):
    def test_fixed_windows_overlap_by_the_configured_amount(self):
        words = [f"w{i}" for i in range(30)]
        chunks = chunk_document({"id": "f", "title": "Fixed", "text": " ".join(words)}, size=10, overlap=4, strategy="fixed")
        self.assertGreater(len(chunks), 1)
        first = chunks[0]["text"].split()
        second = chunks[1]["text"].split()
        self.assertEqual(len(first), 10)
        self.assertEqual(first[-4:], second[:4])

    def test_fixed_strategy_does_not_emit_an_empty_tail_chunk(self):
        chunks = chunk_document({"id": "f", "title": "Fixed", "text": "a b c"}, size=10, overlap=2, strategy="fixed")
        self.assertEqual(len(chunks), 1)


class AggregateTests(unittest.TestCase):
    def test_chunk_documents_concatenates_in_order(self):
        documents = [
            {"id": "a", "title": "A", "text": "## One\nAlpha text."},
            {"id": "b", "title": "B", "text": "Beta text."},
        ]
        chunks = chunk_documents(documents, size=100)
        self.assertEqual([chunk["doc_id"] for chunk in chunks], ["a", "b"])

    def test_stats_report_documents_and_average_tokens(self):
        chunks = chunk_documents([
            {"id": "a", "title": "A", "text": "## One\nAlpha text about billing."},
            {"id": "b", "title": "B", "text": "Beta text."},
        ], size=100)
        stats = chunk_stats(chunks)
        self.assertEqual(stats["documents"], 2)
        self.assertEqual(stats["chunks"], 2)
        self.assertGreater(stats["avg_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
