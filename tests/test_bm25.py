"""BM25 ranking behaviour: ordering, attribution and determinism."""

import unittest

from ragforge.bm25 import BM25Index
from ragforge.chunk import chunk_document


def chunk(chunk_id, text, heading="Notes", doc_id=None):
    return {
        "id": chunk_id,
        "doc_id": doc_id or chunk_id.split("::")[0],
        "title": "Doc",
        "heading": heading,
        "text": text,
    }


def index(chunks, **options):
    return BM25Index.from_chunks(chunks, **options)


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.idx = index([
            chunk("refunds::0", "Refunds are returned to the original payment method within ten business days.", "Refunds and cancellation"),
            chunk("webhooks::0", "Every webhook delivery is signed with an HMAC over the raw body.", "Webhook signatures"),
            chunk("billing::0", "A refund window of 30 days applies to the first purchase.", "Refund window"),
        ])

    def test_rare_terms_outrank_common_ones(self):
        hits = self.idx.search("refund window", k=3)
        self.assertEqual(hits[0]["chunk_id"], "billing::0")

    def test_scores_are_positive_and_ordered(self):
        hits = self.idx.search("webhook signature", k=3)
        self.assertGreater(hits[0]["score"], 0)
        scores = [hit["score"] for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_explanation_terms_sum_to_the_score(self):
        hit = self.idx.search("refund window", k=1)[0]
        self.assertAlmostEqual(sum(hit["terms"].values()), hit["score"], places=5)

    def test_search_respects_k(self):
        self.assertEqual(len(self.idx.search("refund", k=1)), 1)
        self.assertEqual(len(self.idx.search("refund", k=99)), 2)

    def test_zero_score_chunks_are_not_returned(self):
        self.assertEqual(self.idx.search("kubernetes"), [])

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.idx.search("refund", k=0)

    def test_scores_are_deterministic_for_equal_chunks(self):
        tied = index([chunk("b::0", "same text"), chunk("a::0", "same text")])
        self.assertEqual([hit["chunk_id"] for hit in tied.search("same", k=2)], ["a::0", "b::0"])

    def test_heading_match_beats_a_passing_mention(self):
        idx = index([
            chunk("body::0", "The rate limit applies per key and is measured over a window.", "Other"),
            chunk("head::0", "Some body text about clients.", "Rate limits"),
        ])
        self.assertEqual(idx.search("rate limit", k=2)[0]["chunk_id"], "head::0")


class DocumentAggregationTests(unittest.TestCase):
    def test_chunks_of_one_document_are_collapsed(self):
        idx = index([
            chunk("one::0", "cursor pagination returns a next cursor"),
            chunk("one::1", "cursor pagination avoids duplicate rows"),
            chunk("two::0", "a cursor is a database pointer in this unrelated document"),
        ])
        ranked = idx.search_documents("cursor pagination duplicate", k=5)
        self.assertEqual(ranked[0]["doc_id"], "one")
        self.assertEqual(ranked[0]["chunks_hit"], 2)


class ConstructionTests(unittest.TestCase):
    def test_parameters_are_validated(self):
        with self.assertRaises(ValueError):
            BM25Index(k1=0)
        with self.assertRaises(ValueError):
            BM25Index(b=1.4)

    def test_search_before_build_is_an_error(self):
        idx = BM25Index().add([chunk("a::0", "text")])
        with self.assertRaises(RuntimeError):
            idx.search("text")

    def test_cannot_add_after_build(self):
        idx = index([chunk("a::0", "text")])
        with self.assertRaises(RuntimeError):
            idx.add([chunk("b::0", "more")])

    def test_empty_index_is_constructible(self):
        idx = BM25Index.from_chunks([])
        self.assertEqual(idx.search("anything"), [])
        self.assertEqual(idx.stats()["chunks"], 0)

    def test_stats_report_the_parameters(self):
        idx = index([chunk("a::0", "alpha beta gamma")], k1=1.2, b=0.5)
        stats = idx.stats()
        self.assertEqual(stats["chunks"], 1)
        # alpha, beta, gamma plus the heading term "notes"
        self.assertEqual(stats["terms"], 4)
        self.assertEqual(stats["k1"], 1.2)
        self.assertEqual(stats["b"], 0.5)

    def test_indexing_a_real_document_produces_chunks(self):
        doc = {"id": "r", "title": "Retention", "text": "## Window\nRaw events are kept for 30 days."}
        chunks = chunk_document(doc)
        idx = index(chunks)
        self.assertEqual(idx.stats()["documents"], 1)
        # The heading is searchable even though the body never repeats it.
        self.assertGreater(idx.search("window", k=1)[0]["score"], 0)


if __name__ == "__main__":
    unittest.main()
