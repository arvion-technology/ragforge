"""The retriever ties ingestion, chunking, retrieval and answers together."""

import unittest
from pathlib import Path

from ragforge.retriever import Retriever, load_eval_labels


ROOT = Path(__file__).resolve().parent.parent


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever.from_corpus_file(ROOT / "corpus" / "documents.json")

    def test_every_document_is_indexed(self):
        self.assertEqual(len(self.retriever.documents), 20)
        self.assertGreaterEqual(len(self.retriever.chunks), 20)

    def test_stats_expose_the_index_shape(self):
        stats = self.retriever.stats()
        self.assertEqual(stats["index"]["documents"], 20)
        self.assertGreater(stats["index"]["terms"], 200)
        self.assertEqual(stats["options"]["strategy"], "heading")

    def test_manifest_lists_one_row_per_document(self):
        manifest = self.retriever.manifest()
        self.assertEqual(len(manifest), 20)
        self.assertTrue(all(row["chunks"] >= 1 for row in manifest))

    def test_document_lookup(self):
        found = self.retriever.document("webhooks-signing")
        self.assertIsNotNone(found)
        self.assertIn("HMAC", found["text"])
        self.assertIsNone(self.retriever.document("missing"))

    def test_eval_labels_reference_existing_documents(self):
        labels = load_eval_labels(ROOT / "corpus" / "eval.json")
        self.assertGreaterEqual(len(labels), 30)
        known = {doc["id"] for doc in self.retriever.documents}
        for label in labels:
            self.assertTrue(label["question"])
            self.assertTrue(label["relevant"])
            self.assertTrue(set(label["relevant"]) <= known)

    def test_empty_document_list_is_rejected(self):
        with self.assertRaises(ValueError):
            Retriever([])


class QueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever.from_corpus_file(ROOT / "corpus" / "documents.json")

    def test_search_is_ranked_by_score(self):
        hits = self.retriever.search("webhook signature", k=5)
        scores = [hit["score"] for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ask_returns_a_grounded_answer_with_the_right_document(self):
        answer = self.retriever.ask("How long is the webhook signature timestamp tolerance?")
        self.assertTrue(answer["grounded"])
        self.assertEqual(answer["citations"][0]["doc_id"], "webhooks-signing")
        self.assertIn("five minutes", answer["citations"][0]["quote"])

    def test_ask_includes_the_retrieval_trace(self):
        answer = self.retriever.ask("How many events fit in one request?", k=3)
        self.assertEqual(len(answer["retrieval"]), 3)
        self.assertTrue(all("score" in hit and "terms" in hit for hit in answer["retrieval"]))

    def test_out_of_corpus_question_is_not_answered(self):
        answer = self.retriever.ask("What is the airspeed velocity of an unladen swallow?")
        self.assertFalse(answer["grounded"])
        self.assertEqual(answer["citations"], [])


if __name__ == "__main__":
    unittest.main()
