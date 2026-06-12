"""Evaluation maths, including the cases where a metric is misleading alone."""

import unittest

from ragforge.evaluate import (
    answer_quality,
    citation_precision,
    evaluate,
    evaluate_at,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


class MetricPrimitiveTests(unittest.TestCase):
    def test_hit_at_k(self):
        self.assertEqual(hit_at_k(["a", "b"], {"b"}, 2), 1)
        self.assertEqual(hit_at_k(["a", "b"], {"b"}, 1), 0)

    def test_reciprocal_rank_counts_position(self):
        self.assertEqual(reciprocal_rank(["x", "a"], {"a"}, 5), 0.5)
        self.assertEqual(reciprocal_rank(["a"], {"a"}, 5), 1.0)
        self.assertEqual(reciprocal_rank(["x"], {"a"}, 5), 0.0)

    def test_recall_accounts_for_multiple_relevant_documents(self):
        self.assertEqual(recall_at_k(["a", "b"], {"a", "b"}, 2), 1.0)
        self.assertEqual(recall_at_k(["a"], {"a", "b"}, 2), 0.5)
        self.assertEqual(recall_at_k(["a"], set(), 2), 0.0)

    def test_citation_precision(self):
        citations = [{"doc_id": "a"}, {"doc_id": "z"}]
        self.assertEqual(citation_precision(citations, {"a"}), 0.5)
        self.assertEqual(citation_precision([], {"a"}), 0.0)


class FakeIndex:
    """Minimal stand-in so the evaluation harness is tested independently of BM25."""

    def __init__(self, rankings):
        self.rankings = rankings

    def search_documents(self, question, k=5):
        return [{"doc_id": doc_id} for doc_id in self.rankings.get(question, [])[:k]]


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.index = FakeIndex({
            "hit first": ["a", "b"],
            "hit second": ["x", "a"],
            "miss": ["x", "y"],
        })
        self.labels = [
            {"id": "1", "question": "hit first", "relevant": ["a"]},
            {"id": "2", "question": "hit second", "relevant": ["a"]},
            {"id": "3", "question": "miss", "relevant": ["a"]},
        ]

    def test_aggregate_metrics(self):
        report = evaluate(self.index, self.labels, k=5)
        self.assertEqual(report["queries"], 3)
        self.assertAlmostEqual(report["hit_rate"], 2 / 3, places=4)
        self.assertAlmostEqual(report["mrr"], (1 + 0.5) / 3, places=4)
        self.assertEqual(len(report["misses"]), 1)
        self.assertEqual(report["misses"][0]["question"], "miss")

    def test_cutoff_changes_the_numbers(self):
        self.assertAlmostEqual(evaluate(self.index, self.labels, k=1)["hit_rate"], 1 / 3, places=4)
        self.assertAlmostEqual(evaluate(self.index, self.labels, k=2)["hit_rate"], 2 / 3, places=4)

    def test_evaluate_at_returns_a_curve(self):
        report = evaluate_at(self.index, self.labels, cutoffs=(1, 2, 5))
        self.assertEqual(report["cutoffs"], [1, 2, 5])
        self.assertEqual(report["rows"][0]["k"], 1)
        self.assertEqual(report["detail"]["k"], 5)

    def test_empty_label_set_does_not_divide_by_zero(self):
        report = evaluate(self.index, [], k=5)
        self.assertEqual(report["queries"], 0)
        self.assertEqual(report["hit_rate"], 0.0)


class AnswerQualityTests(unittest.TestCase):
    def test_scores_citation_precision_and_coverage(self):
        answer = {
            "question": "refund window",
            "citations": [{"doc_id": "billing", "quote": "The refund window is 30 days."}],
            "coverage": 1.0,
            "grounded": True,
        }
        quality = answer_quality(answer, {"billing"})
        self.assertTrue(quality["grounded"])
        self.assertEqual(quality["citation_precision"], 1.0)
        self.assertEqual(quality["question_terms_covered"], 2)

    def test_ungrounded_answer_scores_zero(self):
        quality = answer_quality({"question": "x", "citations": [], "coverage": 0.0, "grounded": False}, {"a"})
        self.assertFalse(quality["grounded"])
        self.assertEqual(quality["citations"], 0)


if __name__ == "__main__":
    unittest.main()
