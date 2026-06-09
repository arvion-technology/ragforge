"""Answer composition: citations must point at the text they quote."""

import unittest

from ragforge.answer import build_prompt, compose_answer, format_answer


def hit(chunk_id, text, heading="Section", doc_id=None, score=1.0):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id or chunk_id.split("::")[0],
        "title": "Test doc",
        "heading": heading,
        "text": text,
        "score": score,
    }


class ComposeAnswerTests(unittest.TestCase):
    def test_each_sentence_carries_a_marker_that_resolves(self):
        hits = [
            hit("a::0", "Refunds are returned within ten business days.", score=2.0),
            hit("b::0", "Refunds require the original payment method.", score=1.5),
        ]
        result = compose_answer("How long do refunds take?", hits)
        self.assertTrue(result["grounded"])
        self.assertEqual(len(result["citations"]), 2)
        for citation in result["citations"]:
            self.assertIn(f"[{citation['marker']}]", result["answer"])
            self.assertEqual(citation["quote"], citation["quote"].strip())

    def test_quoted_text_comes_from_the_retrieved_chunk(self):
        hits = [hit("a::0", "Seat changes are prorated by whole days.")]
        result = compose_answer("How are seat changes prorated?", hits)
        quote = result["citations"][0]["quote"]
        self.assertIn(quote, hits[0]["text"])

    def test_citation_cap_is_honoured(self):
        hits = [hit(f"d{i}::0", "refund refund refund") for i in range(5)]
        result = compose_answer("refund", hits, max_citations=2)
        self.assertEqual(len(result["citations"]), 2)

    def test_unmatched_question_returns_an_explicit_failure(self):
        result = compose_answer("What is the airspeed of a swallow?", [hit("a::0", "Billing text only.")])
        self.assertFalse(result["grounded"])
        self.assertEqual(result["citations"], [])
        self.assertIn("No passage", result["answer"])

    def test_no_hits_at_all_is_handled(self):
        result = compose_answer("anything", [])
        self.assertFalse(result["grounded"])
        self.assertEqual(result["coverage"], 0.0)

    def test_duration_questions_prefer_a_sentence_with_a_time_unit(self):
        text = (
            "Every webhook delivery is signed. The signature covers the raw body. "
            "Reject the delivery when the timestamp is more than five minutes away, because that window bounds replay."
        )
        result = compose_answer("How long is the webhook timestamp tolerance?", [hit("a::0", text)])
        self.assertIn("five minutes", result["citations"][0]["quote"])

    def test_count_questions_prefer_a_sentence_with_a_number(self):
        text = "Batch requests are supported. A client can send up to 500 events in one request."
        result = compose_answer("How many events can one request contain?", [hit("a::0", text)])
        self.assertIn("500", result["citations"][0]["quote"])

    def test_weak_hits_are_not_padded_into_the_answer(self):
        hits = [
            hit("a::0", "The refund window is 30 days.", score=10.0),
            hit("b::0", "A refund is mentioned in passing here.", score=2.0),
        ]
        result = compose_answer("refund window", hits)
        self.assertEqual(len(result["citations"]), 1)

    def test_a_strong_second_hit_is_kept(self):
        hits = [
            hit("a::0", "The refund window is 30 days.", score=10.0),
            hit("b::0", "A refund window may be extended by support.", score=6.0),
        ]
        result = compose_answer("refund window", hits)
        self.assertEqual(len(result["citations"]), 2)

    def test_coverage_is_computed_from_cited_text(self):
        result = compose_answer("refund window", [hit("a::0", "The refund window is 30 days.")])
        self.assertEqual(result["coverage"], 1.0)

    def test_prompt_contains_every_citation_marker(self):
        result = compose_answer("refund window", [hit("a::0", "The refund window is 30 days.")])
        prompt = build_prompt(result["question"], result)
        for citation in result["citations"]:
            self.assertIn(f"[{citation['marker']}]", prompt)

    def test_plain_text_format_lists_sources(self):
        result = compose_answer("refund window", [hit("a::0", "The refund window is 30 days.")])
        rendered = format_answer(result)
        self.assertIn("Sources", rendered)
        self.assertIn("coverage:", rendered)


if __name__ == "__main__":
    unittest.main()
