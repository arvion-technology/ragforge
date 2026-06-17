"""Tokeniser behaviour that retrieval depends on."""

import unittest

from ragforge.text import STOPWORDS, sentence_split, stem, tokenize, unique_terms


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_drops_stopwords(self):
        self.assertEqual(tokenize("How Long Is The Window?"), ["long", "window"])

    def test_keeps_stopwords_when_asked(self):
        self.assertEqual(tokenize("the window", keep_stopwords=True), ["the", "window"])

    def test_collapses_simple_plurals(self):
        self.assertEqual(tokenize("policies refunds keys"), ["policy", "refund", "key"])

    def test_does_not_strip_double_s(self):
        self.assertEqual(stem("access"), "access")
        self.assertEqual(stem("status"), "status")

    def test_keeps_hyphenated_identifiers_together(self):
        self.assertEqual(tokenize("x-ratelimit-limit header"), ["x-ratelimit-limit", "header"])

    def test_drops_punctuation_only_tokens(self):
        self.assertEqual(tokenize("POST /v1/events, [202]."), ["post", "v1", "event", "202"])

    def test_empty_input_is_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(unique_terms(""), [])

    def test_unique_terms_are_sorted_and_deduplicated(self):
        self.assertEqual(unique_terms("retry retried retry"), ["retried", "retry"])

    def test_stopword_set_does_not_contain_content_words(self):
        for word in ("webhook", "refund", "retention", "saml"):
            self.assertNotIn(word, STOPWORDS)


class SentenceSplitTests(unittest.TestCase):
    def test_splits_plain_prose(self):
        text = "First sentence. Second sentence! Third one?"
        self.assertEqual(sentence_split(text), ["First sentence.", "Second sentence!", "Third one?"])

    def test_keeps_abbreviations_intact(self):
        text = "Plans cost approx. 30 dollars. Billing is monthly."
        self.assertEqual(len(sentence_split(text)), 2)

    def test_paragraph_break_is_a_boundary_even_without_punctuation(self):
        parts = sentence_split("A line without a stop\n\nAnother line")
        self.assertEqual(parts, ["A line without a stop", "Another line"])

    def test_markdown_heading_starts_a_new_sentence(self):
        parts = sentence_split("Intro ends here.\n## Verification\nCheck the digest")
        self.assertEqual(parts[0], "Intro ends here.")
        self.assertIn("Verification", parts[1])


if __name__ == "__main__":
    unittest.main()
