"""Ingestion from Markdown, HTML, plain text and a JSON corpus."""

import json
import tempfile
import unittest
from pathlib import Path

from ragforge.ingest import (
    IngestError,
    load_documents,
    load_file,
    load_json_corpus,
    parse_html,
    parse_markdown,
)


class MarkdownTests(unittest.TestCase):
    def test_first_h1_becomes_the_title(self):
        title, body = parse_markdown("# Webhooks\n\n## Signing\nHMAC.", fallback_title="fallback")
        self.assertEqual(title, "Webhooks")
        self.assertIn("## Signing", body)
        self.assertNotIn("# Webhooks", body)

    def test_missing_h1_uses_the_fallback(self):
        title, body = parse_markdown("## Only a section", fallback_title="From filename")
        self.assertEqual(title, "From filename")
        self.assertEqual(body, "## Only a section")


class HtmlTests(unittest.TestCase):
    def test_extracts_title_and_text(self):
        title, text = parse_html("<html><head><title>Pricing</title></head><body><h1>Plans</h1><p>Three plans exist.</p></body></html>")
        self.assertEqual(title, "Pricing")
        self.assertIn("Three plans exist.", text)

    def test_drops_script_and_style_content(self):
        _, text = parse_html("<p>Keep this.</p><script>var secret = 1;</script><style>.a{color:red}</style>")
        self.assertIn("Keep this.", text)
        self.assertNotIn("secret", text)
        self.assertNotIn("color", text)


class JsonCorpusTests(unittest.TestCase):
    def test_accepts_a_list(self):
        docs = load_json_corpus(json.dumps([{"id": "a", "title": "A", "text": "Alpha"}]))
        self.assertEqual(docs[0]["id"], "a")

    def test_accepts_a_documents_object(self):
        docs = load_json_corpus(json.dumps({"documents": [{"id": "a", "text": "Alpha"}]}))
        self.assertEqual(docs[0]["title"], "a")

    def test_rejects_a_document_without_text(self):
        with self.assertRaises(IngestError):
            load_json_corpus(json.dumps([{"id": "a", "text": ""}]))

    def test_rejects_malformed_json(self):
        with self.assertRaises(IngestError):
            load_json_corpus("{ nope")


class DirectoryLoadingTests(unittest.TestCase):
    def test_loads_supported_files_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.md").write_text("# Beta\n\nBeta body.", encoding="utf-8")
            (root / "a.txt").write_text("Alpha body.", encoding="utf-8")
            (root / "c.html").write_text("<title>Gamma</title><p>Gamma body.</p>", encoding="utf-8")
            (root / "skip.csv").write_text("ignored", encoding="utf-8")
            docs = load_documents(root)
            self.assertEqual([doc["id"] for doc in docs], ["a", "b", "c"])
            self.assertEqual(docs[1]["title"], "Beta")

    def test_unsupported_single_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("a,b", encoding="utf-8")
            with self.assertRaises(IngestError):
                load_file(path)

    def test_missing_path_raises(self):
        with self.assertRaises(IngestError):
            load_documents("/nonexistent/corpus.json")

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(IngestError):
                load_documents(tmp)


class BundledCorpusTests(unittest.TestCase):
    def test_the_demo_corpus_loads(self):
        root = Path(__file__).resolve().parent.parent
        docs = load_documents(root / "corpus" / "documents.json")
        self.assertEqual(len(docs), 20)
        self.assertTrue(all(doc["title"] and doc["text"] for doc in docs))


if __name__ == "__main__":
    unittest.main()
