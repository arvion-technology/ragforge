"""Cross-language parity: the JavaScript port must score like this package.

The fixture in ``test/fixtures/bm25-parity.json`` is produced by
``tools/export_fixture.py``. The Node test suite reads the same file, so a
change to either implementation that is not mirrored in the other fails one of
the two suites instead of silently shipping two different retrieval systems.
"""

import json
import unittest
from pathlib import Path

from ragforge.retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "test" / "fixtures" / "bm25-parity.json"


class ParityFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.retriever = Retriever.from_corpus_file(
            ROOT / cls.fixture["corpus"],
            **cls.fixture["options"],
        )

    def test_index_shape_matches_the_fixture(self):
        stats = self.retriever.index.stats()
        for key in ("chunks", "documents", "terms", "avgdl"):
            self.assertEqual(stats[key], self.fixture["index"][key], key)

    def test_every_query_matches_chunk_ids_and_scores(self):
        self.assertGreaterEqual(len(self.fixture["queries"]), 5)
        for entry in self.fixture["queries"]:
            actual = [
                {"chunk_id": hit["chunk_id"], "score": round(hit["score"], 6)}
                for hit in self.retriever.search(entry["query"], k=5)
            ]
            self.assertEqual(actual, entry["results"], entry["query"])

    def test_fixture_queries_are_not_empty(self):
        for entry in self.fixture["queries"]:
            self.assertTrue(entry["results"], entry["query"])
            self.assertTrue(all("::" in row["chunk_id"] for row in entry["results"]))


if __name__ == "__main__":
    unittest.main()
