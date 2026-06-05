// Parity and behaviour tests for the browser port of the retriever.
// Run with: node --test test/
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  BM25Index,
  buildRetriever,
  chunkDocument,
  composeAnswer,
  evaluateAt,
  splitSections,
  stem,
  tokenize,
} from "../src/retrieval.js";

const readJson = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));
const corpus = readJson("../corpus/documents.json");
const labels = readJson("../corpus/eval.json");
const fixture = readJson("./fixtures/bm25-parity.json");
const baseline = readJson("../corpus/baseline.json");

const retriever = buildRetriever(corpus, fixture.options);

test("tokeniser matches the Python rules", () => {
  assert.deepEqual(tokenize("How Long Is The Window?"), ["long", "window"]);
  assert.equal(stem("policies"), "policy");
  assert.equal(stem("access"), "access");
  assert.deepEqual(tokenize("x-ratelimit-limit header"), ["x-ratelimit-limit", "header"]);
  assert.deepEqual(tokenize("POST /v1/events, [202]."), ["post", "v1", "event", "202"]);
});

test("heading-aware chunking keeps sections together", () => {
  const doc = { id: "d", title: "Doc", text: "Intro.\n\n## Refunds\nTen days.\n\n## Seats\nA seat is a user." };
  const sections = splitSections(doc);
  assert.deepEqual(sections.map((s) => s[0]), ["Doc", "Refunds", "Seats"]);
  const chunks = chunkDocument(doc, { size: 200 });
  assert.equal(chunks.length, 1);
  assert.deepEqual(chunks[0].headings, ["Doc", "Refunds", "Seats"]);
});

test("fixed chunking overlaps by the configured token count", () => {
  const words = Array.from({ length: 30 }, (_, i) => `w${i}`).join(" ");
  const chunks = chunkDocument({ id: "f", title: "F", text: words }, { size: 10, overlap: 4, strategy: "fixed" });
  const first = chunks[0].text.split(" ");
  const second = chunks[1].text.split(" ");
  assert.equal(first.length, 10);
  assert.deepEqual(first.slice(-4), second.slice(0, 4));
});

test("chunk count and index shape match the Python fixture", () => {
  assert.equal(retriever.chunks.length, fixture.index.chunks);
  const stats = retriever.index.stats();
  assert.equal(stats.documents, fixture.index.documents);
  assert.equal(stats.terms, fixture.index.terms);
  assert.equal(stats.avgdl, fixture.index.avgdl);
});

test("every query scores like the Python fixture", () => {
  for (const entry of fixture.queries) {
    const actual = retriever.search(entry.query, { k: 5 });
    assert.equal(actual.length, entry.results.length, entry.query);
    entry.results.forEach((expected, index) => {
      assert.equal(actual[index].chunk_id, expected.chunk_id, `${entry.query} rank ${index}`);
      assert.ok(
        Math.abs(actual[index].score - expected.score) <= 2e-6,
        `${entry.query}: ${actual[index].score} vs ${expected.score}`,
      );
    });
  }
});

test("score explanations sum to the score", () => {
  const hit = retriever.search("webhook signature tolerance", { k: 1 })[0];
  const sum = Object.values(hit.terms).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - hit.score) < 1e-6);
});

test("searchDocuments collapses multiple chunks of one document", () => {
  const idx = BM25Index.fromChunks([
    { id: "one::0", doc_id: "one", title: "One", heading: "H", text: "cursor pagination returns a next cursor" },
    { id: "one::1", doc_id: "one", title: "One", heading: "H", text: "cursor pagination avoids duplicate rows" },
    { id: "two::0", doc_id: "two", title: "Two", heading: "H", text: "a cursor is a pointer in this unrelated document" },
  ]);
  const ranked = idx.searchDocuments("cursor pagination duplicate", { k: 5 });
  assert.equal(ranked[0].doc_id, "one");
  assert.equal(ranked[0].chunks_hit, 2);
});

test("answers carry resolvable citation markers", () => {
  const answer = retriever.ask("How long is the webhook signature timestamp tolerance?");
  assert.equal(answer.grounded, true);
  assert.equal(answer.citations[0].doc_id, "webhooks-signing");
  assert.match(answer.citations[0].quote, /five minutes/);
  for (const citation of answer.citations) assert.match(answer.answer, new RegExp(`\\[${citation.marker}\\]`));
});

test("unknown questions fail explicitly instead of inventing an answer", () => {
  const answer = composeAnswer("What is the airspeed of an unladen swallow?", retriever.search("swallow"));
  assert.equal(answer.grounded, false);
  assert.deepEqual(answer.citations, []);
});

test("live evaluation matches the committed Python baseline", () => {
  const report = evaluateAt(retriever, labels, [1, 3, 5, 10]);
  assert.equal(report.rows.length, baseline.rows.length);
  baseline.rows.forEach((expected, index) => {
    assert.equal(report.rows[index].k, expected.k);
    for (const key of ["hit_rate", "mrr", "recall", "queries"]) {
      assert.ok(
        Math.abs(report.rows[index][key] - expected[key]) <= 1e-4,
        `k=${expected.k} ${key}: ${report.rows[index][key]} vs ${expected[key]}`,
      );
    }
  });
  assert.equal(report.detail.queries, baseline.labels);
  assert.equal(baseline.misses_at_1.length, 3);
  assert.equal(baseline.misses_at_1[0].retrieved.length, 1);
});
