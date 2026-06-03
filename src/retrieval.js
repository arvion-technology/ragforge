// JavaScript port of ragforge's tokeniser, chunker and BM25 retriever.
//
// This file exists because the demo has to run the retrieval maths in the
// browser instead of screenshotting a Python session. It is a port, not a
// reimplementation with different ideas: tokenisation, chunk boundaries, IDF and
// scoring follow ragforge/*.py line for line, and test/bm25.test.mjs checks it
// against test/fixtures/bm25-parity.json, generated from the Python package.
//
// If you change the Python side, run `python3 tools/export_fixture.py` and make
// the change here too; both suites will fail until they agree.

export const STOPWORDS = new Set(
  `a an and are as at be been but by can could did do does for from had has
   have how i if in into is it its may might more most no not of on or our
   should so than that the their them then there these they this to was we
   were what when where which who why will with would you your`.split(/\s+/),
);

const TOKEN_RE = /[a-z0-9]+(?:[._-][a-z0-9]+)*/g;

export function stem(word) {
  if (word.length <= 3) return word;
  if (word.endsWith("ies") && word.length > 4) return word.slice(0, -3) + "y";
  if (/(sses|shes|ches|xes|zes)$/.test(word)) return word.slice(0, -2);
  if (word.endsWith("s") && !/(ss|us|is)$/.test(word)) return word.slice(0, -1);
  return word;
}

export function tokenize(text, { keepStopwords = false } = {}) {
  if (!text) return [];
  const raw = String(text).toLowerCase().match(TOKEN_RE) || [];
  // Stopwords are removed before stemming, exactly as in ragforge/text.py.
  // Stemming first would turn "does" into "doe", which is not a stopword, and
  // the index would carry a term the Python package never sees.
  const kept = keepStopwords ? raw : raw.filter((token) => !STOPWORDS.has(token));
  return kept.map(stem);
}

export function uniqueTerms(text) {
  return [...new Set(tokenize(text))].sort();
}

// Mirrors ragforge.text.sentence_split. Only used when a single section is
// larger than the chunk window, but kept faithful so custom corpora behave the
// same in the browser as in the CLI.
export function sentenceSplit(text) {
  if (!text) return [];
  let protectedText = text;
  for (const abbreviation of ["approx.", "e.g.", "i.e.", "vs.", "etc.", "No."]) {
    protectedText = protectedText.split(abbreviation).join(abbreviation.replace(".", "\u2024"));
  }
  const parts = [];
  const paragraphs = protectedText.split(/\n\s*\n|\n(?=#{1,3}\s)/);
  for (let paragraph of paragraphs) {
    paragraph = paragraph.trim().replace(/^#+/, "").trim();
    if (!paragraph) continue;
    for (let sentence of paragraph.split(/(?<=[.!?])\s+(?=[A-Z0-9"'(])/)) {
      sentence = sentence.replace(/\u2024/g, ".").trim();
      if (sentence) parts.push(sentence);
    }
  }
  return parts;
}

const HEADING_RE = /^##\s+(.+?)\s*$/;

export function splitSections(doc) {
  const text = String(doc.text || "").trim();
  if (!text) return [];
  const title = String(doc.title || doc.id || "document").trim();
  const lines = text.split("\n");
  const sections = [];
  let heading = null;
  let buffer = [];
  const flush = () => {
    const body = buffer.join("\n").trim();
    if (heading === null) {
      if (body) sections.push([title, body]);
    } else if (body) {
      sections.push([heading, body]);
    } else if (!sections.length) {
      sections.push([heading, ""]);
    }
    heading = null;
    buffer = [];
  };
  for (const line of lines) {
    const match = line.match(HEADING_RE);
    if (match) {
      flush();
      heading = match[1].trim();
    } else {
      buffer.push(line);
    }
  }
  flush();
  return sections;
}

function count(text) {
  return tokenize(text, { keepStopwords: true }).length;
}

function windowSections(sections, size, overlap) {
  if (size <= 0) throw new Error("size must be positive");
  if (overlap < 0 || overlap >= size) throw new Error("overlap must be in [0, size)");
  const windows = [];
  let current = [];
  let currentTokens = 0;
  for (const section of sections) {
    const unitTokens = count(section[0]) + count(section[1]);
    if (current.length && currentTokens + unitTokens > size) {
      windows.push(current);
      const tail = [];
      let tailTokens = 0;
      for (let i = current.length - 1; i >= 0; i--) {
        const previous = current[i];
        const previousTokens = count(previous[0]) + count(previous[1]);
        if (previousTokens > overlap || tailTokens + previousTokens > overlap) break;
        tail.unshift(previous);
        tailTokens += previousTokens;
      }
      current = tail;
      currentTokens = tailTokens;
    }
    current.push(section);
    currentTokens += unitTokens;
  }
  if (current.length) windows.push(current);
  return windows;
}

function sentenceWindows(body, size, overlap) {
  const windows = [];
  let current = [];
  let currentTokens = 0;
  for (const sentence of sentenceSplit(body)) {
    const tokens = count(sentence);
    if (current.length && currentTokens + tokens > size) {
      windows.push(current);
      const tail = [];
      let tailTokens = 0;
      for (let i = current.length - 1; i >= 0; i--) {
        const previousTokens = count(current[i]);
        if (previousTokens > overlap || tailTokens + previousTokens > overlap) break;
        tail.unshift(current[i]);
        tailTokens += previousTokens;
      }
      current = tail;
      currentTokens = tailTokens;
    }
    current.push(sentence);
    currentTokens += tokens;
  }
  if (current.length) windows.push(current);
  return windows.map((window) => window.join(" ").trim());
}

export function chunkDocument(doc, { size = 120, overlap = 30, strategy = "heading" } = {}) {
  const docId = doc.id || "doc";
  const title = doc.title || docId;
  const kind = doc.kind || "document";
  const pieces = [];

  if (strategy === "fixed") {
    const words = String(doc.text || "").split(/\s+/).filter(Boolean);
    const step = Math.max(1, size - overlap);
    for (let start = 0; start < words.length; start += step) {
      const body = words.slice(start, start + size).join(" ").trim();
      if (!body) break;
      pieces.push({ headings: [title], text: body });
      if (start + size >= words.length) break;
    }
  } else if (strategy === "heading") {
    for (const window of windowSections(splitSections(doc), size, overlap)) {
      const headings = window.map((section) => section[0]);
      const bodies = window.map((section) => section[1]).filter(Boolean);
      const text = bodies.join("\n\n").trim();
      if (!text) continue;
      if (count(text) > size) {
        for (const part of sentenceWindows(bodies.join("\n\n"), size, overlap)) {
          pieces.push({ headings, text: part });
        }
      } else {
        pieces.push({ headings, text });
      }
    }
  } else {
    throw new Error(`unknown chunk strategy: ${strategy}`);
  }

  return pieces.map((piece, ordinal) => ({
    id: `${docId}::${ordinal}`,
    doc_id: docId,
    title,
    kind,
    heading: piece.headings[0],
    headings: piece.headings,
    ordinal,
    text: piece.text,
  }));
}

export function chunkDocuments(documents, options = {}) {
  return documents.flatMap((doc) => chunkDocument(doc, options));
}

const round = (value, digits = 6) => {
  const factor = 10 ** digits;
  return Math.round((value + Number.EPSILON) * factor) / factor;
};

export class BM25Index {
  constructor({ k1 = 1.5, b = 0.75 } = {}) {
    if (k1 <= 0) throw new Error("k1 must be positive");
    if (b < 0 || b > 1) throw new Error("b must be in [0, 1]");
    this.k1 = k1;
    this.b = b;
    this.chunks = [];
    this._tokens = [];
    this._freqs = [];
    this._df = new Map();
    this._idf = new Map();
    this._avgdl = 0;
    this._built = false;
  }

  add(chunks) {
    if (this._built) throw new Error("cannot add chunks after build()");
    for (const chunk of chunks) {
      const heading = tokenize(chunk.heading || "");
      const tokens = [...heading, ...heading, ...tokenize(chunk.text || "")];
      this.chunks.push(chunk);
      this._tokens.push(tokens);
      const freqs = new Map();
      for (const token of tokens) freqs.set(token, (freqs.get(token) || 0) + 1);
      this._freqs.push(freqs);
      for (const token of new Set(tokens)) this._df.set(token, (this._df.get(token) || 0) + 1);
    }
    return this;
  }

  build() {
    const total = this._tokens.reduce((sum, tokens) => sum + tokens.length, 0);
    this._avgdl = this._tokens.length ? total / this._tokens.length : 0;
    const n = this._tokens.length;
    for (const [term, df] of this._df) {
      this._idf.set(term, Math.max(0, Math.log(1 + (n - df + 0.5) / (df + 0.5))));
    }
    this._built = true;
    return this;
  }

  static fromChunks(chunks, options = {}) {
    return new BM25Index(options).add(chunks).build();
  }

  _termScore(term, termFreq, length) {
    if (!termFreq) return 0;
    const idf = this._idf.get(term) || 0;
    const norm = this.k1 * (1 - this.b + this.b * (this._avgdl ? length / this._avgdl : 0));
    return (idf * (termFreq * (this.k1 + 1))) / (termFreq + norm);
  }

  score(query) {
    if (!this._built) throw new Error("call build() before search()");
    const terms = [...new Set(tokenize(query))];
    const results = [];
    this.chunks.forEach((chunk, i) => {
      const freqs = this._freqs[i];
      const length = this._tokens[i].length;
      const contributions = {};
      let total = 0;
      for (const term of terms) {
        const value = this._termScore(term, freqs.get(term) || 0, length);
        if (value) {
          contributions[term] = round(value);
          total += value;
        }
      }
      results.push({
        chunk_id: chunk.id,
        doc_id: chunk.doc_id,
        title: chunk.title || "",
        heading: chunk.heading || "",
        headings: chunk.headings || [chunk.heading || ""],
        kind: chunk.kind || "",
        text: chunk.text || "",
        score: round(total),
        terms: contributions,
      });
    });
    results.sort((a, b) => b.score - a.score || (a.chunk_id < b.chunk_id ? -1 : a.chunk_id > b.chunk_id ? 1 : 0));
    return results;
  }

  search(query, { k = 5 } = {}) {
    if (k <= 0) throw new Error("k must be positive");
    return this.score(query).filter((row) => row.score > 0).slice(0, k);
  }

  searchDocuments(query, { k = 5 } = {}) {
    const best = new Map();
    for (const hit of this.score(query)) {
      if (hit.score <= 0) continue;
      const current = best.get(hit.doc_id);
      if (!current) best.set(hit.doc_id, { ...hit, chunks_hit: 1 });
      else current.chunks_hit += 1;
    }
    return [...best.values()].sort((a, b) => b.score - a.score || (a.doc_id < b.doc_id ? -1 : 1)).slice(0, k);
  }

  stats() {
    return {
      chunks: this.chunks.length,
      documents: new Set(this.chunks.map((chunk) => chunk.doc_id)).size,
      terms: this._df.size,
      avgdl: Math.round(this._avgdl * 100) / 100,
      k1: this.k1,
      b: this.b,
    };
  }
}

export function buildRetriever(documents, options = {}) {
  const { size = 120, overlap = 30, strategy = "heading", k1 = 1.5, b = 0.75 } = options;
  const chunks = chunkDocuments(documents, { size, overlap, strategy });
  const index = BM25Index.fromChunks(chunks, { k1, b });
  return {
    documents,
    chunks,
    index,
    options: { size, overlap, strategy, k1, b },
    search: (query, opts) => index.search(query, opts),
    searchDocuments: (query, opts) => index.searchDocuments(query, opts),
    ask: (question, opts = {}) => composeAnswer(question, index.search(question, opts)),
  };
}

// ---- answer composition ---------------------------------------------------
// Same rules as ragforge/answer.py, including the question-shape bonus that
// keeps "how long" answers from quoting a sentence with no duration in it.

const NUMBER = "(?:\\d+(?:\\.\\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty|forty|fifty|sixty|ninety|hundred)";
const UNIT = "(?:second|minute|hour|day|week|month|year|business\\s+day)";
const DURATION_SHAPE = new RegExp(`\\b${NUMBER}\\s+(?:business\\s+)?${UNIT}s?\\b`, "i");
const COUNT_SHAPE = /\b\d+\b/;
const DURATION_QUESTION = /\bhow (?:long|soon|much time)\b|\btolerance\b|\bwindow\b|\bretention\b/i;
const COUNT_QUESTION = /\bhow many\b|\bmaximum number\b|\bhighest number\b/i;

function shapeBonus(question, sentence) {
  if (DURATION_QUESTION.test(question) && DURATION_SHAPE.test(sentence)) return 1.5;
  if (COUNT_QUESTION.test(question) && COUNT_SHAPE.test(sentence)) return 1.5;
  return 0;
}

function sentenceScore(sentence, queryTerms, question) {
  const terms = new Set(tokenize(sentence));
  let overlap = 0;
  for (const term of terms) if (queryTerms.has(term)) overlap += 1;
  return overlap + shapeBonus(question, sentence);
}

export function composeAnswer(question, hits, { maxCitations = 3 } = {}) {
  const queryTerms = new Set(tokenize(question));
  const citations = [];
  const quotes = [];
  // Weak hits are dropped for the same reason as in ragforge/answer.py: a
  // citation that only shares one common term is noise, not evidence.
  const floor = hits.length ? hits[0].score * 0.4 : 0;
  for (const hit of hits.slice(0, maxCitations)) {
    if (hit.score < floor) continue;
    let best = null;
    let bestScore = 0;
    for (const sentence of sentenceSplit(hit.text || "")) {
      const score = sentenceScore(sentence, queryTerms, question);
      if (score > bestScore) {
        best = sentence;
        bestScore = score;
      }
    }
    if (best === null) continue;
    const marker = citations.length + 1;
    citations.push({
      marker,
      chunk_id: hit.chunk_id,
      doc_id: hit.doc_id,
      title: hit.title,
      heading: hit.heading,
      quote: best,
      score: hit.score,
    });
    quotes.push(`${best} [${marker}]`);
  }

  const covered = new Set();
  for (const citation of citations) {
    for (const term of tokenize(citation.quote)) if (queryTerms.has(term)) covered.add(term);
  }
  const coverage = queryTerms.size ? covered.size / queryTerms.size : 0;

  if (quotes.length) {
    return {
      question,
      answer: quotes.join(" "),
      citations,
      coverage: Math.round(coverage * 10000) / 10000,
      query_terms: [...queryTerms].sort(),
      grounded: true,
      method: "extractive",
    };
  }
  return {
    question,
    answer:
      "No passage in the corpus matched this question. Try different words, or add the source document to the corpus.",
    citations: [],
    coverage: 0,
    query_terms: [...queryTerms].sort(),
    grounded: false,
    method: "extractive",
  };
}

// ---- evaluation -----------------------------------------------------------

export function hitAtK(ranked, relevant, k) {
  return ranked.slice(0, k).some((docId) => relevant.has(docId)) ? 1 : 0;
}

export function reciprocalRank(ranked, relevant, k) {
  const position = ranked.slice(0, k).findIndex((docId) => relevant.has(docId));
  return position === -1 ? 0 : 1 / (position + 1);
}

export function recallAtK(ranked, relevant, k) {
  if (!relevant.size) return 0;
  const found = ranked.slice(0, k).filter((docId) => relevant.has(docId)).length;
  return found / relevant.size;
}

export function evaluate(retriever, labels, { k = 5 } = {}) {
  const perQuery = labels.map((item) => {
    const relevant = new Set(item.relevant || []);
    const hits = retriever.searchDocuments(item.question, { k });
    const ranked = hits.map((hit) => hit.doc_id);
    return {
      id: item.id,
      question: item.question,
      relevant: item.relevant,
      retrieved: ranked,
      hit: hitAtK(ranked, relevant, k),
      rr: round(reciprocalRank(ranked, relevant, k), 4),
      recall: round(recallAtK(ranked, relevant, k), 4),
      top_score: hits.length ? round(hits[0].score, 4) : 0,
    };
  });
  const count = perQuery.length || 1;
  return {
    k,
    queries: perQuery.length,
    hit_rate: round(perQuery.reduce((sum, row) => sum + row.hit, 0) / count, 4),
    mrr: round(perQuery.reduce((sum, row) => sum + row.rr, 0) / count, 4),
    recall: round(perQuery.reduce((sum, row) => sum + row.recall, 0) / count, 4),
    misses: perQuery.filter((row) => !row.hit),
    results: perQuery,
  };
}

export function evaluateAt(retriever, labels, cutoffs = [1, 3, 5, 10]) {
  const detail = evaluate(retriever, labels, { k: cutoffs[cutoffs.length - 1] });
  const rows = cutoffs.map((k) => {
    const report = evaluate(retriever, labels, { k });
    return { k, hit_rate: report.hit_rate, mrr: report.mrr, recall: report.recall, queries: report.queries };
  });
  return { cutoffs, rows, detail };
}
