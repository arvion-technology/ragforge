// RAGForge demo, rendering and interaction only.
// The retrieval maths lives in ../src/retrieval.js, a checked port of the Python
// package. This file builds the index, runs the labelled evaluation and paints
// the result. It makes no network requests beyond the three local JSON files
// fetched at startup.
import {
  buildRetriever,
  composeAnswer,
  evaluate,
  evaluateAt,
  tokenize,
} from "../src/retrieval.js";

const el = A.el;
const $ = A.$;
const $$ = A.$$;

const PROVIDERS = [
  { name: "mock", model: "deterministic-mock", env: "always ready", available: true },
  { name: "openai", model: "gpt-4o-mini", env: "OPENAI_API_KEY", available: false },
  { name: "anthropic", model: "claude-3-5-haiku", env: "ANTHROPIC_API_KEY", available: false },
  { name: "ollama", model: "llama3.2", env: "OLLAMA_HOST", available: false },
];

const SUGGESTIONS = [
  "How long is the webhook signature timestamp tolerance?",
  "What happens when an API rate limit is exceeded?",
  "Which export formats are supported?",
  "How is a mid-cycle upgrade prorated?",
  "What is the recovery time objective for a regional restore?",
];

const state = {
  view: "chat",
  retriever: null,
  documents: [],
  labels: [],
  baseline: null,
  answer: null,
  hits: [],
  labQuery: "webhook signature timestamp tolerance",
};

/* ---------------------------------------------------------------- helpers */

function fetchJson(path) {
  return fetch(new URL(path, import.meta.url)).then((response) => {
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.json();
  });
}

const escapeHtml = (value) =>
  String(value).replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]));

function highlight(text, terms) {
  const escaped = escapeHtml(text);
  const list = [...terms].filter((term) => term.length > 2).sort((a, b) => b.length - a.length);
  if (!list.length) return escaped;
  const pattern = new RegExp(`\\b(${list.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`, "gi");
  return escaped.replace(pattern, "<mark>$1</mark>");
}

const pct = (value) => `${(value * 100).toFixed(1)}%`;
const marker = (value) => el("span", { class: "cite-marker", text: String(value) });

/* --------------------------------------------------------------- rendering */

function renderChrome() {
  const stats = state.retriever.index.stats();
  $("#corpus-line").textContent = `${stats.documents} documents · ${stats.chunks} chunks · ${stats.terms} terms`;
  $("#corpus-tokens").textContent = `${stats.terms} indexed terms`;

  $("#suggestions").replaceChildren(
    ...SUGGESTIONS.slice(0, 4).map((question) =>
      el("button", {
        class: "chip",
        type: "button",
        text: question,
        onclick: () => {
          $("#question").value = question;
          ask(question);
        },
      }),
    ),
  );

  $("#providers").replaceChildren(
    ...PROVIDERS.map((provider) =>
      el("div", { class: "provider" },
        el("span", { class: "who", text: provider.name }),
        el("span", { class: "why", text: provider.env }),
        el("span", { class: `badge ${provider.available ? "green" : ""}`, text: provider.available ? "ready" : "key" }),
      ),
    ),
  );

  renderLab(state.labQuery);
  renderEval();
  renderCorpus();
}

function renderPipeline(answer, queryTokens) {
  const steps = [
    { label: "Query terms", value: queryTokens.length, note: "after stopwords and stemming" },
    { label: "Chunks scored", value: state.retriever.chunks.length, note: "every chunk in the index" },
    { label: "Cited", value: answer.citations.length, note: "sentences quoted" },
    { label: "Coverage", value: pct(answer.coverage), note: "question terms in cited text" },
  ];
  $("#pipeline").replaceChildren(
    ...steps.map((step) =>
      el("div", { class: "step" },
        el("div", { class: "step-label", text: step.label }),
        el("div", { class: "step-value", text: String(step.value) }),
        el("div", { class: "step-note", text: step.note }),
      ),
    ),
  );
}

function renderAnswer(answer, hits) {
  const queryTerms = tokenize(answer.question);
  if (answer.grounded) {
    // Keep the [n] markers visible inside the prose, styled like the citation cards.
    const escaped = highlight(answer.answer, queryTerms);
    $("#answer-text").innerHTML = escaped.replace(/\[(\d)\]/g, '<span class="cite-marker" style="display:inline-grid;width:18px;height:18px;font-size:10px;vertical-align:middle">$1</span>');
  } else {
    $("#answer-text").textContent = answer.answer;
  }
  $("#answer-meta").textContent = answer.grounded
    ? `${answer.citations.length} citation${answer.citations.length === 1 ? "" : "s"} · extractive · no model call`
    : "no grounded passage · the answer path refuses instead of guessing";
  $("#coverage-badge").textContent = `coverage ${pct(answer.coverage)}`;
  $("#coverage-badge").className = "badge " + (answer.coverage > 0.5 ? "green" : "gold");

  $("#citations").replaceChildren(
    ...answer.citations.map((citation) =>
      el("div", { class: "cite" },
        marker(citation.marker),
        el("div", {},
          el("div", { class: "cite-head" },
            citation.title,
            citation.heading === citation.title ? null : el("span", { class: "muted", text: ` / ${citation.heading}` }),
          ),
          el("p", { class: "cite-quote", html: `“${highlight(citation.quote, queryTerms)}”` }),
          el("div", { class: "cite-foot" },
            el("span", { class: "mono", text: citation.chunk_id }),
            " · BM25 ",
            el("span", { class: "mono", text: citation.score.toFixed(2) }),
          ),
        ),
      ),
    ),
  );

  const max = Math.max(...hits.map((hit) => hit.score), 1);
  $("#hits").replaceChildren(
    ...hits.map((hit, index) =>
      el("div", { class: "hit", style: "cursor:pointer", onclick: () => openChunk(hit) },
        el("div", { class: "hit-head" },
          el("span", { class: "badge", text: String(index + 1) }),
          el("span", { class: "hit-title", text: hit.title }),
          el("span", { class: "hit-score", text: hit.score.toFixed(2) }),
        ),
        el("div", { class: "hit-bar" }, el("i", { style: `width:${((hit.score / max) * 100).toFixed(1)}%` })),
        el("div", { class: "term-row" },
          ...Object.entries(hit.terms).map(([term, value]) =>
            el("span", { class: "term" }, `${term} `, el("b", { text: value.toFixed(2) })),
          ),
        ),
        el("div", { class: "hit-text", text: hit.text }),
      ),
    ),
  );

  renderPipeline(answer, queryTerms);
}

function openChunk(hit) {
  A.modal({
    title: `${hit.title} / ${hit.heading}`,
    body: el("div", {},
      el("div", { class: "row small muted mb-2" },
        el("span", { class: "mono", text: hit.chunk_id }),
        el("span", { class: "spacer" }),
        el("span", { class: "mono", text: `score ${hit.score.toFixed(4)}` }),
      ),
      el("p", { class: "doc-body", text: hit.text }),
      el("div", { class: "divider" }),
      el("div", { class: "term-row" },
        ...Object.entries(hit.terms).map(([term, value]) => el("span", { class: "term" }, `${term} `, el("b", { text: value.toFixed(3) }))),
      ),
    ),
  });
}

function renderCorpus() {
  const counts = new Map();
  for (const chunk of state.retriever.chunks) counts.set(chunk.doc_id, (counts.get(chunk.doc_id) || 0) + 1);
  $("#corpus-table").replaceChildren(
    A.dataTable({
      columns: [
        { key: "title", label: "Document", render: (row) => el("span", { class: "strong", text: row.title }) },
        { key: "kind", label: "Kind", render: (row) => el("span", { class: "badge", text: row.kind }) },
        { key: "chunks", label: "Chunks", align: "right", render: (row) => el("span", { class: "mono", text: String(row.chunks) }) },
        { key: "updated", label: "Updated", align: "right", render: (row) => el("span", { class: "mono muted", text: row.updated || ", " }) },
      ],
      rows: state.documents.map((doc) => ({ ...doc, chunks: counts.get(doc.id) || 0 })),
      onRowClick: (row) => {
        A.modal({
          title: row.title,
          body: el("div", { class: "doc-modal" },
            el("div", { class: "row small muted mb-2" },
              el("span", { class: "mono", text: row.id }),
              el("span", { class: "spacer" }),
              el("span", { class: "badge gold", text: `${counts.get(row.id) || 0} chunks` }),
            ),
            el("p", { class: "doc-body", text: row.text }),
          ),
        });
      },
    }),
  );
}

/* ------------------------------------------------------------------- views */

function ask(question) {
  const text = String(question || "").trim();
  if (!text) return;
  const hits = state.retriever.search(text, { k: 3 });
  const answer = composeAnswer(text, hits);
  state.answer = answer;
  state.hits = hits;
  renderAnswer(answer, hits);
}

function renderLab(query) {
  state.labQuery = query;
  $("#lab-question").value = query;
  const terms = tokenize(query);
  $("#lab-tokens").replaceChildren(...terms.map((term) => el("span", { class: "term", text: term })));
  const stats = state.retriever.index.stats();
  $("#lab-params").replaceChildren(
    ...Object.entries({ k1: stats.k1, b: stats.b, chunks: stats.chunks, avgdl: stats.avgdl }).map(([key, value]) =>
      el("div", { style: "display:contents" }, el("dt", { text: key }), el("dd", { text: String(value) })),
    ),
  );

  const hits = state.retriever.search(query, { k: 12 });
  const table = el("table", { class: "table" });
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: "#" }),
    el("th", { text: "Chunk" }),
    el("th", { text: "Matched terms" }),
    el("th", { style: "text-align:right", text: "Score" }),
  )));
  const body = el("tbody");
  if (!hits.length) {
    body.append(el("tr", {}, el("td", { colspan: 4, class: "muted center", style: "padding:30px", text: "No chunk matched any query term." })));
  }
  hits.forEach((hit, index) => {
    const row = el("tr", {
      class: "lab-row",
      onclick: () => {
        $$(".lab-row", table).forEach((other) => other.classList.remove("selected"));
        row.classList.add("selected");
        $("#lab-chunk").replaceChildren(
          el("div", { class: "cite-head", text: `${hit.title} / ${hit.heading}` }),
          el("div", { class: "row small muted", style: "margin:4px 0 10px" },
            el("span", { class: "mono", text: hit.chunk_id }),
            el("span", { class: "spacer" }),
            el("span", { class: "mono", text: `score ${hit.score.toFixed(4)}` }),
          ),
          el("p", { class: "doc-body", text: hit.text }),
        );
      },
    },
      el("td", { class: "mono muted", text: String(index + 1) }),
      el("td", {}, el("div", { class: "strong", text: hit.title }), el("div", { class: "small muted", text: hit.heading })),
      el("td", {}, el("div", { class: "term-row" }, ...Object.entries(hit.terms).map(([term, value]) => el("span", { class: "term" }, `${term} `, el("b", { text: value.toFixed(2) }))))),
      el("td", { class: "num strong", text: hit.score.toFixed(2) }),
    );
    body.append(row);
  });
  table.append(body);
  $("#lab-table").replaceChildren(el("div", { class: "table-wrap" }, table));
  const first = $(".lab-row", table);
  if (first) first.click();
}

function renderEval() {
  const report = evaluateAt(state.retriever, state.labels, [1, 3, 5, 10]);
  const baselineRows = state.baseline?.rows || [];
  const live = report.rows[0];

  $("#eval-kpis").replaceChildren(
    el("div", { class: "card" }, el("div", { class: "stat" },
      el("div", { class: "stat-label", text: "Hit rate @1" }),
      el("div", { class: "stat-value gold", text: pct(live.hit_rate) }),
      el("div", { class: "small muted", text: `${Math.round(live.hit_rate * report.detail.queries)} of ${report.detail.queries} questions` }),
    )),
    el("div", { class: "card" }, el("div", { class: "stat" },
      el("div", { class: "stat-label", text: "MRR @1" }),
      el("div", { class: "stat-value", text: live.mrr.toFixed(3) }),
      el("div", { class: "small muted", text: "mean reciprocal rank" }),
    )),
    el("div", { class: "card" }, el("div", { class: "stat" },
      el("div", { class: "stat-label", text: "Hit rate @5" }),
      el("div", { class: "stat-value green", text: pct(report.rows[2].hit_rate) }),
      el("div", { class: "small muted", text: "what a four-citation answer uses" }),
    )),
    el("div", { class: "card" }, el("div", { class: "stat" },
      el("div", { class: "stat-label", text: "Labelled questions" }),
      el("div", { class: "stat-value", text: String(report.detail.queries) }),
      el("div", { class: "small muted", text: `over ${state.documents.length} documents` }),
    )),
  );

  const parity = baselineRows.length
    ? baselineRows.every((row, index) =>
        ["hit_rate", "mrr", "recall"].every((key) => Math.abs(row[key] - report.rows[index][key]) <= 1e-4),
      )
    : false;
  $("#parity-badge").textContent = baselineRows.length ? (parity ? "matches Python baseline" : "baseline drift") : "no baseline file";
  $("#parity-badge").className = `badge ${parity ? "green" : "red"}`;

  const table = el("table", { class: "table metric-table" });
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: "k" }),
    el("th", { style: "text-align:right", text: "hit rate" }),
    el("th", { style: "text-align:right", text: "MRR" }),
    el("th", { style: "text-align:right", text: "recall" }),
    el("th", { style: "text-align:right", text: "Python hit" }),
    el("th", { style: "text-align:right", text: "Δ" }),
  )));
  const body = el("tbody");
  report.rows.forEach((row, index) => {
    const base = baselineRows[index];
    const delta = base ? row.hit_rate - base.hit_rate : 0;
    body.append(el("tr", {},
      el("td", { class: "mono strong", text: String(row.k) }),
      el("td", { class: "num strong", text: pct(row.hit_rate) }),
      el("td", { class: "num", text: row.mrr.toFixed(3) }),
      el("td", { class: "num", text: pct(row.recall) }),
      el("td", { class: "num muted", text: base ? pct(base.hit_rate) : ", " }),
      el("td", { class: "num" }, el("span", { class: `delta ${Math.abs(delta) < 1e-9 ? "ok" : "bad"}`, text: base ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}` : ", " })),
    ));
  });
  table.append(body);
  $("#eval-table").replaceChildren(el("div", { class: "table-wrap" }, table));

  const misses = evaluate(state.retriever, state.labels, { k: 1 }).misses;
  $("#miss-count").textContent = `${misses.length} of ${state.labels.length}`;

  const multi = state.labels.filter((label) => (label.relevant || []).length > 1).length;
  const exact = state.labels.length - misses.length;
  const mixRows = [
    { label: "Answered with the documentation's own vocabulary", value: exact },
    { label: "Require a paraphrase to retrieve", value: misses.length },
    { label: "Labels with more than one relevant document", value: multi },
  ];
  $("#eval-mix").replaceChildren(
    ...mixRows.map((row) =>
      el("div", {},
        el("div", { class: "row small", style: "margin-bottom:6px" }, el("span", { text: row.label }), el("span", { class: "spacer" }), el("span", { class: "mono muted", text: `${row.value}/${state.labels.length}` })),
        el("div", { class: "progress" }, el("i", { style: `width:${(row.value / state.labels.length) * 100}%` })),
        el("div", { style: "height:12px" }),
      ),
    ),
    el("p", { class: "small muted", style: "margin:0" },
      "BM25 matches terms, not meaning. The misses below are the honest cost of that: each asks a question whose words do not appear in the answering document.",
    ),
  );

  $("#misses").replaceChildren(
    ...misses.map((miss) =>
      el("div", { class: "miss" },
        el("div", {},
          el("div", { class: "miss-q", text: miss.question }),
          el("div", { class: "miss-meta", text: `expected: ${miss.relevant.join(", ")}` }),
        ),
        el("div", { style: "text-align:right" },
          el("div", { class: "miss-got", text: `retrieved: ${miss.retrieved.join(", ") || "nothing"}` }),
          el("div", { class: "miss-meta", text: "top-1 document by BM25" }),
        ),
      ),
    ),
  );
}

/* --------------------------------------------------------------- navigation */

function show(view) {
  if (!["chat", "retrieval", "eval", "corpus"].includes(view)) view = "chat";
  state.view = view;
  $$("[data-view-panel]").forEach((panel) => { panel.hidden = panel.dataset.viewPanel !== view; });
  $$(".nav-item[data-view]").forEach((link) => link.classList.toggle("active", link.dataset.view === view));
  if (location.hash !== `#${view}`) history.replaceState(null, "", `#${view}`);
}

function bind() {
  $("#ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    ask($("#question").value);
  });
  $("#lab-form").addEventListener("submit", (event) => {
    event.preventDefault();
    renderLab($("#lab-question").value);
  });
  window.addEventListener("hashchange", () => show(location.hash.replace("#", "")));
  $$(".nav-item[data-view]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      show(link.dataset.view);
    });
  });
}

/* -------------------------------------------------------------------- boot */

async function boot() {
  bind();
  try {
    const [documents, labels, baseline] = await Promise.all([
      fetchJson("../corpus/documents.json"),
      fetchJson("../corpus/eval.json"),
      fetchJson("../corpus/baseline.json"),
    ]);
    state.documents = documents;
    state.labels = labels;
    state.baseline = baseline;
    state.retriever = buildRetriever(documents, { size: 120, overlap: 30, strategy: "heading" });

    renderChrome();
    show(location.hash.replace("#", "") || "chat");
    ask($("#question").value);
  } catch (error) {
    console.error(error);
    $("#answer-text").textContent = `Could not load the corpus: ${error.message}. Run the demo through the local server.`;
    $("#answer-meta").textContent = "startup failed";
  }
}

boot();
