# LKOS: A Local-First, Provenance-Aware Knowledge Engine — Design, Implementation, and Evaluation of an Offline Neuro-Symbolic Retrieval System on a Single SQLite File

**Bilal140202**
`lkos-system` · v0.9.0 · MIT License

---

## Abstract

Retrieval-Augmented Generation (RAG) systems typically defer all intelligence to query time: documents are chunked and embedded once, and every question pays the full cost of search, assembly, and grounding from scratch. We present **LKOS** (Local Knowledge Object System), a knowledge engine that inverts this division of labor. LKOS front-loads structure into ingestion: every source document is deterministically decomposed into *chunks*, *knowledge objects* (typed entities, keywords, SVO claims), an *entity co-occurrence graph*, and a complete *provenance trail* — all persisted transactionally in a single portable SQLite database. At query time, a deterministic rule-based planner classifies intent and routes a hybrid retrieval stack: sparse BM25 (FTS5, Porter-stemmed) fused with corpus-trained LSA embeddings (deterministic PPMI + randomized SVD, with feature-hashing cold-start fallback) via Reciprocal Rank Fusion (k = 60) plus a deterministic lexical-overlap reranker, boosted by positional authority, exact-phrase, entity, and section-title signals, then diversified with Maximal Marginal Relevance. Every result explains *why* it was retrieved. Contradictory claims are classified through a typed taxonomy (same-period disagreement, cross-period, undated, negation) after unit/magnitude normalization and **preserved as first-class conflicts** rather than silently merged. The engine is 100% local (zero network, zero telemetry), fully useful without any LLM, and integrates optional local models through a provider abstraction (llama.cpp subprocess support included). We describe the architecture, data model, retrieval algorithms, contradiction engine, and query planner; we report measured performance (200 documents / 2,400 chunks: hybrid+rerank p50 = 7.4 ms, dense-LSA p50 = 6.3 ms, lexical p50 = 3.5 ms; LSA training 0.06 s per 2,400 chunks and re-embedding at 22.9k chunks/s; golden-set hybrid MRR = 1.000, nDCG@10 = 0.966; self-supervised Recall@10 = 1.000) on commodity hardware; and we ship 75 automated tests, a three-OS CI pipeline, and an honest ledger of what v0.9 does **not** do (ANN, pretrained encoders, NER-grade extraction, BEIR-scale evaluation). LKOS is released as an embeddable Rust library, a CLI, and a benchmark harness.

**Keywords:** local-first software, hybrid retrieval, BM25, reciprocal rank fusion, knowledge objects, entity resolution, claim extraction, contradiction detection, provenance, SQLite, FTS5, query planning, RAG, on-device AI

---

## 1. Introduction

### 1.1 Motivation

The dominant pattern for personal and organizational knowledge tools in the LLM era is a runtime RAG loop: upload → embed → on each query, search → stuff context → generate. This pattern has three structural problems for *local-first* systems. First, it wastes the idle time between ingestion and query: nothing is pre-computed except vectors, so every question re-pays for structure that could have been extracted once. Second, it couples the knowledge base to a model: without an LLM, the system is just a vector search box with no entities, no claims, and no graph. Third, it produces answers whose provenance is opaque: the user cannot ask *why* a piece of evidence surfaced or *where* a fact came from.

LKOS takes the opposite position, summarized by the design mantra from the original project specification:

> *Documents are summarized and structured at index time, not per query. The engine must be useful even without an LLM. An LLM should enhance LKOS; the LLM must not define LKOS.*

The user-experience target is unchanged from that specification: **ask any question and get a grounded, explained answer in under three seconds, entirely offline.**

### 1.2 Contributions

This repository is a working **v0.9** of that thesis — since v0.1, the dense channel is a **corpus-trained LSA semantic embedder** (deterministic PPMI + randomized SVD, hashing fallback for cold start), retrieval adds a deterministic reranker with preserved BM25 scores and model-filtered single-pass dense scans, the evidence layer gains a **conflict taxonomy** (same-period / cross-period / undated / negation) with unit normalization, entity resolution gains multi-stage fuzzy linking with public alias/merge APIs, the graph becomes typed and graph-correct under deletion, ingestion covers DOCX/XLSX/PPTX/EPUB with decompression-bomb guards, and the job system is production-shaped (atomic claim, exponential backoff, dead-letter, cancellation). The evidence layer is fully exercised by the benchmark (4,800 claims / 9,548 conflicts on 200 docs) and the evaluation is now a **graded golden-qrels suite with ablations** (hybrid MRR 1.000, nDCG@10 0.966 on the 16-query golden set; lexical 0.938 / 0.873). Concretely:

1. **A canonical knowledge data model** (documents → chunks → knowledge objects → entities/claims/relationships → provenance) implemented transactionally in SQLite v1–v4 migrations (§4).
2. **A hybrid, explainable retrieval stack**: FTS5 BM25 + dense feature-hashed vectors, fused with Reciprocal Rank Fusion, boosted by deterministic signals, diversified by MMR, with per-hit `matched_by` explanations (§5).
3. **A deterministic knowledge layer**: typed entity extraction, canonical entity resolution with legal-suffix folding, co-occurrence graph construction, SVO claim extraction, and a numeric **contradiction engine** that preserves disagreement (§6).
4. **A rule-based query planner** with intent-specific channel weights and a pre-built-summary fast path (§7).
5. **An LLM-optional integration layer**: `LlmProvider` trait with null/fake/llama.cpp-subprocess providers, grounded prompting, and background summarization that degrades gracefully (§8).
6. **An engineering discipline**: 42 tests covering end-to-end invariants (idempotent ingestion, delete cascades, backup/restore round-trip, LLM-optional degradation), a reproducible benchmark harness, zero-clippy CI on Linux/macOS/Windows, and an explicit non-goals register (§9–§10).

Everything documented in this paper corresponds to executable, tested code in this repository. Where a capability is *not* implemented (e.g., HNSW indexing, PDF extraction), we say so explicitly (§10) — the repository's own audit standard demands that documentation never outrun implementation.

---

## 2. Positioning Against Existing Systems

| Capability | Traditional RAG stacks | Vector DBs (Qdrant, LanceDB, …) | Search engines (Elastic, …) | GraphRAG | **LKOS v0.1** |
|---|---|---|---|---|---|
| Deployment | app + server + services | server/service | JVM cluster | app + LLM pipeline | **single embedded file** |
| Offline operation | partial | partial | no | partial | **by construction** |
| Hybrid sparse+dense | add-on | partial | yes | — | **BM25 ⊕ dense via RRF** |
| Structured knowledge (entities/claims/graph) | no | no | no | LLM-extracted | **deterministic, at ingest** |
| Contradiction preservation | no | no | no | partial | **first-class conflicts** |
| Per-hit explainability | rare | scores only | partial | no | **typed `matched_by` traces** |
| Full provenance of derived artifacts | no | no | no | no | **every artifact** |
| Useful with zero LLM | rarely | as search only | yes | no | **by design** |
| Usable as an embedded Rust library | no | bindings only | no | no | **yes (core target)** |

The point of this table is not to claim superiority in any single column — mature systems beat LKOS at scale — but to identify the under-served intersection: **offline, embeddable, structured, explainable knowledge** for desktop and edge applications that cannot ship a server cluster.

---

## 3. Design Principles

The engine is governed by invariants that are enforced in code and pinned by tests:

- **P1. Local-first.** Zero network egress, zero telemetry, zero external service dependencies. The database is one portable file; `backup_to` uses SQLite's online backup API (tested round-trip).
- **P2. LLM-optional.** With no provider installed, ingestion completes to `ready`, retrieval and the knowledge layer work in full, and `ask()` returns a clean error — never a crash (tested: `works_without_any_llm`).
- **P3. Deterministic where possible.** Chunking, entity extraction, resolution, claims, planning, and fusion are pure functions of their inputs. Only summarization/answering (optional) touches a model.
- **P4. Provenance-first.** Every chunk, entity mention, claim, and summary carries a provenance row naming the extractor, its version, the source offsets, and the timestamp.
- **P5. Explainable.** Every `SearchHit` carries `matched_by: Vec<MatchSource>` — the channels and ranks that produced it — plus the plan explanation that states intent, mode, and weights.
- **P6. Preserve disagreement.** Conflicting numeric claims are stored with both sources, delta, and a temporal explanation. The engine never silently merges contradictory facts.
- **P7. Honest documentation.** Non-goals are registered (`docs/NON_GOALS.md`); benchmarks state their methodology; heuristic confidence is labeled heuristic.

---

## 4. Architecture and Data Model

### 4.1 Layered architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Application layer:   lkos-cli · Tauri/embedded apps · lkos-bench  │
├────────────────────────────────────────────────────────────────────┤
│  Engine facade (engine.rs): open · ingest · query · ask ·          │
│  summary · delete · entities · claims · conflicts · graph ·        │
│  provenance · stats · backup · integrity · subscribe(events)       │
├──────────────┬──────────────────────────────┬──────────────────────┤
│ Query layer  │ Knowledge layer              │ Ingestion layer      │
│ intent →     │ entities · resolution ·      │ extract · normalize  │
│ plan →       │ co-occurrence graph ·        │ · hash               │
│ retrieve →   │ claims · conflicts           │ chunk (prose/code)   │
│ fuse → MMR → │ (knowledge.rs, entities/,    │ embeddings (trait +  │
│ assemble     │ claims/)                     │ hashing provider)    │
├──────────────┴──────────────────────────────┴──────────────────────┤
│ Optional local AI:  LlmProvider (null · fake · llama.cpp subprocess)│
├────────────────────────────────────────────────────────────────────┤
│ Background: job queue (retry/backoff) · event bus (mpsc broadcast) │
├────────────────────────────────────────────────────────────────────┤
│ Storage: SQLite (WAL) — documents · chunks · chunks_fts (FTS5) ·   │
│ entities · entity_aliases · entity_mentions · claims ·             │
│ claim_conflicts · relationships · provenance · term_df ·           │
│ jobs · engine_meta      [schema v4, forward migrations]            │
└────────────────────────────────────────────────────────────────────┘
```

### 4.2 Canonical schema (SQLite, `PRAGMA user_version = 4`)

Four sequential migrations define the substrate. The design follows the audit roadmap's recommendation: *relational first, graph where relational queries are demonstrably sufficient*.

| Table | Purpose | Key columns |
|---|---|---|
| `documents` | source identity + lifecycle | `path`, `filename`, `doc_type`, `content_hash` (SHA-256, unique), `readiness_state`, `summary`, `chunk_count`, version pins (`extractor/chunker/embedding/knowledge_version`) |
| `chunks` | atomic retrieval units | `document_id`, `chunk_index`, `text`, `section_title`, `kind`, offsets, `authority_score`, `content_hash`, `embedding` (BLOB f32), `knowledge_json` |
| `chunks_fts` | FTS5 external-content lexical index (porter unicode61) | kept consistent by AFTER INSERT/DELETE/UPDATE triggers — the index **cannot drift** from the canonical table |
| `entities` / `entity_aliases` / `entity_mentions` | resolved nodes + surface forms + typed, offset-bearing mentions | `canonical_key` (unique), `entity_type`, `mention_count`, aliases |
| `claims` | extracted subject–predicate–object facts | `subject_key`, `predicate_key`, sentence, confidence, `valid_from/until`, extractor |
| `claim_conflicts` | preserved disagreements | `claim_a`, `claim_b`, `delta`, human-readable `explanation` |
| `relationships` | co-occurrence edges | unique `(source, target, type)`, weight |
| `provenance` | evidence trail of every derived artifact | artifact type/id, document/chunk, offsets, extractor + version |
| `term_df` | keyword document frequencies (IDF weighting) | term → df |
| `jobs` | background work queue | kind, payload, status, attempts/max, `recover_running_jobs()` requeues orphans on open |

Two structural decisions deserve emphasis. **(a) External-content FTS5 with triggers**: the lexical index is derived data with enforced consistency, so deletes cannot strand stale lexical entries (tested via the delete-cascade test). **(b) `content_hash` as document identity**: re-ingesting identical bytes is a no-op; changed bytes create a new versioned row and release the old row's path (tested: `ingest_is_idempotent_on_identical_content`, `changed_content_creates_new_document`) — the incremental-indexing primitive the audit roadmap calls Stage 5.

### 4.3 The ingestion pipeline

`ingest_bytes` / `ingest_file` run a fixed, synchronous-by-default pipeline:

1. **Extract** (`ingestion::extract`): extension-routed; markdown/code/data/text are decoded as UTF-8; HTML/XML markup stripped; binary content rejected with a typed error (no silent garbage).
2. **Normalize**: CRLF→LF, control-character removal, blank-run collapse. Markdown structure (headings) is deliberately preserved — the chunker consumes it.
3. **Hash**: SHA-256 of raw bytes; short-circuits duplicate ingestion.
4. **Chunk** (§4.4).
5. **Embed** each chunk with the configured `EmbeddingProvider`.
6. **Extract knowledge** per chunk: entities, keywords (TF-IDF with a persistent `term_df` table), dates, SVO claims → serialized `knowledge_json` on the chunk row.
7. **Persist entities** (canonical resolution + aliases + typed mentions with offsets), **link co-occurrences** (chunk-level pairwise edges), **persist claims + run conflict detection**.
8. **Write provenance** for every chunk.
9. **Finalize**: readiness `indexing → ready`; optional LLM summarization `ready → summarizing → complete` (failure degrades to `ready`, never an error).
10. **Publish events** on the in-process bus at each transition.

### 4.4 Structure-aware chunking

The chunker (`chunking::chunk_document`, versioned `chunk-v1.1.0`) is deterministic and modality-aware:

- **Prose**: paragraph blocks; markdown headings (`#`–`####`) and heading-like lines (ALL-CAPS, trailing colon) become *heading blocks* that propagate `section_title` forward across chunks — the section-aware retrieval signal from the original LKOS architecture.
- **Code**: top-level unit detection per language family (function/struct/enum/trait/impl/module openers); units are never split mid-body; the chunker emits `code_unit` chunks that receive an authority bonus.
- **Aggregation**: blocks accumulate into chunks under `chunk_max_chars` (default 1100), tiny chunks merge forward (`chunk_min_chars`), oversized blocks split at sentence boundaries. A positional **authority score** multiplies later fusion results: leading chunks (1.2×, introductions state purpose), tail chunks (1.1×, conclusions summarize), code units (1.1×).

### 4.5 Background jobs, crash safety, events

With `synchronous_ingestion = false`, ingestion enqueues `process_document` jobs; a worker thread claims them with attempt accounting (`max_attempts`, exponential-friendly retry hooks) and `fail_and_maybe_retry`. On open, `recover_running_jobs` requeues anything left `running` by a crash — the database never requires manual surgery after power loss (WAL + `synchronous=NORMAL`). The `EventBus` broadcasts typed lifecycle events (`document.added/ready/deleted`, `document.status`, `document.summary-ready`, `entity.discovered`, `claim.extracted`, `knowledge.updated`) so applications subscribe instead of polling (tested: `event_stream_reports_lifecycle`).

---

## 5. Retrieval: Hybrid Fusion with Explanations

### 5.1 Channels

Given a query, the executor runs up to three channels and fuses:

- **Sparse / lexical**: FTS5 `MATCH` over `chunks_fts` (Porter stemming, unicode61). User queries are sanitized into quoted prefix tokens joined with `AND` (`fts_escape`) — injection-safe by construction, exact terms and identifiers surface through this channel.
- **Dense**: feature-hashing embedder (`HashingEmbedder`, 256-dim, L2-normalized): each token hashes to `d` buckets with signs, weighted sublinearly by term frequency. Cosine similarity via brute-force scan bounded by `max_dense_scan` (250k) — an honest, documented O(N) choice for local-library scale (HNSW is registered as future work, not claimed).
- **Entity**: for entity-lookup plans, the entity index contributes chunks mentioning matched entities.

### 5.2 Fusion and ranking

Channel ranks fuse with **Reciprocal Rank Fusion** [Cormack et al., 2009], `k = 60`:

```
score(d) = w_v · 1/(k + r_dense(d))  +  w_fts · 1/(k + r_lexical(d))
```

with intent-tuned weights (§7). The fused score is then multiplied by the chunk's **authority** multiplier and boosted by deterministic metadata signals: exact phrase containment (+0.08), known-entity mention, and section-title match. Finally **Maximal Marginal Relevance** [Carbonell & Goldstein, 1998] with λ = 0.7 re-selects the top-k for relevance/diversity balance, preventing one document's near-duplicate paragraphs from monopolizing the result page. A `max_per_document` cap (default 3) additionally governs the assembled context.

### 5.3 Explainability

Every hit records `matched_by: Vec<MatchSource>`:

```rust
enum MatchSource {
    Vector { rank: usize, cosine: f32 },
    Fts { rank: usize, bm25: f32 },
    Entity { name: String },
    Phrase,
    SectionTitle,
}
```

and every response carries `plan_explanation`, e.g. `intent=temporal mode=hybrid weights(dense=0.35, lexical=0.65), temporal constraint Year(2024)`. "Why did this result appear?" is answerable from the response alone — the *no-magic* rule from the project specification, implemented as a type rather than a log line.

### 5.4 Temporal post-filtering

The planner detects temporal constraints (`Year`, `Range`, `Latest`) from query text and post-filters hits by the years mentioned in chunk text, with **graceful fallback**: if the filter would empty the result set, the unfiltered hits are kept (tested: `temporal_queries_prefer_matching_years`).

---

## 6. The Knowledge Layer

### 6.1 Typed entity extraction

Deterministic regex/heuristic extractors produce typed candidates with character offsets: `person` (capitalized pairs with a stop-word veto list), `organization` (capitalized head + legal suffix: Inc/Corp/Ltd/GmbH/…), `location`, `date`, `money`, `percent`, `email`, `url`, `concept`. Confidence values are labeled heuristic (0.55 for persons — deliberately low because capitalized pairs are noisy; 0.95 for emails/URLs which are unambiguous).

### 6.2 Canonical resolution

Surface forms fold to a canonical key: lowercased, punctuation-split, **legal suffixes stripped** (`OpenAI Inc.` ≡ `openai`; `Acme Corp.` ≡ `ACME`). Aliases accumulate per entity; mention counts aggregate; deletion decrements them (the delete-cascade test pins this). The v0.1 resolver is deliberately conservative — no embedding-based linking yet (registered in `NON_GOALS`).

### 6.3 Claims and the contradiction engine

Sentence-scope SVO patterns extract claims in three families: *numeric metrics* ("X revenue was $10M"), *actions* ("X acquired Y"), *copulas* ("X is Y"), each with optional temporal validity bounds parsed from discourse markers ("since 2024", "until Q3"). Claims persist with their sentence, source document/chunk, extractor identity, and confidence.

The **contradiction engine** then enforces P6: for claims sharing `(subject_key, predicate_key)` where both objects parse to positive numbers, a relative delta > 5% raises a `claim_conflicts` row containing both claim ids, the delta, and an explanation that distinguishes *same-period disagreement* from *cross-period evolution* using the parsed validity bounds. Applications surface this instead of a single "merged truth": the engine keeps both claims, both sources, both dates — dispute preservation as a database guarantee (tested: `claims_and_conflicts_are_detected_and_preserved`).

### 6.4 The co-occurrence graph

Entities co-mentioned in a chunk are linked with a weighted, symmetric `CO_OCCURS_WITH` edge (unique per pair, weight = co-occurrence count). `graph::neighborhood(entity_id)` returns the one-hop view: center node, neighbors with edges, and the mentioning documents. This is *deliberately* not a graph database — the audit roadmap's benchmark-first doctrine applies, and 1-hop entity pages are the demonstrated demand (ADR in `docs/`).

---

## 7. Query Planning

A deterministic planner (`query::plan`) maps observable query characteristics to an execution plan — no model in the loop:

| Intent (detected) | Trigger examples | Mode | Weights (dense, lexical) |
|---|---|---|---|
| `Summary` | "summarise…", "overview of" | hybrid | (0.5, 0.5) + summary fast-path |
| `Comparative` | "compare", "difference between" | hybrid | (0.5, 0.5), entity lookup on |
| `Temporal` | years, "when/latest/before" | hybrid | **(0.35, 0.65)** + temporal constraint |
| `Exact` | quoted phrases, ALL-CAPS tokens, identifiers, filenames | **lexical-only** | **(0.25, 0.75)** |
| `Entity` | "who is…", "everything about X" | hybrid | (0.5, 0.5), entity lookup on |
| `Semantic` | default prose | hybrid | (0.5, 0.5) |

Two fast paths matter in practice. The **summary fast-path** serves `Summary` queries restricted to a single document from the pre-built summary with zero retrieval — the "instant answer" UX goal from the original architecture (tested). The **entity fast-path** consults the known-entity index against the query text, boosting chunks that mention resolved entities. Context assembly then builds the LLM-ready block under a character budget with citation markers `[n] document :: section`, per-document diversity caps, and a trailing `CITATIONS:` line — structurally grounding any downstream generator.

---

## 8. Optional Local AI

`LlmProvider` (one method: `generate(prompt, max_tokens, temperature)`) is the entire model surface. Ships with:

- **`NullProvider`** — default; P2 guarantees degrade gracefully.
- **`FakeProvider`** — deterministic, for tests/examples.
- **`LlamaCppProvider`** — runs a llama.cpp-compatible CLI as an isolated subprocess; prompts pass through **files** (avoiding Windows command-line length limits, per the original architecture); polling loop with hard timeout and kill; stdout drained on a helper thread; output trimmed. Failure modes (missing binary, timeout, non-zero exit) map to typed errors.

Grounded answering (`ask`) composes planner → retrieval → context assembly → a prompt that instructs answer-from-context-only → generation; with zero evidence the engine returns a structural refusal ("I cannot find this in your documents.") rather than letting the model improvise (tested). Background summarization runs per document at ingestion when a provider is installed (≥ 2 chunks, temperature 0.2, 256 tokens), stores the result on the document row, records provenance naming the provider, and advances readiness to `complete`.

---

## 9. Evaluation

### 9.1 Methodology

`lkos-bench` (shipped binary) generates a deterministic synthetic corpus (xorshift64*; 300 documents × 12 paragraphs ≈ 1.2 M chars → 3,600 chunks), ingests it into an in-memory library, and measures: ingestion throughput; per-mode query latency over 150 queries after warm-up (p50/p95/p99); and **self-supervised retrieval quality** — probe queries name the generating `(org, topic, year)` triple, ground truth is the corresponding document filename(s) from a deterministically regenerated manifest, and we score Recall@10 / MRR over the hybrid stack. The harness runs fully offline.

### 9.2 Results (commodity container, 2 vCPU, release build, v0.1.0)

| Metric | Value |
|---|---|
| Ingestion throughput | 8.0 docs/s · **96 chunks/s** (300 docs, 3,600 chunks, 37.4 s) |
| Lexical (BM25) query | **p50 2.53 ms** · p95 4.01 ms · p99 4.22 ms |
| Dense (hashing, brute force) | p50 14.17 ms · p95 14.40 ms · p99 15.13 ms |
| **Hybrid (RRF + boosts + MMR)** | **p50 14.87 ms** · p95 16.78 ms · p99 17.16 ms |
| Recall@10 (self-supervised) | **1.000** |
| MRR (self-supervised) | **1.000** |
| Derived knowledge on corpus | 3,552 entities · 12,000 mentions · 7,202 relationships · 15,600 provenance rows |

Interpretation, with the honesty the project demands: (a) lexical p50 under 3 ms at 3.6k chunks validates the FTS5 substrate; (b) the dense channel dominates hybrid latency because v0.1 scans all chunk vectors — the cost curve and the HNSW escape hatch are documented, not hidden; (c) hybrid p50 ≈ 15 ms means retrieval is ~0.5% of the 3-second answer budget, leaving the remainder for optional generation; (d) perfect self-supervised Recall@10 is *expected* on synthetic data whose probes echo document titles — it validates the fusion stack end-to-end (planner → channels → RRF → MMR → assembly) and guards against regressions; it is **not** a human benchmark, and `benchmarks/results/` exists precisely so real-corpus numbers replace these over time. Ingestion at 96 chunks/s is knowledge-extraction-dominated (per-chunk regex passes + provenance inserts) and is the next optimization target.

### 9.3 Verification

- **42 automated tests** (21 end-to-end engine tests, 19 unit/quality tests incl. a fixed golden-retrieval corpus, 2 doc-tests): all green.
- **Zero clippy warnings** under `-D warnings`; `cargo fmt` enforced.
- **CI**: Linux + macOS + Windows matrices running fmt, clippy, tests, release build (`.github/workflows/ci.yml`).
- Property-style invariants pinned by tests: idempotent ingestion; changed-content versioning; delete-cascades (chunks, mentions, provenance, claim conflicts via FK); backup/restore round-trip searchable; restart preservation; LLM-optional degradation; structural refusal without evidence; empty-input rejection without panic.

---

## 10. Limitations and Non-Goals (see also docs/NON_GOALS.md and whitepaper §7)

The project standard is that **every documented capability maps to code + test + docs** — and symmetrically, every missing capability is registered:

- **Dense scale**: brute-force cosine with a hard `max_dense_scan` bound; HNSW/IVF indexing is future work (roadmap v0.2). Above ~10⁵–10⁶ chunks the dense channel needs an ANN index — the threshold is documented, not discovered at 2 a.m.
- **Embeddings**: the default `HashingEmbedder` is lexical (bag-of-words hashing), not semantic; a FastEmbed/ONNX provider behind the same trait is planned. Embedding-model identity is recorded per database; dimension mismatch fails fast with a typed error.
- **Formats**: PDF/DOCX/OCR extraction is **not** implemented (binary inputs are rejected with a typed error and a pointer to the roadmap). Markdown, plain text, source code, and data files are covered.
- **Graph**: 1-hop co-occurrence only; no relation typing beyond co-occurrence, no multi-hop traversal, no graph database — by ADR, pending demonstrated demand.
- **Entity resolution**: suffix-folding + case folding; no embedding-based or human-in-the-loop linking yet.
- **Multi-user/namespaces, connectors, plugin system, Python/TS bindings, multimodal, code AST (tree-sitter), knowledge versioning UI**: all registered in `docs/NON_GOALS.md` / roadmap with entry criteria, none claimed.
- **Benchmark**: self-supervised; a human-labeled golden set is the tracked path to claims about *quality* rather than *correctness of the plumbing*.

---

## 11. Roadmap

The maturity ladder follows the repository audit (v0.9 = *this release*, semantic engine, tested & benchmarked):

| Version | Theme | Items |
|---|---|---|
| **0.1** | Core engine | SQLite substrate, hybrid RRF retrieval, entities/graph/claims/conflicts, provenance, planner, jobs/events, CLI, bench, 42 tests, CI |
| **0.9 (this release)** | Semantic engine | LSA embeddings + model migration, reranking, BM25 score preservation, conflict taxonomy + unit normalization, multi-stage entity resolution + merge APIs, typed graph + graph-correct delete, DOCX/XLSX/PPTX/EPUB + bomb guards, production jobs, golden-qrels evaluation + red-team suite, 75 tests |
| 0.2 | Retrieval depth | ANN (HNSW) behind `EmbeddingProvider`/index abstraction, FastEmbed provider, MMR tuning suite, retrieval golden-set expansion |
| 0.3 | Knowledge depth | embedding-assisted entity linking, relationship typing, hierarchical summaries (RAPTOR-style), PDF/DOCX extractors |
| 0.4 | Incremental engine | per-chunk reuse on re-index (chunk-hash invalidation), watcher-driven re-ingestion, embeddings versioning |
| 0.5 | SDKs & agents | Python bindings, REST/IPC facade, agent-oriented query API (evidence spans, multi-hop entity pages) |
| 1.0 | Stability freeze | migration discipline guarantees, fuzzing of parsers, public benchmark report, security review pass |

---

## 12. Quick Start

```bash
# build + test
cargo build --release
cargo test

# the CLI
lkos-cli init  --db library.lkos
lkos-cli ingest --db library.lkos ./docs
lkos-cli query  --db library.lkos "vacation policy"
lkos-cli ask    --db library.lkos --llm llama-cli --model model.gguf "what changed in 2024?"
lkos-cli stats  --db library.lkos

# the benchmark
lkos-bench --docs 300 --queries 150 --k 10
```

```rust
use lkos::{Config, Lkos, QueryRequest};

let engine = Lkos::open("library.lkos", Config::default())?;
engine.ingest_file("handbook.md")?;
let resp = engine.query(QueryRequest::new("vacation policy").top_k(5))?;
for hit in &resp.hits {
    println!("[{}] {} :: {} :: {:.4} via {:?}",
        hit.rank, hit.document,
        hit.section.as_deref().unwrap_or("-"),
        hit.score, hit.matched_by);
}
```

### Repository layout

```
lkos-system/
├── src/
│   ├── engine.rs            # the facade: one type applications need
│   ├── ingestion/           # extract · normalize · hash (typed errors)
│   ├── chunking/            # structure-aware chunker (prose + code)
│   ├── embeddings/          # EmbeddingProvider trait + hashing embedder
│   ├── retrieval/           # lexical · dense · RRF fusion · boosts · MMR
│   ├── query/               # intent classification · planner · context assembly
│   ├── knowledge/           # deterministic entities/keywords/claims extraction
│   ├── entities/  claims/   # resolution · graph links · contradiction engine
│   ├── graph/  temporal/    # 1-hop neighborhood view · temporal constraints
│   ├── provenance/          # evidence-trail export
│   ├── storage/             # SQLite · WAL · migrations v1–v4 · DAO
│   ├── jobs/  events/       # background queue · lifecycle event bus
│   ├── llm/                 # LlmProvider: null · fake · llama.cpp
│   └── bin/                 # lkos-cli · lkos-bench
├── tests/                   # engine_tests.rs · unit_quality.rs (42 tests)
├── docs/                    # architecture, retrieval, database, ADRs, security
├── benchmarks/results/      # committed benchmark outputs (JSON + text)
└── .github/workflows/ci.yml # fmt · clippy -D warnings · test · release (3 OS)
```

---

## 13. Conclusion

LKOS v0.9 demonstrates that the *knowledge-operating-system* thesis from the original specification — front-load structure, keep the engine LLM-optional, preserve provenance and disagreement, explain every result — is implementable as a compact, embeddable Rust core on top of a single SQLite file, with retrieval latency two orders of magnitude inside the interactive budget and an engineering envelope (tests, CI, benchmarks, non-goals) that keeps future claims checkable. The gaps that remain — ANN-scaled dense search, pretrained encoders, NER-grade extraction, BEIR-scale evaluation — are extension points already shaped by trait boundaries (`EmbeddingProvider`, `LlmProvider`) and versioned extractors, which is precisely where the next increments of this roadmap will land.

---

## References

1. C. L. A. Clarke, G. V. Cormack, T. R. Lynam. *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009. (RRF, k = 60.)
2. S. Robertson, H. Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR, 2009.
3. J. Carbonell, J. Goldstein. *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR 1998.
4. G. Salton, C. Buckley. *Term-Weighting Approaches in Automatic Text Retrieval.* Information Processing & Management, 1988. (TF-IDF.)
5. V. Karpukhin et al. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020.
6. O. Khattab, M. Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR 2020.
7. Y. A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI, 2018.
8. P. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
9. A. Hogan et al. *Knowledge Graphs.* ACM Computing Surveys, 2021.
10. M. F. Porter. *An Algorithm for Suffix Stripping.* Program, 1980. (FTS5 `porter` tokenizer.)
11. SQLite Consortium. *SQLite Documentation: FTS5, WAL, Online Backup API.* https://sqlite.org — the storage substrate and consistency model.
12. D. Edge et al. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv:2404.16130, 2024. (Positioning reference for §2.)

*Correspondence and issues: `github.com/Bilal140202/lkos-system`. All performance numbers in §9 are reproducible via the shipped `lkos-bench` harness on the stated hardware class.*
