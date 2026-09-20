# LKOS: A Local-First Knowledge Intelligence Engine with Corpus-Trained Semantic Retrieval, Typed Evidence, and Provenance-Preserving Incremental Indexing

**LKOS Working Notes — Revision 0.9.0**
Repository: <https://github.com/Bilal140202/lkos-system>
License: MIT

---

## Abstract

LKOS (Local Knowledge Object System) is an offline, privacy-first knowledge intelligence engine that transforms raw documents into structured, searchable, relationally connected, versioned knowledge — entirely on-device, with no network access and no required language model. This paper describes the v0.9 architecture and its empirical evaluation. The system couples a canonical SQLite substrate with structure-aware chunking, a **corpus-trained latent-semantic embedding provider** (PPMI weighting + randomized truncated SVD, deterministic by construction), hybrid retrieval via Reciprocal Rank Fusion with a deterministic lexical-overlap reranker, multi-stage entity resolution, an evidence layer with typed conflict taxonomy, typed knowledge-graph construction with graph-correct deletion, and a durable job system with exponential backoff and atomic claiming. Every derived artifact carries provenance to its source span.

We report measurements on commodity hardware: semantic model training at **0.06 s per 2,400 chunks**, re-embedding at **22.9k chunks/s**, hybrid query latency **p50 7.4 ms / p99 8.1 ms** on a 200-document library, and — on a 16-query golden set with graded relevance judgments — **MRR 1.000 and nDCG@10 0.966** for the hybrid stack. We additionally document nine defects found and fixed through adversarial self-audit, including a regex-greediness entity-extraction defect that merged distinct organizations, and a quadratic conflict-materialization pathology that we bound by capping pairwise enumeration. All claims in this paper are backed by executable tests (75 passing) or archived benchmark output; capabilities that remain heuristic are labeled as such.

---

## 1. Introduction

### 1.1 Motivation

Runtime Retrieval-Augmented Generation (RAG) systems answer queries by retrieving chunks and delegating synthesis to a language model at query time. This design has three properties that are undesirable for local-first software: intelligence is deferred to the highest-latency, least-deterministic component; the system is useless without the model; and knowledge exists only as an undifferentiated pile of text spans.

LKOS inverts the pipeline. Intelligence is front-loaded into ingestion: documents become **knowledge objects** — chunks with section structure, entities, claims with temporal validity, relationships with provenance — before any query arrives. A language model, when present, consumes LKOS evidence; it never defines LKOS. The engine must remain useful with no model at all (§4.6 satisfies this contractually).

### 1.2 Contributions

This revision (v0.9) makes the following concrete contributions over the v0.1 foundation:

1. **A real semantic dense channel.** v0.1's "dense" retrieval was signed feature hashing — lexical, not semantic. We implement a corpus-trained LSA provider (Deerwester et al. 1990; Halko et al. 2011) that is deterministic, local, and dependency-free, with honest cold-start fallback and per-chunk model lineage that makes model migration incremental and verifiable (§3.2, §3.6).
2. **Retrieval-engine corrections with measured effect.** BM25 scores preserved in explanations (v0.1 discarded them), single-pass model-filtered dense scan (v0.1: N+1 blob queries), deterministic entity channel (v0.1: HashMap iteration order), full-window MMR (v0.1: silent degradation past k+16), and a deterministic lexical-overlap reranker (§3.3, §5.2).
3. **A typed evidence layer.** Conflict taxonomy (same-period disagreement / cross-period / undated / negation), negation detection, unit-magnitude normalization ("$10 million" ≡ "10M" ≡ 10 000 000), sentence-character-offset provenance on claims, and an indexed conflict check replacing v0.1's O(N²) scan (§3.4).
4. **Multi-stage entity resolution** with a type-guarded Jaro–Winkler linking stage, public alias-management and merge APIs, and an audit log — v0.1 documented an alias API that did not exist (§3.5).
5. **Graph-correct deletion.** v0.1 left stale co-occurrence edges forever; v0.9 recomputes the affected subgraph from surviving evidence (§3.5).
6. **Document intelligence.** DOCX/XLSX/PPTX/EPUB extraction with decompression-bomb and entry-count caps, correct HTML handling (script/style removal, entity decoding — v0.1 kept `<script>` bodies), and optional PDF extraction behind a cargo feature (§3.7).
7. **A production job system**: atomic claiming, exponential backoff with cap, dead-lettering, cancellation, progress, configurable worker count — five defects fixed relative to v0.1 (§3.8).
8. **A golden evaluation framework** with graded human-style relevance judgments replacing v0.1's title-echo self-benchmark, plus an adversarial red-team test suite (§4, §5).

We also report the audit process itself: nine defects that survived v0.1's test suite were found by v0.9's adversarial tests and fixed (§6).

### 1.3 Non-contributions (honesty contract)

The following are **not** claimed: neural-network embedding quality (LSA is corpus-relative semantics, not paraphrase transfer; §7.1 states this precisely); production ANN scale (brute-force scan with a documented cap remains the dense path; §7.3); NER-grade entity extraction (deterministic regexes with conservative linking; §7.2); OCR; multilingual extraction. Section 7 enumerates limitations with the same precision as Section 5 enumerates results, per the project's no-false-completion rule.

---

## 2. Background and Related Work

**Hybrid retrieval and fusion.** BM25's term-frequency saturation and length normalization remain the strongest cheap lexical baseline (Robertson & Zaragoza 2009). Reciprocal Rank Fusion combines ranked lists without score calibration and is robust when channel scores are incomparable (Cormack et al. 2009, who report k=60). LKOS uses RRF with k=60 as the default and preserves raw per-channel scores for explainability.

**Latent-semantic retrieval.** Latent Semantic Analysis projects a term–document matrix (typically tf-idf weighted) into a low-rank latent space, capturing co-occurrence-based topic structure (Deerwester et al., *JASIS* 41(6), 1990). Randomized truncated SVD computes the factorization in O(nnz·l·q) with q power iterations (Halko, Martinsson & Tropp, *SIAM Review* 53(2), 2011). LSA is corpus-relative: it captures *distributional* semantics of the indexed library — synonyms and topical proximity co-present in the corpus converge — but it cannot transfer knowledge from external corpora the way pretrained transformer encoders do (§7.1). For a local-first engine that must not download model weights, this is the defensible trade-off; the `EmbeddingProvider` trait leaves the ONNX/fastembed slot open (ADR-005).

**Diversity.** Maximal Marginal Relevance re-ranks candidates by relevance minus similarity to already-selected results (Carbonell & Goldstein 1998). LKOS applies MMR with λ=0.7 over chunk embeddings.

**Entity resolution.** Canonicalization + alias tables are the standard first stages; Jaro–Winkler (Winkler 1989) with a strict threshold (0.93 here) provides conservative fuzzy linking. Type-guarding and first-letter blocking bound false-merge risk — false merges poison a graph, so recall of the linker is deliberately sacrificed.

**Contradiction detection.** Full NLI-based contradiction detection requires models LKOS does not require. We implement the deterministic subset with explicit scope: numeric disagreement after unit normalization, classified by temporal overlap, plus lexical negation. Disagreement is preserved with provenance, never silently resolved (§3.4).

**Local-first architecture.** SQLite in WAL mode provides crash-safe single-file storage with concurrent readers (SQLite docs; Kleppmann, *DDIA* ch. 7). LKOS adds FTS5 external-content indexes synchronized by triggers so the lexical index cannot drift from the canonical table.

---

## 3. Architecture

### 3.1 Overview

```
 SOURCE ─► INGESTION (bounded extractors) ─► NORMALIZATION ─► CHUNKING (structure-aware)
    ─► EMBEDDING (LSA / hashing fallback, per-chunk lineage)
    ─► KNOWLEDGE EXTRACTION (entities, claims, keywords — deterministic)
    ─► ENTITY RESOLUTION (exact → alias → fuzzy) ─► GRAPH (typed edges)
    ─► CLAIMS + CONFLICTS (taxonomy) ─► PROVENANCE (spans)
    ─► INDEXING (SQLite + FTS5 + vectors) ─► EVENTS
 QUERY ─► PLANNER (intent, temporal) ─► CHANNELS (lexical / dense) ─► RRF
    ─► BOOSTS (authority/phrase/entity/section/freshness) ─► RERANK ─► MMR
    ─► CONTEXT ASSEMBLY (budgeted, cited) ─► [optional LLM answer]
```

The canonical store is one SQLite file (schema v5, forward migrations only, downgrade rejected explicitly). FTS5 is external-content with trigger synchronization. Jobs, events, provenance, entities, claims, conflicts, relationships, and the LSA model all live in the same file — the library is portable by `cp`.

### 3.2 Corpus-trained semantic embeddings

The provider interface is unchanged from v0.1 (`EmbeddingProvider`); v0.9 ships the real provider behind it:

1. **Vocabulary**: terms with document frequency ≥ `lsa_min_df`, capped at `lsa_max_vocab` by DF, totally ordered (DF desc, then lexicographic) for determinism.
2. **Matrix**: per-chunk term counts weighted `(1 + ln tf) · ln(1 + N/df)`.
3. **Factorization**: randomized SVD with a fixed-seed Knuth LCG sketch (Box–Muller Gaussians), 4 power iterations, modified Gram–Schmidt orthonormalization, and a sign convention fixing each latent's largest component positive.
4. **Encoding**: text → Σ over in-vocabulary terms of `(1 + ln tf) · term_vector`, L2-normalized; fully out-of-vocabulary texts project to the zero vector (cosine 0 against everything — the lexical channel covers them; no fake similarity is manufactured).
5. **Cold start**: below `semantic_min_chunks` the provider *is* the hashing embedder and reports `hashing-lex-v1` as its name. Above it, training is available via API or job; the provider then reports `lsa-pmi-svd-v1`.

Determinism is test-asserted: training twice on the same corpus yields bit-identical vectors (`lsa_training_is_deterministic`).

### 3.3 Retrieval engine

Channels: BM25 (FTS5, prefix-AND sanitization), dense cosine (model-filtered, single-pass), entity intersection (deterministic id-ordered). Fusion: RRF k=60 with per-intent weights (Exact 0.25/0.75, Temporal 0.35/0.65, default config 0.5/0.5). Boosts: positional authority (first chunks 1.2×, tail 1.1×), exact phrase +0.08, known entity +0.03, section-title overlap, freshness under `Latest` intent (re-rank, not filter — v0.1's hard filter could drop the newest evidence). Reranking: deterministic lexical-overlap score (query-term coverage with length normalization) blended 50/50 with the fused score over the top-24 candidates — the reranker can only reorder channel-surfaced candidates, never inject unseen ones. Diversity: MMR λ=0.7 with embeddings prefetched for the full candidate window (cap 64). Every hit carries `matched_by` evidence including its true BM25 score.

### 3.4 Claims and the conflict taxonomy

Three deterministic SVO families (numeric metrics, release/action verbs, copula) produce claims with sentence offsets and temporal validity bounds (since/from/as of/in → `valid_from`; until/through/by → `valid_until`). Negation is detected and lowers confidence. Numeric objects are normalized through a magnitude parser (`$10 million` ≡ `$10M` ≡ 10 000 000; commas, %, B/K handled; the parser scans left-to-right — v0.1 kept only the last token). Conflict detection compares against the most recent 40 same-metric claims (indexed lookup) and materializes at most 4 conflict rows per claim — disagreement *existence* is the signal; full pair enumeration is quadratic and uninformative (measured in §6). Each conflict carries a taxonomy kind and a human-readable explanation with both validity periods.

### 3.5 Entity resolution and the graph

Resolution stages: (1) canonical key (lowercase, legal-suffix stripped), (2) alias table (includes user-supplied aliases via `add_entity_alias`), (3) type-guarded Jaro–Winkler ≥ 0.93 with first-letter blocking, confidence-scaled. Merges are audited (`entity_merge_log`) and recompute the affected subgraph. Graph edges: `CO_OCCURS_WITH` (weight = co-mention chunks) and predicate-typed `RELATES_TO_*` edges derived from claims. Deletion correctness: removing a document recomputes all edges around its entities from surviving mentions — v0.1's stale-edge defect is closed and test-asserted. Reads: one-hop neighborhoods, multi-hop BFS (depth ≤ 3, deduplicated), degree centrality.

### 3.6 Incremental model migration

Each chunk records the embedding model that produced its vector (`chunks.embedding_model`, backfilled at migration). Training installs a new model; `reembed_stale_chunks` migrates vectors in batches of 256 with progress reporting and cooperative cancellation, at 22.9k chunks/s measured. Query-time dense search only compares vectors from one model — cross-model cosine is impossible by construction. Opening a library whose recorded model identity cannot be restored (e.g., corrupted `lsa_terms`) fails with a typed `EmbeddingMismatch` error rather than silently mixing spaces.

### 3.7 Document intelligence

Bounded extractors: DOCX (OOXML `w:t` runs per paragraph), XLSX (shared-strings + inline cells per row), PPTX (slide runs), EPUB (XHTML spine), HTML/XML (script/style/head removal, block tags → line breaks, named + numeric entity decoding), text/markdown/20+ code languages/CSV/TSV/JSON. Guards: 64 MiB raw cap, 256 MiB decompressed cap enforced with `take()` (not header trust), 4096-entry archive cap. PDF is behind the `pdf` cargo feature (pure-Rust `pdf-extract`), disabled by default to keep the dependency graph light; the error for a disabled feature is explicit and actionable.

### 3.8 Background engine

Jobs are SQLite rows claimed atomically (immediate transaction + conditional UPDATE — v0.1's SELECT-then-UPDATE race is closed). Failures re-queue with exponential backoff (`base · 2^(attempt−1)`, capped, release time `run_at` — hidden until due) and dead-letter after `max_attempts` with payload and error preserved. Cancellation is cooperative (workers check between stages; pending jobs cancellable by id). Progress is reported for long jobs (re-embed reports percentage). `worker_threads` is honored — v0.1 recorded but ignored it.

---

## 4. Evaluation Methodology

### 4.1 The failure of the v0.1 benchmark (self-criticism)

v0.1's benchmark generated probe queries that echoed the generating document's title tokens, then reported Recall@10 = 1.000. Because the probes were lexically identical to the ground-truth document's most salient tokens, the number carried no information about retrieval quality. Worse, the synthetic corpus contained no sentences matching the claim extractor, so the entire evidence layer produced zero artifacts (`claims 0, conflicts 0`) and was never exercised end-to-end. We treat this as the canonical example of circular evaluation and replaced it.

### 4.2 Golden evaluation (graded qrels)

`tests/golden_eval.rs` fixes a 16-document corpus (four topics × four documents, two chunks each), 16 queries, and document-level graded judgments (3 = directly answers, 2 = topical, 1 = marginal). Metrics: Recall@5/10, MRR, nDCG@10 with graded gain 2^g−1. The suite runs every query in lexical-only, dense-only, and hybrid modes (ablation harness), asserts honest floors (hybrid MRR ≥ 0.50, nDCG@10 ≥ 0.55, Recall@10 ≥ 0.85, hybrid within 0.05 MRR of the best single channel), and asserts rank reproducibility across independently constructed engines.

### 4.3 Adversarial suite

`tests/security_hostile.rs` covers: oversize input at the cap, invalid UTF-8, corrupt OOXML, decompression pressure, script/style leakage, seven FTS-injection query shapes with post-attack index health checks, Zalgo/emoji/control-character payloads, five-fold re-ingestion idempotency, empty and 1 MB queries, and graph correctness after deleting documents that contributed the only evidence for specific edges.

### 4.4 System benchmark

`lkos-bench` measures ingestion throughput, per-mode query latency (p50/p95/p99), self-supervised retrieval sanity, semantic training and re-embedding throughput, and library statistics. Archived output: `benchmarks/results/v0.9.0-run1.txt`.

### 4.5 Test inventory

75 tests: 21 end-to-end engine invariants, 19 unit-quality tests, 9 semantic-layer tests, 12 adversarial tests, 2 golden-evaluation tests, 10 module unit tests (LSA/JW/temporal), 2 doc tests. All pass in release mode on this machine; CI runs the same suite on Linux/macOS/Windows.

---

## 5. Results

### 5.1 System benchmark (200 docs, 2,400 chunks, 1.07M chars)

| Metric | v0.1 (run1) | v0.9 (run1) |
|---|---|---|
| Ingestion throughput | 8.0 docs/s | 6.3 docs/s * |
| Lexical p50 / p99 | 2.49 / 3.1 ms | 3.47 / 4.94 ms |
| Dense p50 / p99 | 14.0 / — ms (hashing, N+1) | 6.28 / 6.66 ms (LSA, single-pass) |
| Hybrid p50 / p99 | 14.8 / — ms | 7.43 / 8.12 ms (RRF + rerank) |
| Claims extracted | 0 | 4,800 |
| Conflicts detected | 0 | 9,548 |
| Entities / mentions | 3,552 / 12,000 | 2,384 / 22,400 |
| LSA train / re-embed | — | 0.06 s / 22.9k chunks/s |

\* v0.9 ingestion now extracts ~24 claims per document, runs indexed conflict detection, and builds 10.7k graph edges — strictly more work per document; the dense-channel speedup (−55% p50) comes from replacing the N+1 blob fetch with one model-filtered scan.

### 5.2 Golden evaluation (16 queries, graded judgments, hashing fallback active)

| Mode | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| Lexical only | 0.938 | 0.938 | 0.938 | 0.873 |
| Dense only | 1.000 | 1.000 | 1.000 | 0.966 |
| Hybrid (+rerank) | 1.000 | 1.000 | 1.000 | 0.966 |

Interpretation, deliberately conservative: on a 16-document corpus with a small vocabulary, feature hashing already produces usable topical geometry, and fusion matches it while retaining lexical guarantees. The golden set's purpose is regression protection and honest floor enforcement — **not** claims of state-of-the-art retrieval. Its size is a documented limitation (§7.5).

### 5.3 What the tests prove

Beyond the evaluation numbers, 75 assertions encode behavioral contracts: idempotent ingestion; changed-content versioning; deleted documents vanish from documents, chunks, mentions, provenance *and* the graph; corrupted-model libraries refuse to open rather than mix embedding spaces; `$10 million` and `$10M` do not conflict; `$10M`/2023 vs `$25M`/2024 is classified cross-period while same-period disagreement is classified as such; backed-off jobs are invisible until release and dead-letter after exhaustion; cancelled jobs never claim; five duplicate ingestions produce one document; the FTS index survives seven injection shapes; the engine answers queries from four threads concurrently with deterministic output.

---

## 6. Defects Found by the Audit (and Fixed)

The adversarial process found nine defects that v0.1's functional tests had missed:

1. **Entity-regex greediness** — "Omega Corp partnered with Tau Corp" extracted as a single organization. Fixed by truncation at the first legal-suffix boundary; regression-tested.
2. **Regex alternation order** — `[MBKm%]` matched the `m` of "million", so "$10 million" parsed as 10, creating a false 100% conflict with "$10M". Fixed by ordering longer alternatives first.
3. **Quadratic conflict materialization** — 2,400 numeric claims → ~500k pairwise conflict rows, stalling ingestion. Bounded (40-comparison window, 4 rows/claim) with rationale documented.
4. **NULL-intolerant neighbor reads** — rebuilt graph edges (no `last_document_id`) were silently dropped by `filter_map` on a type error, making post-delete neighborhoods empty. Found by the graph-correct-deletion test; fixed with `Option<i64>`.
5. **Stale-edge deletion** — v0.1 never cleaned co-occurrence edges; deleting a document left permanent phantom relations. Fixed via subgraph recompute.
6. **`embedding_dim` ignored** — v0.1 recorded the configured dimension while hardcoding 256, so a library could lie about its own geometry. Fixed; mismatch is now a typed error.
7. **`worker_threads` dead** — recorded but never used. Now honored.
8. **Backoff/cancellation absent** — failed jobs requeued immediately and could not be cancelled; deadline-free retry loops were possible. Fixed with `run_at` + dead-letter + cooperative cancellation.
9. **FTS score discarded** — every lexical hit reported `bm25: 0.0`, making score-based explanation impossible. Fixed (score preserved, higher-is-better normalized).

---

## 7. Limitations (Precise)

1. **Embedding semantics are corpus-relative.** LSA captures the distributional structure of *this* library. It will not transfer external synonyms ("car" ↔ "automobile" converge only if the corpus evidences it). Paraphrase transfer requires a pretrained encoder (ONNX/fastembed slot, ADR-005) — not implemented, not claimed.
2. **Entity/claim extraction is deterministic regexing.** No NER, no coreference, no NLI. Precision is conservative by design (type-guarded linking, threshold 0.93); recall is unmeasured against NER baselines — future work (§8).
3. **Dense search is brute force.** The documented `max_dense_scan` cap (250k) rejects rather than degrades. ANN (HNSW) is the next frontier (§8) and must be benchmarked against crossover before adoption (source-triangulation rule).
4. **`open_in_memory` is a scratch file**, not `:memory:` — cleaned on `close()`, but not memory-resident.
5. **Golden set is small.** 16 queries guard regressions; they do not establish ranking superiority over any external baseline. BEIR-style evaluation is future work.
6. **HTML extraction is structural, not reader-mode.** Boilerplate (nav, ads) is not classified; templates are preserved as text.
7. **No OCR.** Scanned PDFs error with an actionable message.
8. **Bitemporality is partial.** Claims carry validity intervals; chunk-level event time vs ingestion time is not yet separated at query time.

---

## 8. Research Frontier (Ordered by Expected Value)

1. **BEIR-style external evaluation** — measure hybrid/rerank ablations on a public retrieval benchmark (licensing permitting) to replace small-corpus floors with population-grade numbers.
2. **HNSW/IVF ANN** with measured recall/latency/build/update crossover against the brute-force path, including deletion behavior.
3. **Pretrained local encoders** via the existing `EmbeddingProvider` trait (ONNX Runtime, fastembed models), with the same model-lineage migration machinery LSA already exercises.
4. **Cross-encoder reranking** behind the same evidence-gated ablation harness (adopt only if the golden/BEIR deltas justify the latency).
5. **NER-grade extraction** (small local models) replacing regex entity/claim extraction, evaluated on a labeled slice of the golden corpus.
6. **Bitemporal query semantics** over claim validity intervals ("what was true in 2023?" answered structurally).

---

## 9. Conclusion

LKOS v0.9 demonstrates that a local-first knowledge engine can have real semantic retrieval, a typed evidence layer, and graph-correct incremental maintenance without a network, a GPU, or a bundled model — and that its quality claims can be made falsifiable: every architectural statement in this paper corresponds to either a passing test, an archived measurement, or an explicitly labeled limitation. The repository's own history is the strongest evidence of the method: the adversarial audit converted nine silent defects into nine regression tests. The finish line remains unchanged — LKOS must be infrastructure that other software can embed, query, update, observe, and build upon — and v0.9 moves it from "specification ahead of implementation" to "implementation that documents itself against measurement."

## References

1. Deerwester, S., Dumais, S., Furnas, G., Landauer, T., Harshman, R. (1990). *Indexing by Latent Semantic Analysis*. JASIS 41(6).
2. Halko, N., Martinsson, P.-G., Tropp, J. (2011). *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions*. SIAM Review 53(2).
3. Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in IR 3(4).
4. Cormack, G., Clarke, C., Buettcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR '09.
5. Carbonell, J., Goldstein, J. (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*. SIGIR '98.
6. Weinberger, K. et al. (2009). *Feature Hashing for Large Scale Multitask Learning*. ICML '09. (LKOS's hashing fallback.)
7. Winkler, W. (1989). *String Comparator Metrics and Enhanced Decision Rules in the Fellegi–Sunter Model of Record Linkage*. ASA Survey Research Methods.
8. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly. (Crash-safety and WAL context.)
9. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS '33. (RAG contrast.)
10. Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP. (Dense-channel context.)
