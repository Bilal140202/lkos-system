# Architecture Decision Records (v0.1)

Condensed ADRs; each records decision, alternatives, and the evidence that
decided it. New ADRs append; superseded ones stay for history.

## ADR-001 — SQLite is the canonical store

**Decision.** All artifacts (documents, chunks, FTS, entities, claims, graph,
provenance, jobs, metadata) live in one SQLite file; WAL; foreign keys on.
**Alternatives.** RocksDB/LMDB (KV — loses relational queries & FTS5), DuckDB
(OLAP-shaped), Postgres+pgvector (server; violates local-first), LanceDB
(vector-first; weak relational/graph story).
**Evidence.** FTS5 p50 2.5 ms @ 3.6k chunks; online backup API gives portable
snapshots; triggers give an un-driftable lexical index; the whole engine ships
as a single embeddable file. **Consequences.** Write throughput bound by a
single-writer WAL; scale ceilings documented (max_dense_scan; §10 of README).

## ADR-005 — Hashing embedder as the default provider

**Decision.** Feature-hashing, 256-dim, L2-normalized, TF-weighted; provider
trait from day one. **Alternatives.** FastEmbed/ONNX (planned, heavier dep),
cloud APIs (forbidden: P1). **Evidence.** Deterministic, dependency-free,
case-insensitive, separates topics in unit tests; recall@10 = 1.0 on the
self-supervised suite. Honest cost: lexical, not semantic — documented
everywhere, provider boundary is the escape hatch. **Consequences.** Embedding
identity + dimension recorded per DB; mismatch fails with a typed error.

## ADR-006 — Deterministic knowledge extraction before any LLM

**Decision.** Entities/keywords/claims come from versioned regex/heuristic
extractors (`*-v1`), stored with provenance and labeled heuristic confidence.
**Alternatives.** LLM extraction at ingest (slow, non-reproducible, costs),
LLM at query (re-pays per question). **Evidence.** Extraction adds ~10 ms/chunk
deterministically; identical input → identical knowledge (property test);
conflicts need stable subject/predicate keys to be detectable across documents.
**Consequences.** Recall of extractors is modest by construction; the extractor
version chain (`extractor → chunker → knowledge_version`) makes reprocessing
targetable in v0.4.

## ADR-007 — The graph is relational, not a graph database

**Decision.** Co-occurrence edges in a `relationships` table (unique per pair,
weight = count); one-hop neighborhood view. **Alternatives.** Embedded graph
DB (new engine + query language for 1-hop queries), RDF (mismatched to local
files). **Evidence.** The demonstrated demand is "everything about X" /
entity pages — a 2-query SQL join; measured 7,202 edges @ 3.6k chunks with
microsecond reads. **Consequences.** Multi-hop reasoning is out until a
workload demands it (NON_GOALS entry criteria).

## ADR-008 — Rule-based query planner (no LLM in the query path)

**Decision.** Intent classification + channel weights + fast-paths from
observable query features (quotes, ALL-CAPS tokens, years, intent keywords).
**Alternatives.** LLM query planning (latency, non-determinism, defeats the
offline contract), fixed weights (leaves exact/lexical queries mis-served).
**Evidence.** Planner runs in µs; exact intents route lexical-only with
(0.25, 0.75) weights — quoted identifiers and ALL-CAPS tokens are noise for
dense hashing vectors; temporal post-filtering has a graceful fallback so a
wrong guess never empties results. **Consequences.** New intents are new rules +
tests; an LLM planner would need an ablation win on a held-out set to replace
this (NON_GOALS).

## ADR-009 — Contradictions are preserved, not resolved

**Decision.** Numeric claims sharing `(subject_key, predicate_key)` with >5%
relative delta raise a conflict row referencing both claims, with a temporal
explanation when periods differ. **Alternatives.** Last-writer-wins (destroys
evidence), averaging (fabricates facts). **Evidence.** The audit's contradiction
requirement; the engine cannot know which document is right — that is the
application's/user's call. **Consequences.** Conflict storage grows with
disagreement; surfacing UI is the app's job; resolution APIs land with
human-in-the-loop in a later version.
