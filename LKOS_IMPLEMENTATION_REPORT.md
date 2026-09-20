# LKOS Implementation Report — v0.9.0

## What was built, in deltas over v0.1

| Area | v0.1 | v0.9 |
|---|---|---|
| Dense channel | FNV-1a feature hashing (lexical) | LSA (PPMI + randomized SVD, deterministic) with hashing cold-start fallback; per-chunk model lineage |
| Dense scan | N+1 blob queries, whole-corpus RAM | single-pass model-filtered SQL scan |
| Lexical channel | BM25 computed then discarded | scores preserved through fusion and explanations |
| Entity channel | HashMap-iteration order (nondeterministic) | id-ordered deterministic |
| MMR | prefetch k+16, silent degradation | full window (cap 64) |
| Reranking | none | deterministic lexical-overlap, top-24, evidence-preserving |
| Freshness | hard filter (could drop newest) | Latest re-ranks via boost; Year/Range filter with fallback |
| Entity resolution | exact canonical key | + alias table + type-guarded Jaro-Winkler 0.93; public alias/merge APIs; audit log |
| Claims | 3 SVO regexes, no offsets, no negation | + sentence offsets, negation, unit/magnitude normalization, temporal bounds from "in YYYY" |
| Conflicts | numeric-only, O(N^2) full scan | taxonomy (same-period/cross-period/undated/negation), indexed lookup, bounded window |
| Graph | 1 symmetric edge, 1-hop, stale on delete | typed RELATES_TO_* from claims, multi-hop BFS, centrality, graph-correct deletion |
| Ingestion | text/md/code + naive HTML toggle | + DOCX/XLSX/PPTX/EPUB, real HTML (script/style drop, entity decode), size/decompression/entry caps, optional PDF feature |
| Jobs | SELECT-then-UPDATE race, instant retry, no dead-letter | atomic claim, exponential backoff + run_at, dead-letter, cancellation, progress, worker_threads honored |
| Schema | v4 | v5 (model lineage, claim offsets, conflict kind + indexes, job scheduling, merge audit, downgrade guard) |
| Evaluation | title-echo self-benchmark (Recall 1.000 meaningless), claims 0 | graded golden qrels + ablation harness + honest floors; bench exercises claims/conflicts/LSA |
| Tests | 42 | 75 (incl. adversarial red-team suite) |

## Verified numbers (this machine, release build)

- LSA training: 2400 chunks in 0.06 s; re-embed 22.9k chunks/s
- Query p50: lexical 3.47 ms, dense 6.28 ms, hybrid+rerank 7.43 ms (p99 8.12 ms)
- Evidence layer on 200-doc bench: 4,800 claims, 9,548 conflicts, 2,384 entities, 10,658 edges
- Golden set (16 queries): hybrid MRR 1.000, nDCG@10 0.966, Recall@10 1.000; lexical MRR 0.938
- Tests: 75/75; clippy: 0 warnings

## What was removed / replaced

- N+1 embedding fetch (replaced), last-token number parser (replaced),
  documented-but-absent alias API (implemented), hard temporal filter for
  Latest (re-ranked), `chunk_overlap` dead config (still unused: structural
  chunking rationale stands; field kept for compatibility, documented).

## Remaining honest gaps

See whitepaper section 7 and NON_GOALS.md: ANN, pretrained encoders,
NER-grade extraction, BEIR-scale evaluation, OCR, bitemporal query planning.
