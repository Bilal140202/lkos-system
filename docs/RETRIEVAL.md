# RETRIEVAL.md — The Hybrid Stack in Detail

## Channels

**Lexical (FTS5).** `chunks_fts` is an external-content FTS5 index over
`chunks.text` with the `porter unicode61` tokenizer. Queries are sanitized by
`retrieval::fts_escape`: user text is tokenized (alphanumeric runs, lowercased)
and each token becomes a quoted prefix term (`"vacation"*`), joined with `AND`.
Operator characters, parentheses, and quotes from user input can therefore never
alter query structure — injection-safe by construction.

**Dense.** The dense channel embeds the query with the active provider
(`HashingEmbedder` 256-dim, or corpus-trained LSA) and compares cosine
similarity against stored BLOBs. Two execution paths (v0.10, ADR-010):

- **Brute force** (exact): single model-filtered scan; used below
  `Config::ann_min_chunks` (20,000 chunks), under any active filter, or when
  `ann_mode = "brute"`. The `max_dense_scan` cap (250,000) still fails loudly
  in `brute` mode.
- **ANN** (approximate): a deterministic pure-Rust HNSW graph
  (`lkos::ann::HnswIndex`, M=16, efC=200, `ann_ef_search`=64) built once per
  embedding model and cached on the engine. The cache fingerprint
  `(model, COUNT, MAX(chunk id))` is *exact* because chunk vectors are
  immutable within a model name (written at insert or at model-change
  migration only) — inserts/deletes/migrations always move one counter.
  Invalidation triggers a full rebuild on the next dense query (measured
  ≈ 1 build; see `benchmarks/results/ann-crossover-v0.10.0.txt`).
  Determinism: no RNG anywhere — layer assignment is a splitmix64 function
  of the chunk id, insertion order is a hash shuffle, every tie-breaks by
  (distance, id). `auto` mode (default) uses ANN at/above the crossover and
  **degrades to ANN past `max_dense_scan` instead of refusing**; filtered
  queries always brute-force the filtered set. Measured at 100k chunks
  (128-d, clustered): brute p50 20.24 ms vs ANN p50 0.30 ms @ recall@10
  0.853 (ef=64) / 0.42 ms @ 0.967 (ef=128); the default ef=64 holds
  recall@10 ≥ 0.95 through 50k chunks.

**Entity.** Entity-lookup plans scan `entity_mentions` for matched entities and
inject chunks with rank-based RRF contributions (weight 0.8).

## Fusion

Reciprocal Rank Fusion with `Config::rrf_k = 60`:

```
score(d) = w_vector / (60 + r_dense(d)) + w_fts / (60 + r_lexical(d))
```

Weights come from the planner (see below). Fusion is computed on candidate
rank lists truncated to `k` per channel.

## Boosts (deterministic, ordered)

1. `score *= authority_score` (chunk-level: 1.2 leading, 1.1 tail / code-unit, else 1.0)
2. `+0.08` exact phrase containment (lowercased substring)
3. entity-match boost (chunk mentions a known entity present in the query)
4. section-title match boost

Each applied boost appends a `MatchSource` variant to the hit's `matched_by`.

## Diversity

`mmr_select` with `Config::mmr_lambda = 0.7` re-selects the top-k from the
scored candidate pool using MMR over the same embedding space; when the pool is
close to k or λ ≥ 0.999 it short-circuits (pure relevance). Context assembly
then enforces `max_per_document` (default 3) and a character budget with
`[n] doc :: section` citation markers and a trailing `CITATIONS:` line.

## Planner weights by intent

| Intent | dense | lexical | extra |
|---|---|---|---|
| Exact | 0.25 | 0.75 | lexical-only mode |
| Temporal | 0.35 | 0.65 | temporal post-filter with graceful fallback |
| everything else | config (0.5/0.5) | config | summary fast-path / entity lookup where applicable |

## Temporal constraints

`temporal::detect_temporal` extracts `Year(y)`, `Range(lo,hi)`, `Latest`, or
`None` from the query. Post-filtering keeps chunks whose text mentions
conforming years; **if the filter would empty the set, the unfiltered results
are returned** (graceful fallback, pinned by test).
