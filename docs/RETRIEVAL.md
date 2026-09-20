# RETRIEVAL.md — The Hybrid Stack in Detail

## Channels

**Lexical (FTS5).** `chunks_fts` is an external-content FTS5 index over
`chunks.text` with the `porter unicode61` tokenizer. Queries are sanitized by
`retrieval::fts_escape`: user text is tokenized (alphanumeric runs, lowercased)
and each token becomes a quoted prefix term (`"vacation"*`), joined with `AND`.
Operator characters, parentheses, and quotes from user input can therefore never
alter query structure — injection-safe by construction.

**Dense.** `HashingEmbedder` (256-dim, L2-normalized): tokens hash into buckets
with signed updates, term frequency weighted sublinearly. Cosine similarity is
computed brute-force over stored BLOBs, bounded by `Config::max_dense_scan`
(250,000 chunks). The bound fails loudly rather than silently degrading.

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
