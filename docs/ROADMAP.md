# ROADMAP.md

Maturity ladder (v0.1 = this release). Each version ships only when its
"done" test — code + test + docs + measured benchmark — is met.

## v0.2 — Retrieval depth
- ANN index (HNSW) behind an index abstraction; benchmark vs brute force at
  10k/100k chunks (recall ≥ 0.99 target)
- FastEmbed/ONNX `EmbeddingProvider` (semantic vectors), provider-selection in
  `Config`, controlled re-embedding on provider switch
- Golden-set expansion; MMR/λ ablation table committed to `benchmarks/`

## v0.3 — Knowledge depth
- Embedding-assisted entity linking + human-in-the-loop correction API
- Typed relationships beyond co-occurrence (deterministic pattern set first)
- PDF/DOCX extractors through the same extract→normalize contract (fuzzed)
- Optional at-rest encryption guidance (SQLCipher) for the DB file

## v0.4 — Incremental engine
- Chunk-hash reuse: re-index touches only changed chunks (the stored
  `content_hash` + version chain already support this)
- File-watcher driven re-ingestion with debouncing; embeddings version pinning
- Batched inserts (target: >200 chunks/s ingestion)

## v0.5 — SDKs & agents
- Python bindings (PyO3) over the facade; JSON query API
- Agent-oriented surface: evidence spans, entity pages, multi-hop entity walks
  (first feature to justify a graph upgrade if the ablation proves it)

## v1.0 — Stability freeze
- Migration discipline guarantee (all past versions openable)
- Parser fuzzing suite; security review pass; public benchmark report on a
  real (non-synthetic) corpus with human-labeled probes

---

## v0.9 delivered (this release)

- [x] Real semantic embeddings: LSA (PPMI + randomized SVD), deterministic,
      corpus-trained; hashing fallback; per-chunk model lineage
- [x] Incremental model migration: stale-chunk re-embed jobs, measured 22.9k chunks/s
- [x] Retrieval 2.0: BM25 scores preserved, single-pass model-filtered dense scan,
      deterministic entity channel, full-window MMR, lexical-overlap reranker,
      freshness re-rank for Latest
- [x] Evidence 2.0: conflict taxonomy, negation, unit normalization, offsets, indexed checks
- [x] Entity resolution 2.0: alias/fuzzy stages, public alias+merge APIs, audit log
- [x] Graph 2.0: typed RELATES_TO_*, multi-hop BFS, centrality, graph-correct delete
- [x] Document intelligence: DOCX/XLSX/PPTX/EPUB, proper HTML, bomb guards, optional PDF feature
- [x] Jobs: atomic claim, backoff, dead-letter, cancellation, progress, worker count
- [x] Evaluation: golden qrels + ablation harness + red-team suite (77 tests total)

## v0.10 — Dense-channel scale (shipped)

- [x] Deterministic in-process HNSW (`src/ann.rs`): no RNG (splitmix64 layer
      assignment + hash-order insertion), cosine distance, positive-similarity
      parity with the brute-force channel
- [x] Exact cache invalidation via the (model, COUNT, MAX(id)) fingerprint
      (valid because vectors are immutable within a model name)
- [x] `ann_mode` policy: auto (default; crossover 20k) / brute (v0.9 contract) /
      hnsw (always); degrade-not-refuse past `max_dense_scan`
- [x] ANN crossover benchmark (`lkos-bench ann`) archived under
      `benchmarks/results/ann-crossover-v0.10.0.txt`; decision recorded in ADR-010
- [x] Two scale defects found and fixed: result-heap Ord inversion;
      insertion-order recall saturation on cluster-sequential corpora
- [x] ANN test suite: exactness vs brute force, invalidation, policy paths,
      filtered bypass (86 tests total)

## Next (ordered by whitepaper section 8)

1. BEIR-scale external evaluation
2. Persisted/serialized ANN graphs; IVF comparison at extreme N; streaming inserts
3. Pretrained local encoders via EmbeddingProvider (ONNX/fastembed)
4. Cross-encoder reranking behind the ablation harness
5. NER-grade extraction on a labeled slice
6. Bitemporal query planning over claim validity
