# LKOS State of the Art — Synthesis (v0.9)

Scope: what the field knows about building local retrieval/knowledge engines,
what LKOS adopts, modifies, or rejects, and why. Triangulation rule: no
decision from a single source; every adoption below is backed by a paper +
implementation reality + LKOS measurement (or explicitly marked future work).

## Adopted (with evidence)

1. **BM25/FTS5 lexical channel** — Robertson&Zaragoza 2009; SQLite FTS5 porter
   index with trigger sync. Kept from v0.1; scores now preserved.
2. **RRF fusion, k=60** — Cormack 2009. Robust when channel scores are
   incomparable (ours are: BM25 vs cosine). Adopted unchanged.
3. **MMR diversity, lambda 0.7** — Carbonell&Goldstein 1998. Window fixed
   (v0.1 degraded silently past k+16).
4. **LSA via PPMI/tf-idf + randomized SVD** — Deerwester 1990; Halko 2011.
   Adopted as the corpus-relative semantic provider with deterministic
   seeds/sign fixing (our addition, test-asserted).
5. **Feature hashing fallback** — Weinberger 2009. Retained as cold-start;
   honestly labeled lexical.
6. **Jaro-Winkler with strict threshold + blocking** — Winkler 1989. Adopted
   with type guarding; false-merge risk deliberately minimized.
7. **SQLite WAL + external-content FTS5** — crash-safe single-file substrate.
   Retained; downgrade guard added.
8. **Exponential backoff + dead-letter** — standard durable-queue practice
   (Kleppmann 2017 context). Adopted for the job system.

## Modified (and why)

- **Temporal filtering**: pure filters drop the newest evidence for "latest"
  queries; we re-rank for Latest and filter only explicit Year/Range with
  fallback (measured behavior in tests).
- **Conflict detection**: NLI is out of scope without models; numeric
  disagreement + negation with explicit taxonomy preserves the useful subset
  deterministically. Pair enumeration bounded (quadratic pathology measured).

## Rejected (for now, with reasons)

- **Pretrained encoders as default** — violates zero-download local-first
  contract; trait slot exists (ADR-005), pending optional-feature design.
- **Graph databases** — relational SQLite covers 1-2-hop entity pages at our
  scale; introducing another engine must first beat SQL traversal in a
  benchmark (NON_GOALS).
- **Learned sparse (SPLADE-style)** — requires model artifacts; revisit with
  the ONNX provider.
- **Microservices / external vector DBs** — anti-AI-slop rule: every component
  must earn its existence; SQLite does.

## Frontier (ordered)

BEIR-scale evaluation > HNSW with measured crossover > pretrained local
encoders > cross-encoder reranking > NER-grade extraction > bitemporal
planning. Rationale and gating in whitepaper section 8.
