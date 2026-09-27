# Embeddings

`EmbeddingProvider` (model-agnostic trait): `name()`, `dim()`, `embed_batch()`.

## Providers

| Provider | name | Semantic? | Notes |
|---|---|---|---|
| `LsaEmbedder` (untrained) | `hashing-lex-v1` | No (lexical hashing) | cold start; Weingberger-style unigram+bigram+char-trigram |
| `LsaEmbedder` (trained) | `lsa-pmi-svd-v1` | Corpus-relative semantics | TF-IDF matrix ((1+ln tf)·ln(1+N/df)) + randomized SVD; the model name is a legacy identifier from v0.9 (the weighting was never PMI) |

Trainer note (v0.10.1, ADR-011): the truncated subspace is selected **spectrally** — sketch columns are ranked by pre-orthonormalization energy (≈ singular values, Halko et al. 2011 §1.4) and the top-`dim` kept. Earlier releases kept the first `dim` sketch-order columns, an arbitrary subspace that measurably scrambled cross-topic similarity (detected by the issue #8 paraphrase probe).

## Determinism

- vocabulary order: DF desc, then lexicographic (total order)
- sketch: fixed-seed Knuth LCG -> Box-Muller Gaussians
- sign fix: each latent's largest component positive
- test: `lsa_training_is_deterministic` (bit-identical vectors)

## Lifecycle

1. Ingest with cold-start provider (chunks record `embedding_model`).
2. `engine.train_semantic_index()` when chunks >= `semantic_min_chunks`.
3. `engine.reembed_stale_chunks("lsa-pmi-svd-v1", ...)` migrates batches of
   256 with progress; measured 22.9k chunks/s.
4. Reopen restores the persisted model from `lsa_terms`; unloadable model =
   typed `EmbeddingMismatch` error (never mix spaces).

Measured: training 2400 chunks in 0.06 s (bench v0.9.0-run1).
