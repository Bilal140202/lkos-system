# Embeddings

`EmbeddingProvider` (model-agnostic trait): `name()`, `dim()`, `embed_batch()`.

## Providers

| Provider | name | Semantic? | Notes |
|---|---|---|---|
| `LsaEmbedder` (untrained) | `hashing-lex-v1` | No (lexical hashing) | cold start; Weingberger-style unigram+bigram+char-trigram |
| `LsaEmbedder` (trained) | `lsa-pmi-svd-v1` | Corpus-relative semantics | PPMI/tf-idf matrix + randomized SVD, deterministic seeds/signs |

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
