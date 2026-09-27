# Evaluation

## Golden set — two fixed configurations (tests/golden_eval.rs)

16 docs (4 topics x 4; 16 chunks), 16 queries, document-level graded judgments
(3=directly answers, 2=topical, 1=marginal). Metrics: Recall@5/10, MRR,
nDCG@10 (gain 2^g-1). Ablations run lexical-only / dense-only / hybrid.

**Two fixed configurations are evaluated since v0.10.1 (issue #8):**

- **semantic-off** — dense channel on the hashing fallback. This was the ONLY
  configuration before v0.10.1: the golden suite set
  `semantic_min_chunks = usize::MAX`, so the trained semantic layer was never
  evaluated anywhere in CI (audit finding B).
- **semantic-on** — the same corpus trains the LSA model (16 chunks;
  `semantic_min_chunks = 12`, dim 8, `min_df = 1` for this fixed corpus) and
  every chunk is migrated; `dense` and `hybrid` rows measure the real
  corpus-trained channel. Determinism across freshly built engines is asserted.

Measured (hybrid includes reranker), `benchmarks/results/v0.10.1-evaluation-integrity.txt`:

| Configuration | Mode | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| semantic-off | lexical | 0.938 | 0.938 | 0.938 | 0.873 |
| semantic-off | dense (hashing) | 1.000 | 1.000 | 1.000 | 0.966 |
| semantic-off | hybrid | 1.000 | 1.000 | 1.000 | 0.966 |
| semantic-on | lexical | 0.938 | 0.938 | 0.938 | 0.873 |
| semantic-on | dense (LSA) | 1.000 | 1.000 | 1.000 | 0.956 |
| semantic-on | hybrid | 1.000 | 1.000 | 1.000 | 0.956 |

Honest floors enforced by CI (both configurations): hybrid MRR >= 0.50,
nDCG@10 >= 0.55, Recall@10 >= 0.85, hybrid within 0.05 MRR of the best single
channel; semantic-on adds dense-channel floors (Recall@10 >= 0.60,
nDCG@10 >= 0.45) so a semantic regression localizes to the LSA layer, and a
no-loss constraint against the semantic-off hybrid (training must not degrade
the keyword-matchable query set). Reproducibility across engines is asserted
for both configurations (deterministic ranking).

Scope: regression protection on a small fixed corpus — NOT a leaderboard.
BEIR-scale evaluation is the top research-frontier item (issue #1).

## Paraphrase-transfer probe (tests/paraphrase_probe.rs)

A paraphrase claim needs queries that CANNOT succeed lexically. The probe
corpus (4 themes x 5 docs, one chunk each) is built so that for each of the
8 pairs the query shares **zero tokens** with its target chunk (asserted at
runtime against the production tokenizer) and the query vocabulary occurs
only in a sibling "bridge" chunk that co-occurs it with target-exclusive
vocabulary. Ranking the target through the dense channel is therefore only
possible through the trained embedding space. Corpus constraints (theme-
exclusive content words, no high-document-frequency tokens) are checked by
`scripts/check_probe_corpus.py` design tooling; ubiquitous function words
create global TF-IDF correlation that leaks across themes.

Measured (LSA dim 8, spectral selection — ADR-011):

| Pair (query -> target) | LSA rank | Hashing-fallback rank |
|---|---|---|
| guitar chords -> music-strings | 2 | 7 |
| singer vocals -> music-opera | 2 | 5 |
| fertilizer vegetables -> garden-soil | 2 | miss (>24) |
| blooming flowers -> garden-roses | 2 | miss (>24) |
| graphics card -> compute-tensor | 2 | miss (>24) |
| supercomputer datacenter -> compute-cluster | 2 | 10 |
| underwater ecosystems -> ocean-coral | 2 | 4 |
| moon gravity -> ocean-tides | 2 | 4 |
| **mean reciprocal rank** | **0.500** | **0.118** |

Floors enforced by CI: every target in the top-3, mean RR >= 0.40, every
document ranked above a target from the same theme (no cross-topic leakage),
and a discrimination canary: the SAME pairs on the hashing fallback must miss
the top-3 on >= 3 pairs and lose by >= 0.20 MRR — proving the probe fails if
the semantic layer degrades to lexical-grade behavior (issue #8 acceptance:
"a semantic-regression is detectable").

The rank-2 pattern is structural, not tuned luck: the bridge chunk that
literally contains the query vocabulary ranks 1; the target is its unique
co-occurrence partner.

## Trainer correctness note (ADR-011)

Building this probe exposed a trainer defect: the randomized-SVD subspace was
truncated in sketch order rather than by energy, so `lsa_dim` selected an
arbitrary subspace (targets scattered ranks 2–9; cross-theme documents ranked
above targets). v0.10.1 selects the top-`dim` sketch columns by
pre-orthonormalization energy (≈ singular values; Halko et al. 2011 §1.4).
Same-corpus vectors change; retrain via the normal `train_semantic_index`
cycle. See ADR-011 and docs/EMBEDDINGS.md.

## System benchmark

`lkos-bench` (self-supervised sanity + latency + throughput). Archived:
benchmarks/results/v0.9.0-run1.txt. The bench generates claim-bearing
sentences so the evidence layer is exercised (v0.1 produced 0 claims).

## Naming note

The persisted model name `lsa-pmi-svd-v1` is a legacy identifier from v0.9:
the weighting was always TF-IDF ((1+ln tf)·ln(1+N/df)), never PMI. Doc claims
saying "PPMI" were corrected in v0.10.1; the stored name is kept for embedding
lineage stability (renaming would orphan every trained index).
