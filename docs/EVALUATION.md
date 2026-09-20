# Evaluation

## Golden set (tests/golden_eval.rs)

16 docs (4 topics x 4), 16 queries, document-level graded judgments
(3=directly answers, 2=topical, 1=marginal). Metrics: Recall@5/10, MRR,
nDCG@10 (gain 2^g-1). Ablations run lexical-only / dense-only / hybrid.

Measured (hashing fallback active; hybrid includes reranker):

| Mode | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| lexical | 0.938 | 0.938 | 0.938 | 0.873 |
| dense | 1.000 | 1.000 | 1.000 | 0.966 |
| hybrid | 1.000 | 1.000 | 1.000 | 0.966 |

Honest floors enforced by CI: hybrid MRR >= 0.50, nDCG@10 >= 0.55,
Recall@10 >= 0.85, and hybrid within 0.05 MRR of the best single channel.
Reproducibility across engines is asserted (deterministic ranking).

Scope: regression protection on a small fixed corpus — NOT a leaderboard.
BEIR-scale evaluation is the top research-frontier item.

## System benchmark

`lkos-bench` (self-supervised sanity + latency + throughput). Archived:
benchmarks/results/v0.9.0-run1.txt. The bench generates claim-bearing
sentences so the evidence layer is exercised (v0.1 produced 0 claims).
