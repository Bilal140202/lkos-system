# TESTING.md — Verification Strategy

## Suite

`cargo test` → 77 tests, all green, fully offline:

| Suite | Count | Covers |
|---|---|---|
| `tests/engine_tests.rs` | 23 | end-to-end invariants through the public API |
| `tests/unit_quality.rs` | 19 | deterministic core units + golden retrieval corpus |
| `tests/semantic_tests.rs` | 9 | LSA training determinism, model lineage, incremental re-embed, semantic e2e |
| `tests/security_hostile.rs` | 12 | adversarial red-team: oversize/corrupt inputs, FTS injection, idempotency under attack, graph-correct deletion |
| `tests/golden_eval.rs` | 2 | graded qrels evaluation (Recall/MRR/nDCG) + per-mode ablation floors |
| module units (in-crate) | 10 | LSA math, Jaro–Winkler, temporal parsing |
| doc-tests | 2 | README/engine examples compile |

## End-to-end invariants (property-style, spec §95)

- **Idempotent ingestion** — same bytes → same document, no duplicate chunks.
- **Changed content → new version** — content-hash identity.
- **Delete cascade** — chunks, mentions, provenance, claim conflicts all removed; entity mention counts decrement.
- **Backup/restore round-trip** — backup file opens, is searchable, passes integrity check.
- **Restart preservation** — reopen keeps documents and chunk counts.
- **LLM-optional** — no provider: `ready` state, full retrieval, typed `ask` error. Failing provider: ingestion still completes (`ready`).
- **Structural refusal** — zero evidence → fixed refusal string, no generation.
- **Empty/binary input** — typed errors, never a panic.
- **Event stream** — lifecycle events delivered for added/ready/knowledge-updated/deleted.
- **Explainability** — every hit has non-empty `matched_by`; hybrid hits cite both channels.

## Golden retrieval corpus

`tests/unit_quality.rs::golden_retrieval_miniature_corpus` is a fixed 4-document
corpus with 6 probes and rank-1 expectations. A ranking change that breaks a
golden is a retrieval regression **or** a deliberate improvement — either way it
must be resolved consciously in the PR, which is the anti-regression mechanism
the repository audit demanded.

## CI

`.github/workflows/ci.yml` runs on Linux, macOS, Windows:
`cargo fmt --check` → `cargo clippy --all-targets -- -D warnings` →
`cargo test --release` → `cargo build --release`.
