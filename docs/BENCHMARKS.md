# BENCHMARKS.md — Methodology and Results

## Harness

`lkos-bench [--docs N] [--queries N] [--k K] [--json]` — deterministic
xorshift64* corpus (N documents × 12 paragraphs ≈ 4 KB each → chunks), in-memory
library, warm-up queries, then:

1. **Ingestion throughput** — wall time for the full pipeline (extract → chunk →
   embed → knowledge → provenance).
2. **Query latency** — N queries per mode (lexical / vector / hybrid), reported
   as p50 / p95 / p99 in µs/ms.
3. **Self-supervised quality** — probes name the generating
   `(org, topic, year)` triple; ground truth = document filename(s) from a
   deterministically regenerated manifest; Recall@K and MRR over the hybrid
   stack.

The corpus is synthetic and the probes echo document titles: quality numbers
validate the fusion stack end-to-end (planner → channels → RRF → MMR →
assembly) and guard against regressions. They are **not** a human benchmark and
must never be quoted as one (README §9 states this explicitly).

## Results — v0.1.0, release build, 2 vCPU container, 300 docs / 3,600 chunks

```
== Ingestion ==
  corpus           : 300 documents, 3600 chunks, 1202535 chars
  wall time        : 37.39 s
  throughput       : 8.0 docs/s, 96 chunks/s

== Query latency (150 queries per mode) ==
  lexical (BM25)       p50    2.53 ms   p95    4.01 ms   p99    4.22 ms
  vector (hash embed)  p50   14.17 ms   p95   14.40 ms   p99   15.13 ms
  hybrid (RRF)         p50   14.87 ms   p95   16.78 ms   p99   17.16 ms

== Retrieval quality (hybrid, self-supervised) ==
  Recall@10        : 1.000
  MRR              : 1.000

== Library stats ==
  entities 3552 | mentions 12000 | claims 0 | conflicts 0 | relationships 7202 | provenance 15600
```

Machine-readable copies live in `benchmarks/results/`.

## Reading the numbers honestly

- **Dense ≈ 14 ms is brute force** over 3,600 × 256-dim vectors — linear in
  corpus size with the `max_dense_scan` hard stop. HNSW (v0.2) is the fix; the
  ceiling where it becomes mandatory (~250k chunks) is configured, not folklore.
- **Hybrid ≈ dense + 0.7 ms** — RRF, boosts, and MMR are effectively free
  relative to the scan; lexical p50 2.5 ms shows the FTS5 substrate is not the
  bottleneck.
- **Ingestion 96 chunks/s** is knowledge-extraction-dominated (per-chunk regex
  extraction + provenance writes). Batched inserts and lazy extraction are the
  next levers; measured again after each change.
- **Claims = 0 on this corpus** — expected: the synthetic sentences do not
  contain SVO metric patterns. Conflict detection is validated by dedicated
  tests (`claims_and_conflicts_are_detected_and_preserved`), not by this corpus.

## Reproducing

```bash
cargo build --release
./target/release/lkos-bench --docs 300 --queries 150 --k 10 --json | tee benchmarks/results/$(date +%F)-$(git rev-parse --short HEAD).json
```
