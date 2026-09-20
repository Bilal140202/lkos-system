# Spec Archive — Where This Implementation Came From

LKOS v0.1 was not designed in a vacuum. This directory preserves the complete
source-material chain, per the project's provenance-first principle (applied to
the project itself):

| File | Role |
|---|---|
| `SOURCE-A-privacythink-architecture.md` | The original **PrivacyThink / LKOS architecture**: RAG-foundation phases, background summarization pipeline, the "front-load intelligence into ingestion" thesis, the <3s answer budget. |
| `SOURCE-B-repository-audit-roadmap.md` | The independent **repository audit**: verdict that spec outran implementation, the 12-stage maturity ladder, the demand that every documented capability map to executable code + tests + benchmarks. |
| `legacy/README-v0-spec.md` | The v0 repository README (specification-only state). |
| `legacy/AGENTS.md`, `legacy/CLAUDE.md` | Agent-protocol documents from the v0 repository state. |

What changed between spec and v0.1 (the audit's central demand, satisfied):

- The described engine now **exists**: ~6.4k lines of Rust, 18 modules, real
  SQLite schema v1–v4, 42 passing tests, CI, benchmark harness.
- Claims that could not be kept in v0.1 (PDF/DOCX, HNSW, semantic embeddings)
  are registered in [`../NON_GOALS.md`](../NON_GOALS.md) instead of being
  implied by prose — documentation no longer outruns implementation.
- The audit's "Stage 1 — Build the Real Rust Core" is this release; the
  remaining stages map to [`../ROADMAP.md`](../ROADMAP.md).
