# ARCHITECTURE.md — Index

The full architectural description lives in the repository's research-paper-style
[README](../README.md) (§4 Architecture, §5 Retrieval, §6 Knowledge layer, §7
Planner, §8 Local AI). This file is the navigation index:

| Document | Contents |
|---|---|
| [README.md](../README.md) | the paper: design, algorithms, evaluation, positioning |
| [RETRIEVAL.md](RETRIEVAL.md) | channels, RRF, boosts, MMR, planner weights, temporal filter |
| [DATABASE.md](DATABASE.md) | schema v1–v4, identity model, crash safety, cascades |
| [API.md](API.md) | the `Lkos` facade — every public method, contracts |
| [SECURITY_PRIVACY.md](SECURITY_PRIVACY.md) | threat model + privacy posture |
| [TESTING.md](TESTING.md) | suite map, property invariants, golden corpus, CI |
| [BENCHMARKS.md](BENCHMARKS.md) | methodology, measured results, honest reading |
| [NON_GOALS.md](NON_GOALS.md) | unimplemented-by-design register + entry criteria |
| [ROADMAP.md](ROADMAP.md) | v0.2 → v1.0 ladder with done-tests |
| [adr/ADR.md](adr/ADR.md) | ADR-001/005/006/007/008/009 (decision records) |

Layering (dependencies point downward only):

```
bin (cli, bench) → engine facade → query | knowledge | retrieval
                                 → ingestion → chunking → embeddings
                                 → entities / claims / graph / temporal / provenance
                                 → jobs / events / llm
                                 → storage (SQLite, WAL, migrations)
```

Module boundaries: `storage` is the only owner of SQL; `retrieval` never writes;
`knowledge` is pure (no I/O); `llm` is behind a trait and swappable; the engine
facade is the single entry point for applications.
