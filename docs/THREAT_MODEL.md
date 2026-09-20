# Threat Model

Assets: documents, embeddings, entities/claims (knowledge graph), the
library file itself.

## Attack surface & mitigations (implemented, test-verified)

| Vector | Mitigation | Test |
|---|---|---|
| Oversized input (DoS) | 64 MiB raw cap, typed error | oversize_input_rejected_at_cap |
| Decompression bombs | 256 MiB take() cap (not header trust), 4096 entries | zip_bomb_entry_size_capped |
| Corrupt containers | zip/xml parse errors -> typed error, no panic | fake_docx_rejected_not_panicked |
| Script/style injection into index | script/style/head bodies dropped, entities decoded | hostile_html_never_leaks_script_bodies |
| FTS5 MATCH injection | tokenized, quoted, prefix-AND; hostile shapes tested + post-attack health check | fts_injection_queries_are_safe |
| Hostile metadata / unicode | normalization strips NUL/control; zalgo/emoji/BOM survive | hostile_metadata_and_unicode_survive, deep_unicode_... |
| Duplicate-ingestion abuse | content-hash idempotency | duplicate_ingestion_is_idempotent... |
| Query DoS | empty rejected; 1 MB query bounded (return-or-error, no hang) | query_api_rejects_empty_and_oversized_queries |
| Phantom graph after delete | subgraph recompute from surviving evidence | document_delete_is_graph_correct... |

## Out of scope (honest)

- Malicious PDF/office payloads beyond the caps (parser CVE surface) — the
  `pdf` feature is opt-in; keep system parsers sandboxed for hostile use.
- Cross-user ACLs / multi-tenant encryption (single-user local file).
- Supply-chain: `cargo audit` in CI; no HTTP clients in the dependency graph;
  no runtime network calls by construction (privacy by architecture, plus
  the dependency-tree audit in CI).

## Privacy notes

Local-first: no telemetry, no external model calls unless an LLM provider is
explicitly installed by the application. `close()` cancels pending jobs and
removes in-memory scratch files. Deletion removes rows; SQLite may retain
bytes in free pages until VACUUM (documented; `lkos-cli` exposes integrity
+ backup, VACUUM available on Store).
