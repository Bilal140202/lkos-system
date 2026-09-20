# SECURITY.md & PRIVACY.md

## Threat model (v0.1 scope)

LKOS processes untrusted documents on a trusted machine, for a single local
user, with no network features. Assets: the knowledge database (contains the
user's private documents), the host filesystem, and answer integrity.

| Threat | v0.1 posture |
|---|---|
| Malicious document content (parser attacks) | Extractors are UTF-8 decode + small regex/line passes over text; binary formats are rejected, not parsed. No parser consumes untrusted binary structures. |
| FTS5 / SQL injection | Queries are parameterized everywhere; user query text reaches FTS5 only through `fts_escape` (quoted prefix tokens; operators/parens/quotes cannot change structure). Pinned by unit test. |
| Prompt injection via documents | Out of engine scope by design: LKOS returns evidence + a grounded prompt; the *application* owns the model policy. The engine's structural refusal (no evidence → no generation) removes the empty-context injection surface. |
| Knowledge poisoning | Provenance names the extractor + source for every artifact; conflicting numeric claims are surfaced as conflicts, never merged; heuristic confidence is labeled heuristic. |
| Oversized input / resource exhaustion | Chunker bounds (`chunk_max_chars`, sentence-boundary splitting), `max_dense_scan` retrieval bound, typed errors instead of panics. |
| Path traversal | `ingest_file` reads the exact path given by the embedding application; no user-controlled path resolution inside the engine. |

## Privacy posture

- **No network egress, ever.** The engine performs zero network I/O; there is no
  telemetry, no analytics, no update check, no crash reporting. The optional
  llama.cpp provider spawns a *local* subprocess reading local files.
- **Data stays in the database file** (plus WAL/SHM siblings); temporary prompt
  files created for llama.cpp are unlinked after each run; logs are absent by
  default (applications own logging).
- **Deletion is real**: `delete_document` cascades chunks, FTS entries,
  mentions, claims, conflicts, relationships, and provenance (pinned by test);
  `VACUUM` is exposed via `Store::vacuum` for physical shrink.
- **Backup is local**: `backup_to` writes a single file the user controls.

Known residual risks (documented, not hidden): prompt/temp files are created in
the OS temp dir during LLM calls (world-readable depending on platform umask);
the database file itself is unencrypted — full-disk encryption or SQLCipher is
an application-level decision and a tracked v0.3 candidate.
