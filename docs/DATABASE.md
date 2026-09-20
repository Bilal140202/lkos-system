# DATABASE.md — Storage, Migrations, Durability

## Canonical store

One SQLite file (`library.lkos` by default). WAL journal, `synchronous =
NORMAL`, `foreign_keys = ON`, `busy_timeout = 5000ms`. The file is portable:
`backup_to` uses the SQLite online backup API (works against a live database),
and `integrity_check` exposes `PRAGMA integrity_check` rows.

## Schema versions (`PRAGMA user_version`)

| Version | Contents |
|---|---|
| 1 | `documents`, `chunks`, `chunks_fts` (external content, porter) + sync triggers |
| 2 | `chunks.knowledge_json`, `documents.language`, `documents.pending_text`, `term_df` |
| 3 | `entities`, `entity_aliases`, `entity_mentions`, `claims`, `claim_conflicts`, `relationships`, `provenance` |
| 4 | `jobs`, `engine_meta` |

Migrations run sequentially on open; each is applied only when
`user_version` is below its number. `chunks_fts` is an external-content index
kept consistent by `AFTER INSERT/DELETE/UPDATE` triggers on `chunks`, so the
lexical index cannot drift from the canonical table.

## Identity and incremental indexing

- Document identity = `content_hash` (SHA-256 of raw bytes, UNIQUE).
  Re-ingesting identical bytes returns the existing document unchanged
  (idempotent; pinned by test).
- Changed bytes with the same path create a **new version row**; the old row's
  path is released (NULLed) so the new version claims it. Historical versions
  remain searchable by content.
- Chunk identity = rowid + `content_hash` of chunk text (embedding reuse on
  re-index is the v0.4 incremental step; the hash is already stored).

## Crash safety

- WAL + NORMAL sync: a killed process leaves a consistent database.
- Jobs stuck in `running` are requeued (`status = 'pending'`) on next open by
  `recover_running_jobs`; the count is recorded in `engine_meta.last_recovery`.
- Deletion order is: mentions (with entity-count decrements) → claims →
  document row; `ON DELETE CASCADE` covers chunks, FTS (via triggers),
  claim conflicts, relationships, and provenance.

## Tables (quick reference)

`documents`, `chunks`, `chunks_fts`, `term_df`, `entities`,
`entity_aliases`, `entity_mentions`, `claims`, `claim_conflicts`,
`relationships`, `provenance`, `jobs`, `engine_meta` — full DDL in
`src/storage/schema.rs` (the executable source of truth).
