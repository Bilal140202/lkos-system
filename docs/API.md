# API.md — The Engine Facade

`Lkos` is clone-safe (shared `Arc` inner); all methods are synchronous.

## Lifecycle
| Method | Notes |
|---|---|
| `open(path, config)` | creates/opens the DB; verifies embedding identity; recovers interrupted jobs |
| `open_in_memory(config)` | temp-file-backed engine for tests/examples |
| `set_llm(provider)` | optional; `Arc<dyn LlmProvider>` |
| `subscribe()` | `mpsc::Receiver<Event>` of lifecycle events |
| `close()` | stops background worker |
| `config()` | active `Config` |

## Ingestion
| Method | Notes |
|---|---|
| `ingest_bytes(filename, bytes)` | extract→normalize→hash→chunk→embed→knowledge→provenance; idempotent on identical content; errors on empty/binary |
| `ingest_file(path)` | as above from disk; stores the path |
| `process_document(doc_id)` | run/re-run the pipeline (used by the background worker) |

## Querying
| Method | Notes |
|---|---|
| `query(req: QueryRequest) -> QueryResponse` | plan → retrieve → fuse → MMR → assemble; `QueryResponse` has `intent`, `plan_explanation`, `hits` (each with `matched_by`), `context` (budgeted + citations), optional `provenance`, `related_entities`, `elapsed_us` |
| `ask(question) -> String` | grounded generation via the configured provider; structural refusal with zero evidence; clean error without a provider |
| `summary(doc_id) -> Option<String>` | pre-built summary (fast path data) |

`QueryRequest::new(text)` defaults: `top_k = 8`, `mode = Auto`,
`context_budget = 3000`, `max_per_document = 3`; builders `.top_k(k)` /
`.mode(m)`; `Filters` supports document ids, doc types, must-contain,
entities, as-of.

## Library management
`documents()`, `document(id)`, `delete_document(id)` (full cascade),
`backup_to(dest)`, `integrity_check()`, `stats()` (`LibraryStats`).

## Knowledge & graph
`document_entities(doc)`, `list_entities(limit)`, `entity_id_by_name(name)`,
`neighborhood(entity_id, limit)` (center + neighbors + documents),
`document_claims(doc)`, `claims_about(subject)` (canonicalized),
`conflicts(limit)`, `provenance_of(artifact_type, artifact_id)`.

## Performance & privacy characteristics
All methods are local-only; every call opens a short-lived SQLite connection
(pool-less by design for v0.1); query p50 ≈ 15 ms hybrid at 3.6k chunks
(see BENCHMARKS.md). No method performs network I/O under any configuration.
