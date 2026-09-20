# LKOS Agent Protocols & Development Guidelines

> Standard operating procedures for AI agents, developers, and autonomous systems maintaining or extending the LKOS architecture.

## 1. Prime Directives
- **Zero Cloud Leakage**: User documents, extracted entities, and vector embeddings must never leave the local device.
- **Carmack First-Principles**: Prefer deterministic algorithms (regex heuristics, compiled FTS5 indices, exact cosine similarity) over speculative LLM steps when extracting structure.
- **Idempotency**: All database migrations, background pipelines, and file system creations must be strictly idempotent.

## 2. Code Boundaries
- `src/vector/store.rs`: The sole authority for SQLite access. All schema migrations and SQL queries reside here.
- `src/rag/knowledge_object.rs`: Pure functional extraction logic. Does not touch IO or the database directly.
- `src/background.rs`: Tokio async orchestration. Manages transition states (`indexing` -> `ready` -> `summarizing` -> `complete`).
- `src/llm/llamacpp_subprocess.rs`: Manages external `llama-cli` process execution via file pipes.

## 3. Workflow for Adding New Capabilities
1. Update `capabilities/default.json` with granular permissions.
2. Implement storage primitives in `vector/store.rs`.
3. Expose IPC command in `commands/`.
4. Register handler in `lib.rs`.
5. Add unit and integration tests in `vector/tests.rs`.