//! Versioned SQL migrations. Each migration is idempotent in the sense that it
//! is only applied when `PRAGMA user_version` is below its version.
//!
//! See `docs/MIGRATIONS.md` for the narrative history.

/// v1 — base schema: documents, chunks, FTS5 external-content index.
pub const V1_BASE: &str = r#"
BEGIN;

CREATE TABLE IF NOT EXISTS documents (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    path                 TEXT UNIQUE,
    filename             TEXT NOT NULL,
    doc_type             TEXT NOT NULL DEFAULT 'text',
    size                 INTEGER NOT NULL DEFAULT 0,
    content_hash         TEXT UNIQUE,
    text_chars           INTEGER NOT NULL DEFAULT 0,
    readiness_state      TEXT NOT NULL DEFAULT 'pending',
    summary              TEXT,
    summary_generated_at TEXT,
    section_count        INTEGER NOT NULL DEFAULT 0,
    chunk_count          INTEGER NOT NULL DEFAULT 0,
    extractor_version    TEXT,
    chunker_version      TEXT,
    embedding_model      TEXT,
    knowledge_version    TEXT,
    created_at           TEXT NOT NULL,
    indexed_at           TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    text            TEXT NOT NULL,
    section_title   TEXT,
    kind            TEXT NOT NULL DEFAULT 'prose',
    start_offset    INTEGER NOT NULL DEFAULT 0,
    end_offset      INTEGER NOT NULL DEFAULT 0,
    authority_score REAL NOT NULL DEFAULT 1.0,
    content_hash    TEXT,
    embedding       BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

-- External-content FTS5 index over chunk text. Kept in sync with triggers so
-- the lexical index can never drift from the canonical table.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

COMMIT;
"#;

/// v2 — knowledge enrichment: per-chunk knowledge objects, term document
/// frequencies (for TF-IDF keywords), normalized document metadata, and the
/// deferred-pipeline payload column.
pub const V2_KNOWLEDGE: &str = r#"
BEGIN;

ALTER TABLE chunks ADD COLUMN knowledge_json TEXT;
ALTER TABLE documents ADD COLUMN language TEXT DEFAULT 'en';
ALTER TABLE documents ADD COLUMN pending_text TEXT;

CREATE TABLE IF NOT EXISTS term_df (
    term TEXT PRIMARY KEY,
    df   INTEGER NOT NULL DEFAULT 0
);

COMMIT;
"#;

/// v3 — entities, mentions, claims, conflicts, relationships, provenance.
pub const V3_GRAPH_CLAIMS: &str = r#"
BEGIN;

CREATE TABLE IF NOT EXISTS entities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_key TEXT UNIQUE NOT NULL,
    display_name  TEXT NOT NULL,
    entity_type   TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 0,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    metadata      TEXT
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias     TEXT NOT NULL,
    PRIMARY KEY (entity_id, alias)
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL,
    chunk_id    INTEGER NOT NULL,
    surface     TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    start_offset INTEGER NOT NULL DEFAULT 0,
    end_offset   INTEGER NOT NULL DEFAULT 0,
    confidence  REAL NOT NULL DEFAULT 0.5,
    extractor   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_mentions_doc ON entity_mentions(document_id);
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON entity_mentions(chunk_id);

CREATE TABLE IF NOT EXISTS claims (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subject      TEXT NOT NULL,
    predicate    TEXT NOT NULL,
    object       TEXT NOT NULL,
    sentence     TEXT NOT NULL,
    document_id  INTEGER NOT NULL,
    chunk_id     INTEGER NOT NULL,
    confidence   REAL NOT NULL DEFAULT 0.4,
    extractor    TEXT NOT NULL,
    subject_key  TEXT,
    predicate_key TEXT,
    valid_from   TEXT,
    valid_until  TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_key);
CREATE INDEX IF NOT EXISTS idx_claims_doc ON claims(document_id);

CREATE TABLE IF NOT EXISTS claim_conflicts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_key   TEXT NOT NULL,
    predicate_key TEXT NOT NULL,
    claim_a       INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    claim_b       INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    delta         REAL NOT NULL,
    explanation   TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    weight            REAL NOT NULL DEFAULT 1.0,
    first_document_id INTEGER,
    last_document_id  INTEGER,
    created_at        TEXT NOT NULL,
    UNIQUE (source_entity_id, target_entity_id, relationship_type)
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);

CREATE TABLE IF NOT EXISTS provenance (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_type     TEXT NOT NULL,
    artifact_id       TEXT NOT NULL,
    document_id       INTEGER,
    chunk_id          INTEGER,
    start_offset      INTEGER,
    end_offset        INTEGER,
    extractor         TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prov_artifact ON provenance(artifact_type, artifact_id);
CREATE INDEX IF NOT EXISTS idx_prov_doc ON provenance(document_id);

COMMIT;
"#;

/// v4 — job queue and engine metadata.
pub const V4_JOBS_META: &str = r#"
BEGIN;

CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'pending',
    priority     INTEGER NOT NULL DEFAULT 5,
    attempts     INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    last_error   TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority);

CREATE TABLE IF NOT EXISTS engine_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

COMMIT;
"#;

/// v5 — semantic model storage, per-chunk embedding lineage, claim offsets,
/// conflict-detection indexes, job scheduling columns, entity merge audit.
pub const V5_SEMANTIC_INCREMENTAL: &str = r#"
BEGIN;

-- Latent term vectors of the corpus-trained LSA model.
CREATE TABLE IF NOT EXISTS lsa_terms (
    term   TEXT PRIMARY KEY,
    idx    INTEGER NOT NULL,
    vector BLOB NOT NULL
);

-- Which embedding model produced each chunk's vector. NULL on legacy rows
-- (v0.1 data was hashing-lex-v1); backfilled immediately after migration.
ALTER TABLE chunks ADD COLUMN embedding_model TEXT;
UPDATE chunks SET embedding_model = 'hashing-lex-v1' WHERE embedding_model IS NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_model ON chunks(embedding_model);

-- Character offsets of the claim's source sentence span within its chunk.
ALTER TABLE claims ADD COLUMN start_offset INTEGER;
ALTER TABLE claims ADD COLUMN end_offset INTEGER;

-- Conflict detection needs (subject_key, predicate_key) lookups, not scans.
CREATE INDEX IF NOT EXISTS idx_claims_keys ON claims(subject_key, predicate_key);
CREATE INDEX IF NOT EXISTS idx_conflicts_keys ON claim_conflicts(subject_key, predicate_key);
ALTER TABLE claim_conflicts ADD COLUMN conflict_kind TEXT NOT NULL DEFAULT 'undated-disagreement';

-- Job scheduling: exponential backoff release time + progress percentage.
ALTER TABLE jobs ADD COLUMN run_at TEXT;
ALTER TABLE jobs ADD COLUMN progress INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_jobs_sched ON jobs(status, priority, run_at);

-- Entity merge audit trail (resolution transparency is part of the contract).
CREATE TABLE IF NOT EXISTS entity_merge_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    survivor_id  INTEGER NOT NULL,
    merged_id    INTEGER NOT NULL,
    reason       TEXT NOT NULL,
    merged_at    TEXT NOT NULL
);

COMMIT;
"#;
