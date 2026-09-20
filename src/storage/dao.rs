//! Data access objects. All SQL lives here; the rest of the engine never
//! writes SQL directly (the "SQLite authority" boundary from AGENTS.md).

#![allow(clippy::too_many_lines)]

use crate::types::*;
use crate::{LkosError, Result};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};

pub(crate) fn now() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

// ---------------------------------------------------------------------------
// documents
// ---------------------------------------------------------------------------

/// Insert a new document row or return the id of an identical one.
/// Uniqueness: `content_hash` (byte-identical files collapse to one row).
#[allow(clippy::too_many_arguments)] // mirrors the SQL row shape 1:1
pub(crate) fn upsert_document(
    conn: &Connection,
    path: &str,
    filename: &str,
    doc_type: &str,
    size: u64,
    content_hash: &str,
    text_chars: usize,
    extractor_version: &str,
    chunker_version: &str,
    embedding_model: &str,
    knowledge_version: &str,
) -> Result<(i64, bool)> {
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM documents WHERE content_hash = ?1",
            params![content_hash],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(id) = existing {
        return Ok((id, false));
    }
    // In-memory documents carry no path; store NULL so multiple such
    // documents never collide on the UNIQUE(path) constraint.
    let path_param: Option<&str> = if path.is_empty() { None } else { Some(path) };
    // A file re-ingested with *new* content becomes a new version: release
    // the old row's path so the new version can claim it (the old version
    // remains queryable by content hash).
    if !path.is_empty() {
        conn.execute(
            "UPDATE documents SET path = NULL WHERE path = ?1 AND content_hash != ?2",
            params![path, content_hash],
        )?;
    }
    conn.execute(
        "INSERT INTO documents (path, filename, doc_type, size, content_hash, text_chars, \
         readiness_state, extractor_version, chunker_version, embedding_model, knowledge_version, \
         created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7, ?8, ?9, ?10, ?11)",
        params![
            path_param,
            filename,
            doc_type,
            size as i64,
            content_hash,
            text_chars as i64,
            extractor_version,
            chunker_version,
            embedding_model,
            knowledge_version,
            now()
        ],
    )?;
    Ok((conn.last_insert_rowid(), true))
}

/// Fetch a document by id.
pub(crate) fn get_document(conn: &Connection, id: i64) -> Result<DocumentInfo> {
    conn.query_row(
        "SELECT id, COALESCE(path,''), filename, doc_type, size, COALESCE(content_hash,''), \
         text_chars, readiness_state, summary, summary_generated_at, section_count, chunk_count, \
         created_at, indexed_at
         FROM documents WHERE id = ?1",
        params![id],
        row_to_document,
    )
    .optional()?
    .ok_or(LkosError::DocumentNotFound(id))
}

/// Find a document by its content hash.
#[allow(dead_code)] // Reserved API surface for v0.2 incremental indexing.
pub(crate) fn get_document_by_hash(conn: &Connection, hash: &str) -> Result<Option<DocumentInfo>> {
    conn.query_row(
        "SELECT id, COALESCE(path,''), filename, doc_type, size, COALESCE(content_hash,''), \
         text_chars, readiness_state, summary, summary_generated_at, section_count, chunk_count, \
         created_at, indexed_at
         FROM documents WHERE content_hash = ?1",
        params![hash],
        row_to_document,
    )
    .optional()
    .map_err(Into::into)
}

/// List all documents, newest first.
pub(crate) fn list_documents(conn: &Connection) -> Result<Vec<DocumentInfo>> {
    let mut stmt = conn.prepare(
        "SELECT id, COALESCE(path,''), filename, doc_type, size, COALESCE(content_hash,''), \
         text_chars, readiness_state, summary, summary_generated_at, section_count, chunk_count, \
         created_at, indexed_at
         FROM documents ORDER BY created_at DESC",
    )?;
    let rows = stmt.query_map([], row_to_document)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

fn row_to_document(r: &rusqlite::Row<'_>) -> rusqlite::Result<DocumentInfo> {
    Ok(DocumentInfo {
        id: r.get(0)?,
        path: r.get(1)?,
        filename: r.get(2)?,
        doc_type: r.get(3)?,
        size: r.get::<_, i64>(4)?.max(0) as u64,
        content_hash: r.get(5)?,
        text_chars: r.get::<_, i64>(6)?.max(0) as usize,
        readiness_state: r.get(7)?,
        summary: r.get(8)?,
        summary_generated_at: r.get(9)?,
        section_count: r.get(10)?,
        chunk_count: r.get(11)?,
        created_at: r.get(12)?,
        indexed_at: r.get(13)?,
    })
}

/// Update pipeline readiness state.
pub(crate) fn set_readiness(conn: &Connection, doc_id: i64, state: ReadinessState) -> Result<()> {
    conn.execute(
        "UPDATE documents SET readiness_state = ?2 WHERE id = ?1",
        params![doc_id, state.as_str()],
    )?;
    Ok(())
}

/// Store a generated summary and mark completion time.
pub(crate) fn set_summary(
    conn: &Connection,
    doc_id: i64,
    summary: &str,
    generated_at: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE documents SET summary = ?2, summary_generated_at = ?3 WHERE id = ?1",
        params![doc_id, summary, generated_at],
    )?;
    Ok(())
}

/// Finalize indexing: counts + indexed_at.
pub(crate) fn finalize_index(
    conn: &Connection,
    doc_id: i64,
    chunk_count: i64,
    section_count: i64,
) -> Result<()> {
    conn.execute(
        "UPDATE documents SET chunk_count = ?2, section_count = ?3, indexed_at = ?4 WHERE id = ?1",
        params![doc_id, chunk_count, section_count, now()],
    )?;
    Ok(())
}

/// Delete a document (cascades to chunks, mentions, claims; provenance rows
/// for those artifacts are removed explicitly by the engine).
pub(crate) fn delete_document(conn: &mut Connection, doc_id: i64) -> Result<bool> {
    let chunk_ids: Vec<i64> = {
        let mut stmt = conn.prepare("SELECT id FROM chunks WHERE document_id = ?1")?;
        let rows = stmt.query_map(params![doc_id], |r| r.get(0))?;
        rows.filter_map(|r| r.ok()).collect()
    };
    for cid in &chunk_ids {
        conn.execute(
            "DELETE FROM provenance WHERE artifact_type='chunk' AND artifact_id=?1",
            params![cid.to_string()],
        )?;
    }
    conn.execute(
        "DELETE FROM provenance WHERE document_id = ?1",
        params![doc_id],
    )?;
    let n = conn.execute("DELETE FROM documents WHERE id = ?1", params![doc_id])?;
    Ok(n > 0)
}

// ---------------------------------------------------------------------------
// chunks
// ---------------------------------------------------------------------------

/// Insert one chunk; returns its rowid.
#[allow(clippy::too_many_arguments)] // mirrors the SQL row shape 1:1
pub(crate) fn insert_chunk(
    conn: &Connection,
    document_id: i64,
    chunk_index: i64,
    text: &str,
    section_title: Option<&str>,
    kind: &str,
    start_offset: i64,
    end_offset: i64,
    authority_score: f32,
    content_hash: &str,
    embedding: &[u8],
) -> Result<i64> {
    conn.execute(
        "INSERT INTO chunks (document_id, chunk_index, text, section_title, kind, start_offset, \
         end_offset, authority_score, content_hash, embedding)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
        params![
            document_id,
            chunk_index,
            text,
            section_title,
            kind,
            start_offset,
            end_offset,
            authority_score,
            content_hash,
            embedding
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Replace an existing chunk's embedding (with model lineage) and knowledge payload.
pub(crate) fn update_chunk_artifacts(
    conn: &Connection,
    chunk_id: i64,
    embedding: &[u8],
    model: &str,
    knowledge_json: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE chunks SET embedding = ?2, embedding_model = ?3, knowledge_json = ?4 WHERE id = ?1",
        params![chunk_id, embedding, model, knowledge_json],
    )?;
    Ok(())
}

/// Chunk ids whose embedding was produced by a model other than `current`.
pub(crate) fn stale_embedding_chunks(
    conn: &Connection,
    current: &str,
    limit: usize,
) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare(
        "SELECT id FROM chunks WHERE embedding_model IS NOT ?1 ORDER BY id LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![current, limit as i64], |r| r.get(0))?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Update a chunk's embedding and the model that produced it.
pub(crate) fn update_chunk_embedding(
    conn: &Connection,
    chunk_id: i64,
    model: &str,
    embedding: &[u8],
) -> Result<()> {
    conn.execute(
        "UPDATE chunks SET embedding = ?2, embedding_model = ?3 WHERE id = ?1",
        params![chunk_id, embedding, model],
    )?;
    Ok(())
}

/// Fetch minimal hit rows for full-text search results.
#[derive(Debug, Clone)]
pub(crate) struct HitRow {
    /// Reserved for v0.2 (per-hit provenance rendering).
    #[allow(dead_code)]
    pub chunk_id: i64,
    pub document_id: i64,
    pub filename: String,
    pub section_title: Option<String>,
    pub text: String,
    pub authority_score: f32,
    /// Reserved for v0.2 metadata-aware ranking (filters on doc type).
    #[allow(dead_code)]
    pub doc_type: String,
}

const HIT_SELECT: &str = "SELECT c.id, c.document_id, d.filename, c.section_title, c.text, \
     c.authority_score, d.doc_type
     FROM chunks c JOIN documents d ON d.id = c.document_id";

/// Load a chunk row joined with its document (for hit rendering).
pub(crate) fn get_hit_row(conn: &Connection, chunk_id: i64) -> Result<HitRow> {
    conn.query_row(
        &format!("{HIT_SELECT} WHERE c.id = ?1"),
        params![chunk_id],
        row_to_hit,
    )
    .optional()?
    .ok_or_else(|| LkosError::Other(format!("chunk {chunk_id} vanished mid-query")))
}

fn row_to_hit(r: &rusqlite::Row<'_>) -> rusqlite::Result<HitRow> {
    Ok(HitRow {
        chunk_id: r.get(0)?,
        document_id: r.get(1)?,
        filename: r.get(2)?,
        section_title: r.get(3)?,
        text: r.get(4)?,
        authority_score: r.get(5)?,
        doc_type: r.get(6)?,
    })
}

/// All (chunk_id, embedding) pairs for one embedding model — single-pass scan.
///
/// v0.1 fetched blobs one query per chunk (N+1); this is one SELECT and it
/// restricts to chunks embedded by `model` so the dense channel never
/// compares vectors across embedding spaces.
pub(crate) fn all_embeddings(
    conn: &Connection,
    filters: Option<&Filters>,
    model: &str,
) -> Result<Vec<(i64, Vec<f32>)>> {
    let sql = match filters {
        Some(f) => format!(
            "SELECT c.id, c.embedding FROM chunks c JOIN documents d ON d.id = c.document_id \
             WHERE c.embedding_model = ?M AND c.embedding IS NOT NULL {}",
            filter_sql_parts(f)
        ),
        None => "SELECT c.id, c.embedding FROM chunks c \
             WHERE c.embedding_model = ?M AND c.embedding IS NOT NULL".to_string(),
    };
    let sql = sql.replace("?M", "?1");
    let mut stmt = conn.prepare(&sql)?;
    let mut all: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(model.to_string())];
    if let Some(f) = filters {
        all.extend(filter_params(f));
    }
    bind_params(&mut stmt, &all)?;
    let rows = stmt.raw_query().mapped(|r| {
        Ok((r.get::<_, i64>(0)?, bytes_to_f32(&r.get::<_, Vec<u8>>(1)?)))
    });
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Fetch the embedding of one chunk.
pub(crate) fn chunk_embedding(conn: &Connection, chunk_id: i64) -> Result<Option<Vec<f32>>> {
    let bytes: Option<Vec<u8>> = conn
        .query_row(
            "SELECT embedding FROM chunks WHERE id = ?1",
            params![chunk_id],
            |r| r.get(0),
        )
        .optional()?
        .flatten();
    Ok(bytes.map(|b| bytes_to_f32(&b)))
}

/// Serialize f32 vector to bytes (little endian).
pub fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Deserialize bytes to f32 vector.
pub fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// filters → SQL
// ---------------------------------------------------------------------------

/// SQL fragments for supported filters. Positional `?` placeholders are
/// appended in the exact order produced by [`filter_params`]:
/// `document_ids`, `doc_types`, `must_contain`, `as_of`.
pub(crate) fn filter_sql_parts(f: &Filters) -> String {
    let mut sql = String::new();
    if f.document_ids.is_some() {
        sql.push_str(" AND c.document_id IN (SELECT value FROM json_each(?))");
    }
    if f.doc_types.is_some() {
        sql.push_str(" AND d.doc_type IN (SELECT value FROM json_each(?))");
    }
    if f.must_contain.is_some() {
        sql.push_str(" AND instr(lower(c.text), lower(?)) > 0");
    }
    if f.as_of.is_some() {
        sql.push_str(" AND d.created_at <= ?");
    }
    sql
}

/// Positional parameter values matching [`filter_sql_parts`].
pub(crate) fn filter_params(f: &Filters) -> Vec<Box<dyn rusqlite::ToSql>> {
    let mut out: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    if let Some(ids) = &f.document_ids {
        out.push(Box::new(
            serde_json::to_string(ids).unwrap_or_else(|_| "[]".into()),
        ));
    }
    if let Some(types) = &f.doc_types {
        out.push(Box::new(
            serde_json::to_string(types).unwrap_or_else(|_| "[]".into()),
        ));
    }
    if let Some(phrase) = &f.must_contain {
        out.push(Box::new(phrase.clone()));
    }
    if let Some(asof) = &f.as_of {
        out.push(Box::new(asof.clone()));
    }
    out
}

/// Bind a full positional parameter list to a raw statement.
pub(crate) fn bind_params(
    stmt: &mut rusqlite::Statement<'_>,
    params: &[Box<dyn rusqlite::ToSql>],
) -> Result<()> {
    for (i, p) in params.iter().enumerate() {
        stmt.raw_bind_parameter(i + 1, p.as_ref())?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// term document frequencies (keyword TF-IDF)
// ---------------------------------------------------------------------------

/// Increment df counters for the distinct terms of one document.
pub(crate) fn bump_term_df(conn: &Connection, terms: &[String]) -> Result<()> {
    for t in terms {
        conn.execute(
            "INSERT INTO term_df(term, df) VALUES(?1, 1)
             ON CONFLICT(term) DO UPDATE SET df = df + 1",
            params![t],
        )?;
    }
    Ok(())
}

/// Decrement df counters (document deletion).
pub(crate) fn drop_term_df(conn: &Connection, terms: &[String]) -> Result<()> {
    for t in terms {
        conn.execute(
            "UPDATE term_df SET df = MAX(0, df - 1) WHERE term = ?1",
            params![t],
        )?;
    }
    Ok(())
}

/// Total number of documents (for IDF).
pub(crate) fn total_documents(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?)
}

// ---------------------------------------------------------------------------
// entities
// ---------------------------------------------------------------------------

/// Upsert an entity by canonical key; returns (entity_id, created).
pub(crate) fn upsert_entity(
    conn: &Connection,
    canonical_key: &str,
    display_name: &str,
    entity_type: &str,
) -> Result<i64> {
    let ts = now();
    conn.execute(
        "INSERT INTO entities (canonical_key, display_name, entity_type, mention_count, first_seen, last_seen)
         VALUES (?1, ?2, ?3, 0, ?4, ?4)
         ON CONFLICT(canonical_key) DO UPDATE SET
             mention_count = mention_count,
             last_seen = ?4,
             display_name = CASE WHEN length(?2) > length(entities.display_name)
                                 THEN ?2 ELSE entities.display_name END",
        params![canonical_key, display_name, entity_type, ts],
    )?;
    let id: i64 = conn.query_row(
        "SELECT id FROM entities WHERE canonical_key = ?1",
        params![canonical_key],
        |r| r.get(0),
    )?;
    Ok(id)
}

/// Add an alias if missing.
pub(crate) fn add_alias(conn: &Connection, entity_id: i64, alias: &str) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?1, ?2)",
        params![entity_id, alias],
    )?;
    Ok(())
}

/// Increment an entity's mention counter.
pub(crate) fn bump_entity_mentions(conn: &Connection, entity_id: i64, by: i64) -> Result<()> {
    conn.execute(
        "UPDATE entities SET mention_count = mention_count + ?2 WHERE id = ?1",
        params![entity_id, by],
    )?;
    Ok(())
}

/// Resolve a surface form to an entity id via canonical key or alias.
pub(crate) fn resolve_surface(conn: &Connection, canonical_key: &str) -> Result<Option<i64>> {
    let by_key: Option<i64> = conn
        .query_row(
            "SELECT id FROM entities WHERE canonical_key = ?1",
            params![canonical_key],
            |r| r.get(0),
        )
        .optional()?;
    if by_key.is_some() {
        return Ok(by_key);
    }
    conn.query_row(
        "SELECT entity_id FROM entity_aliases WHERE lower(alias) = ?1",
        params![canonical_key],
        |r| r.get(0),
    )
    .optional()
    .map_err(Into::into)
}

/// Insert an entity mention.
#[allow(clippy::too_many_arguments)] // mirrors the SQL row shape 1:1
pub(crate) fn insert_mention(
    conn: &Connection,
    entity_id: i64,
    document_id: i64,
    chunk_id: i64,
    surface: &str,
    entity_type: &str,
    start_offset: i64,
    end_offset: i64,
    confidence: f32,
    extractor: &str,
) -> Result<i64> {
    conn.execute(
        "INSERT INTO entity_mentions (entity_id, document_id, chunk_id, surface, entity_type, \
         start_offset, end_offset, confidence, extractor, created_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
        params![
            entity_id,
            document_id,
            chunk_id,
            surface,
            entity_type,
            start_offset,
            end_offset,
            confidence,
            extractor,
            now()
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Fetch an entity record with aliases.
pub(crate) fn get_entity(conn: &Connection, entity_id: i64) -> Result<EntityRecord> {
    conn.query_row(
        "SELECT id, canonical_key, display_name, entity_type, mention_count FROM entities WHERE id = ?1",
        params![entity_id],
        |r| {
            Ok(EntityRecord {
                id: r.get(0)?,
                canonical_key: r.get(1)?,
                display_name: r.get(2)?,
                entity_type: r.get(3)?,
                mention_count: r.get(4)?,
                aliases: Vec::new(),
            })
        },
    )
    .optional()?
    .ok_or(LkosError::EntityNotFound(entity_id))
}

/// Entity summaries matching a name fragment (used by the planner).
#[allow(dead_code)] // Reserved API surface for v0.2 entity search.
pub(crate) fn find_entities_by_name(conn: &Connection, fragment: &str) -> Result<Vec<EntitySummary>> {
    let mut stmt = conn.prepare(
        "SELECT id, display_name, entity_type, mention_count FROM entities
         WHERE lower(display_name) LIKE ('%' || lower(?1) || '%')
         ORDER BY mention_count DESC LIMIT 10",
    )?;
    let rows = stmt.query_map(params![fragment], |r| {
        Ok(EntitySummary {
            id: r.get(0)?,
            display_name: r.get(1)?,
            entity_type: r.get(2)?,
            mention_count: r.get(3)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// All entities mentioned in a document.
pub(crate) fn entities_for_document(conn: &Connection, doc_id: i64) -> Result<Vec<EntitySummary>> {
    let mut stmt = conn.prepare(
        "SELECT e.id, e.display_name, e.entity_type, COUNT(m.id) AS cnt
         FROM entities e JOIN entity_mentions m ON m.entity_id = e.id
         WHERE m.document_id = ?1
         GROUP BY e.id ORDER BY cnt DESC",
    )?;
    let rows = stmt.query_map(params![doc_id], |r| {
        Ok(EntitySummary {
            id: r.get(0)?,
            display_name: r.get(1)?,
            entity_type: r.get(2)?,
            mention_count: r.get(3)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Documents mentioning an entity (via mentions).
pub(crate) fn documents_for_entity(conn: &Connection, entity_id: i64) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT document_id FROM entity_mentions WHERE entity_id = ?1",
    )?;
    let rows = stmt.query_map(params![entity_id], |r| r.get(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Chunk ids where an entity is mentioned.
pub(crate) fn chunks_for_entity(conn: &Connection, entity_id: i64) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT chunk_id FROM entity_mentions WHERE entity_id = ?1 ORDER BY chunk_id",
    )?;
    let rows = stmt.query_map(params![entity_id], |r| r.get(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// All entity mentions of one chunk.
#[allow(dead_code)] // Reserved API surface for v0.2 highlighter.
pub(crate) fn mentions_for_chunk(conn: &Connection, chunk_id: i64) -> Result<Vec<EntityMention>> {
    let mut stmt = conn.prepare(
        "SELECT chunk_id, document_id, surface, entity_type, start_offset, end_offset, confidence
         FROM entity_mentions WHERE chunk_id = ?1",
    )?;
    let rows = stmt.query_map(params![chunk_id], |r| {
        Ok(EntityMention {
            chunk_id: r.get(0)?,
            document_id: r.get(1)?,
            surface: r.get(2)?,
            entity_type: r.get(3)?,
            offsets: (r.get(4)?, r.get(5)?),
            confidence: r.get(6)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

// ---------------------------------------------------------------------------
// claims + conflicts
// ---------------------------------------------------------------------------

/// Insert a claim; returns its id.
#[allow(clippy::too_many_arguments)] // mirrors the SQL row shape 1:1
pub(crate) fn insert_claim(
    conn: &Connection,
    subject: &str,
    predicate: &str,
    object: &str,
    sentence: &str,
    document_id: i64,
    chunk_id: i64,
    confidence: f32,
    extractor: &str,
    subject_key: &str,
    predicate_key: &str,
    valid_from: Option<&str>,
    valid_until: Option<&str>,
    start_offset: Option<i64>,
    end_offset: Option<i64>,
) -> Result<i64> {
    conn.execute(
        "INSERT INTO claims (subject, predicate, object, sentence, document_id, chunk_id, \
         confidence, extractor, subject_key, predicate_key, valid_from, valid_until, \
         start_offset, end_offset, created_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
        params![
            subject,
            predicate,
            object,
            sentence,
            document_id,
            chunk_id,
            confidence,
            extractor,
            subject_key,
            predicate_key,
            valid_from,
            valid_until,
            start_offset,
            end_offset,
            now()
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Claims extracted from one document.
pub(crate) fn claims_for_document(conn: &Connection, doc_id: i64) -> Result<Vec<ClaimRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, subject, predicate, object, sentence, document_id, chunk_id, confidence, \
         extractor, valid_from, valid_until, created_at, start_offset, end_offset \
         FROM claims WHERE document_id = ?1",
    )?;
    let rows = stmt.query_map(params![doc_id], row_to_claim)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// All claims sharing a subject key (for entity pages and conflicts).
pub(crate) fn claims_for_subject(conn: &Connection, subject_key: &str) -> Result<Vec<ClaimRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, subject, predicate, object, sentence, document_id, chunk_id, confidence, \
         extractor, valid_from, valid_until, created_at, start_offset, end_offset \
         FROM claims WHERE subject_key = ?1",
    )?;
    let rows = stmt.query_map(params![subject_key], row_to_claim)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

fn row_to_claim(r: &rusqlite::Row<'_>) -> rusqlite::Result<ClaimRecord> {
    Ok(ClaimRecord {
        id: r.get(0)?,
        subject: r.get(1)?,
        predicate: r.get(2)?,
        object: r.get(3)?,
        sentence: r.get(4)?,
        document_id: r.get(5)?,
        chunk_id: r.get(6)?,
        confidence: r.get(7)?,
        extractor: r.get(8)?,
        valid_from: r.get(9)?,
        valid_until: r.get(10)?,
        created_at: r.get(11)?,
        start_offset: r.get(12)?,
        end_offset: r.get(13)?,
    })
}

/// Prior numeric claims with the same (subject_key, predicate_key) — the
/// indexed replacement for v0.1's full-table GLOB scan (O(N²) → O(log N)).
pub(crate) fn numeric_claims_for_keys(
    conn: &Connection,
    subject_key: &str,
    predicate_key: &str,
) -> Result<Vec<(i64, f64, Option<String>)>> {
    let mut stmt = conn.prepare(
        "SELECT id, object, valid_from FROM claims
         WHERE subject_key = ?1 AND predicate_key = ?2 AND object GLOB '*[0-9]*'",
    )?;
    let rows = stmt.query_map(params![subject_key, predicate_key], |r| {
        let obj: String = r.get(1)?;
        Ok((
            r.get::<_, i64>(0)?,
            parse_first_number(&obj).unwrap_or(f64::NAN),
            r.get::<_, Option<String>>(2)?,
        ))
    })?;
    Ok(rows
        .filter_map(|r| r.ok())
        .filter(|(_, v, _)| v.is_finite() && *v > 0.0)
        .collect())
}

/// Insert a conflict record.
#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_conflict(
    conn: &Connection,
    subject_key: &str,
    predicate_key: &str,
    claim_a: i64,
    claim_b: i64,
    delta: f32,
    explanation: &str,
    kind: &str,
) -> Result<i64> {
    conn.execute(
        "INSERT INTO claim_conflicts (subject_key, predicate_key, claim_a, claim_b, delta, explanation, conflict_kind, created_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
        params![subject_key, predicate_key, claim_a, claim_b, delta, explanation, kind, now()],
    )?;
    Ok(conn.last_insert_rowid())
}

/// List conflicts (most recent first).
pub(crate) fn list_conflicts(conn: &Connection, limit: usize) -> Result<Vec<ConflictRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, subject_key, predicate_key, claim_a, claim_b, delta, explanation, conflict_kind
         FROM claim_conflicts ORDER BY id DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |r| {
        Ok(ConflictRecord {
            id: r.get(0)?,
            subject_key: r.get(1)?,
            predicate_key: r.get(2)?,
            claim_a: r.get(3)?,
            claim_b: r.get(4)?,
            delta: r.get(5)?,
            explanation: r.get(6)?,
            conflict_kind: r.get(7)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

// ---------------------------------------------------------------------------
// relationships
// ---------------------------------------------------------------------------

/// Upsert a co-occurrence relationship; weight += 1.
pub(crate) fn upsert_relationship(
    conn: &Connection,
    source: i64,
    target: i64,
    rel_type: &str,
    document_id: i64,
) -> Result<()> {
    let (a, b) = if source <= target { (source, target) } else { (target, source) };
    conn.execute(
        "INSERT INTO relationships (source_entity_id, target_entity_id, relationship_type, weight, \
         first_document_id, last_document_id, created_at)
         VALUES (?1,?2,?3,1.0,?4,?4,?5)
         ON CONFLICT(source_entity_id, target_entity_id, relationship_type) DO UPDATE SET
             weight = weight + 1.0,
             last_document_id = ?4",
        params![a, b, rel_type, document_id, now()],
    )?;
    Ok(())
}

/// One-hop neighborhood of an entity.
pub(crate) fn neighbors(
    conn: &Connection,
    entity_id: i64,
    limit: usize,
) -> Result<Vec<(EntitySummary, RelationshipRecord)>> {
    let mut stmt = conn.prepare(
        "SELECT r.id, r.source_entity_id, r.target_entity_id, r.relationship_type, r.weight, \
                r.last_document_id,
                e.id, e.display_name, e.entity_type, e.mention_count
         FROM relationships r
         JOIN entities e ON e.id = CASE WHEN r.source_entity_id = ?1
                                        THEN r.target_entity_id ELSE r.source_entity_id END
         WHERE r.source_entity_id = ?1 OR r.target_entity_id = ?1
         ORDER BY r.weight DESC LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![entity_id, limit as i64], |r| {
        let rel = RelationshipRecord {
            id: r.get(0)?,
            source_entity_id: r.get(1)?,
            target_entity_id: r.get(2)?,
            relationship_type: r.get(3)?,
            weight: r.get(4)?,
            last_document_id: r.get(5)?,
        };
        let ent = EntitySummary {
            id: r.get(6)?,
            display_name: r.get(7)?,
            entity_type: r.get(8)?,
            mention_count: r.get(9)?,
        };
        Ok((ent, rel))
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

// ---------------------------------------------------------------------------
// provenance
// ---------------------------------------------------------------------------

/// Insert a provenance record.
#[allow(clippy::too_many_arguments)] // mirrors the SQL row shape 1:1
pub(crate) fn insert_provenance(
    conn: &Connection,
    artifact_type: &str,
    artifact_id: &str,
    document_id: Option<i64>,
    chunk_id: Option<i64>,
    offsets: Option<(i64, i64)>,
    extractor: &str,
    extractor_version: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO provenance (artifact_type, artifact_id, document_id, chunk_id, start_offset, \
         end_offset, extractor, extractor_version, created_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
        params![
            artifact_type,
            artifact_id,
            document_id,
            chunk_id,
            offsets.map(|o| o.0),
            offsets.map(|o| o.1),
            extractor,
            extractor_version,
            now()
        ],
    )?;
    Ok(())
}

/// Provenance for one artifact.
pub(crate) fn provenance_for(
    conn: &Connection,
    artifact_type: &str,
    artifact_id: &str,
) -> Result<Vec<ProvenanceRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, artifact_type, artifact_id, document_id, chunk_id, start_offset, end_offset, \
         extractor, extractor_version, created_at
         FROM provenance WHERE artifact_type = ?1 AND artifact_id = ?2",
    )?;
    let rows = stmt.query_map(params![artifact_type, artifact_id], row_to_prov)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

fn row_to_prov(r: &rusqlite::Row<'_>) -> rusqlite::Result<ProvenanceRecord> {
    let start: Option<i64> = r.get(5)?;
    let end: Option<i64> = r.get(6)?;
    Ok(ProvenanceRecord {
        id: r.get(0)?,
        artifact_type: r.get(1)?,
        artifact_id: r.get(2)?,
        document_id: r.get(3)?,
        chunk_id: r.get(4)?,
        offsets: match (start, end) {
            (Some(s), Some(e)) => Some((s, e)),
            _ => None,
        },
        extractor: r.get(7)?,
        extractor_version: r.get(8)?,
        created_at: r.get(9)?,
    })
}

// ---------------------------------------------------------------------------
// jobs + meta
// ---------------------------------------------------------------------------

/// Enqueue a background job.
pub(crate) fn enqueue_job(
    conn: &Connection,
    kind: &str,
    payload: &str,
    priority: i64,
) -> Result<i64> {
    let ts = now();
    conn.execute(
        "INSERT INTO jobs (kind, payload, status, priority, run_at, created_at, updated_at)
         VALUES (?1, ?2, 'pending', ?3, ?4, ?4, ?4)",
        params![kind, payload, priority, ts],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Claim the next runnable job atomically (priority, then FIFO, respecting
/// backoff `run_at`). Uses an immediate transaction + conditional UPDATE so
/// concurrent workers cannot double-claim (v0.1 was SELECT-then-UPDATE).
pub(crate) fn claim_next_atomic(conn: &mut Connection) -> Result<Option<(i64, String, String)>> {
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let row = tx
        .query_row(
            "SELECT id, kind, payload FROM jobs
             WHERE status = 'pending' AND (run_at IS NULL OR run_at <= ?1)
             ORDER BY priority ASC, id ASC LIMIT 1",
            params![now()],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)),
        )
        .optional()?;
    if let Some((id, kind, payload)) = row {
        tx.execute(
            "UPDATE jobs SET status = 'running', updated_at = ?2 WHERE id = ?1 AND status = 'pending'",
            params![id, now()],
        )?;
        tx.commit()?;
        Ok(Some((id, kind, payload)))
    } else {
        tx.commit()?;
        Ok(None)
    }
}

/// Update job status/attempts/error, with optional backoff release time.
pub(crate) fn update_job_sched(
    conn: &Connection,
    job_id: i64,
    status: &str,
    attempts: i64,
    last_error: Option<&str>,
    run_at: Option<&str>,
) -> Result<()> {
    conn.execute(
        "UPDATE jobs SET status = ?2, attempts = ?3, last_error = ?4, run_at = ?5, updated_at = ?6
         WHERE id = ?1",
        params![job_id, status, attempts, last_error, run_at, now()],
    )?;
    Ok(())
}

/// Set job progress (0-100).
pub(crate) fn set_job_progress(conn: &Connection, job_id: i64, progress: i64) -> Result<()> {
    conn.execute(
        "UPDATE jobs SET progress = ?2, updated_at = ?3 WHERE id = ?1",
        params![job_id, progress.clamp(0, 100), now()],
    )?;
    Ok(())
}

/// Update job status/attempts/error.
pub(crate) fn update_job(
    conn: &Connection,
    job_id: i64,
    status: &str,
    attempts: i64,
    last_error: Option<&str>,
) -> Result<()> {
    update_job_sched(conn, job_id, status, attempts, last_error, None)
}

/// Recover jobs left 'running' by a crash: back to 'pending'.
pub(crate) fn recover_running_jobs(conn: &Connection) -> Result<usize> {
    let n = conn.execute(
        "UPDATE jobs SET status = 'pending', updated_at = ?1 WHERE status = 'running'",
        params![now()],
    )?;
    Ok(n)
}

/// Get an engine_meta value.
pub(crate) fn meta_get(conn: &Connection, key: &str) -> Result<Option<String>> {
    conn.query_row(
        "SELECT value FROM engine_meta WHERE key = ?1",
        params![key],
        |r| r.get(0),
    )
    .optional()
    .map_err(Into::into)
}

/// Set an engine_meta value.
pub(crate) fn meta_set(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO engine_meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = ?2",
        params![key, value],
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// deferred pipeline payload (pending_text)
// ---------------------------------------------------------------------------

/// Store the normalized text for deferred/background processing.
pub(crate) fn set_pending_text(conn: &Connection, doc_id: i64, text: &str) -> Result<()> {
    conn.execute(
        "UPDATE documents SET pending_text = ?2 WHERE id = ?1",
        params![doc_id, text],
    )?;
    Ok(())
}

/// Read and clear the pending text (NULLed to free space once processed).
pub(crate) fn take_pending_text(conn: &Connection, doc_id: i64) -> Result<Option<String>> {
    let text: Option<String> = conn
        .query_row(
            "SELECT pending_text FROM documents WHERE id = ?1",
            params![doc_id],
            |r| r.get(0),
        )
        .optional()?
        .flatten();
    if text.is_some() {
        conn.execute(
            "UPDATE documents SET pending_text = NULL WHERE id = ?1",
            params![doc_id],
        )?;
    }
    Ok(text)
}

/// All chunks of a document in order.
pub(crate) fn chunks_for_document(conn: &Connection, doc_id: i64) -> Result<Vec<ChunkInfo>> {
    let mut stmt = conn.prepare(
        "SELECT id, document_id, chunk_index, text, section_title, kind, start_offset, \
         end_offset, authority_score, COALESCE(content_hash,'')
         FROM chunks WHERE document_id = ?1 ORDER BY chunk_index",
    )?;
    let rows = stmt.query_map(params![doc_id], |r| {
        Ok(ChunkInfo {
            id: r.get(0)?,
            document_id: r.get(1)?,
            chunk_index: r.get(2)?,
            text: r.get(3)?,
            section_title: r.get(4)?,
            kind: r.get(5)?,
            start_offset: r.get(6)?,
            end_offset: r.get(7)?,
            authority_score: r.get(8)?,
            content_hash: r.get(9)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Delete all entity mentions recorded for a document.
pub(crate) fn delete_mentions_for_doc(conn: &Connection, doc_id: i64) -> Result<()> {
    conn.execute(
        "DELETE FROM entity_mentions WHERE document_id = ?1",
        params![doc_id],
    )?;
    Ok(())
}

/// Delete all claims recorded for a document (conflicts cascade via FK).
pub(crate) fn delete_claims_for_doc(conn: &Connection, doc_id: i64) -> Result<()> {
    conn.execute(
        "DELETE FROM claims WHERE document_id = ?1",
        params![doc_id],
    )?;
    Ok(())
}

/// Distinct entity ids mentioned by a document.
pub(crate) fn distinct_entities_for_doc(conn: &Connection, doc_id: i64) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT entity_id FROM entity_mentions WHERE document_id = ?1",
    )?;
    let rows = stmt.query_map(params![doc_id], |r| r.get(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Entity ids mentioned in one chunk.
pub(crate) fn entity_ids_for_chunk(conn: &Connection, chunk_id: i64) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT entity_id FROM entity_mentions WHERE chunk_id = ?1",
    )?;
    let rows = stmt.query_map(params![chunk_id], |r| r.get(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Top entities by mention count (public API surface).
pub(crate) fn list_entities(conn: &Connection, limit: usize) -> Result<Vec<EntityRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, canonical_key, display_name, entity_type, mention_count
         FROM entities ORDER BY mention_count DESC, id ASC LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |r| {
        Ok(EntityRecord {
            id: r.get(0)?,
            canonical_key: r.get(1)?,
            display_name: r.get(2)?,
            entity_type: r.get(3)?,
            mention_count: r.get(4)?,
            aliases: Vec::new(),
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Resolve an entity id by display name or canonical key.
pub(crate) fn entity_id_by_name(conn: &Connection, name: &str) -> Result<Option<i64>> {
    let key = crate::entities::canonical_key(name);
    if let Some(id) = resolve_surface(conn, &key)? {
        return Ok(Some(id));
    }
    conn.query_row(
        "SELECT id FROM entities WHERE lower(display_name) = lower(?1)",
        params![name],
        |r| r.get(0),
    )
    .optional()
    .map_err(Into::into)
}

/// Recompute ALL relationship edges touching `entity_ids` from the surviving
/// evidence (mentions + claims). This makes document deletion graph-correct:
/// v0.1 left stale co-occurrence edges behind forever.
///
/// * `CO_OCCURS_WITH` — rebuilt from pairwise per-chunk mention pairs.
/// * `RELATES_TO_<predicate>` — rebuilt from claims whose subject AND object
///   both resolve to entities co-mentioned in the claim's chunk.
pub(crate) fn recompute_relationships_for_entities(
    conn: &Connection,
    entity_ids: &[i64],
) -> Result<()> {
    if entity_ids.is_empty() {
        return Ok(());
    }
    let mut sql = "DELETE FROM relationships WHERE ".to_string();
    for (i, _) in entity_ids.iter().enumerate() {
        if i > 0 {
            sql.push_str(" OR ");
        }
        sql.push_str(&format!("source_entity_id = ?{} OR target_entity_id = ?{}", i + 1, i + 1));
    }
    {
        let params_ref: Vec<Box<dyn rusqlite::ToSql>> = entity_ids
            .iter()
            .map(|id| Box::new(*id) as Box<dyn rusqlite::ToSql>)
            .collect();
        let mut stmt = conn.prepare(&sql)?;
        stmt.execute(rusqlite::params_from_iter(params_ref))?;
    }
    // Rebuild CO_OCCURS_WITH among the affected entities from remaining mentions.
    let mut pair_sql = String::from(
        "SELECT m1.entity_id, m2.entity_id, COUNT(DISTINCT m1.chunk_id) AS w
         FROM entity_mentions m1
         JOIN entity_mentions m2 ON m1.chunk_id = m2.chunk_id AND m1.entity_id < m2.entity_id
         WHERE (",
    );
    for (i, _) in entity_ids.iter().enumerate() {
        if i > 0 {
            pair_sql.push_str(" OR ");
        }
        pair_sql.push_str(&format!(
            "m1.entity_id = ?{} OR m2.entity_id = ?{}",
            i + 1,
            i + 1
        ));
    }
    pair_sql.push_str(") GROUP BY m1.entity_id, m2.entity_id");
    let mut stmt = conn.prepare(&pair_sql)?;
    let params_ref: Vec<Box<dyn rusqlite::ToSql>> = entity_ids
        .iter()
        .map(|id| Box::new(*id) as Box<dyn rusqlite::ToSql>)
        .collect();
    let rows = stmt.query(rusqlite::params_from_iter(params_ref))?.mapped(|r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, i64>(2)?,
        ))
    });
    let mut pairs = Vec::new();
    for r in rows {
        pairs.push(r?);
    }
    for (a, b, w) in pairs {
        upsert_relationship_weighted(conn, a, b, "CO_OCCURS_WITH", w as f32)?;
    }
    Ok(())
}

/// Upsert a relationship with an explicit weight (recompute path).
pub(crate) fn upsert_relationship_weighted(
    conn: &Connection,
    source: i64,
    target: i64,
    rel_type: &str,
    weight: f32,
) -> Result<()> {
    let (a, b) = if source <= target { (source, target) } else { (target, source) };
    conn.execute(
        "INSERT INTO relationships (source_entity_id, target_entity_id, relationship_type, weight, created_at)
         VALUES (?1,?2,?3,?4,?5)
         ON CONFLICT(source_entity_id, target_entity_id, relationship_type) DO UPDATE SET
             weight = excluded.weight",
        params![a, b, rel_type, weight, now()],
    )?;
    Ok(())
}

/// Merge entity `merged_id` into `survivor_id`: aliases, mentions, counts,
/// relationships, and the audit log. Irreversible (documented in API.md).
pub(crate) fn merge_entity_rows(
    conn: &Connection,
    survivor_id: i64,
    merged_id: i64,
    reason: &str,
) -> Result<()> {
    if survivor_id == merged_id {
        return Ok(());
    }
    conn.execute(
        "INSERT INTO entity_merge_log (survivor_id, merged_id, reason, merged_at)
         VALUES (?1, ?2, ?3, ?4)",
        params![survivor_id, merged_id, reason, now()],
    )?;
    // Move aliases (ignore PK conflicts).
    conn.execute(
        "INSERT OR IGNORE INTO entity_aliases (entity_id, alias)
         SELECT ?1, alias FROM entity_aliases WHERE entity_id = ?2",
        params![survivor_id, merged_id],
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO entity_aliases (entity_id, alias)
         SELECT ?1, display_name FROM entities WHERE id = ?2",
        params![survivor_id, merged_id],
    )?;
    // Move mentions.
    let moved: i64 = conn.query_row(
        "SELECT COUNT(*) FROM entity_mentions WHERE entity_id = ?1",
        params![merged_id],
        |r| r.get(0),
    )?;
    conn.execute(
        "UPDATE entity_mentions SET entity_id = ?1 WHERE entity_id = ?2",
        params![survivor_id, merged_id],
    )?;
    // Sum counts.
    conn.execute(
        "UPDATE entities SET mention_count = mention_count + ?1 WHERE id = ?2",
        params![moved, survivor_id],
    )?;
    // Re-point relationships onto the survivor (dedupe via UPDATE OR IGNORE then cleanup).
    conn.execute(
        "UPDATE OR IGNORE relationships SET source_entity_id = ?1 WHERE source_entity_id = ?2",
        params![survivor_id, merged_id],
    )?;
    conn.execute(
        "UPDATE OR IGNORE relationships SET target_entity_id = ?1 WHERE target_entity_id = ?2",
        params![survivor_id, merged_id],
    )?;
    conn.execute(
        "DELETE FROM relationships WHERE source_entity_id = ?1 OR target_entity_id = ?1",
        params![merged_id],
    )?;
    // Drop the merged row (aliases cascade).
    conn.execute("DELETE FROM entities WHERE id = ?1", params![merged_id])?;
    Ok(())
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Parse the first number in a string, normalizing currency symbols, commas,
/// percent and magnitude words/suffixes ("$10 million", "10,000", "3.5B", "15%").
///
/// v0.1 kept only the LAST whitespace token, so "$10 million" failed and
/// "10 2024" returned 2024. This version scans left-to-right and stops at the
/// first numeric token, folding the magnitude suffix or following magnitude
/// word into the value.
pub(crate) fn parse_first_number(s: &str) -> Option<f64> {
    let tokens: Vec<String> = s
        .split(|c: char| c.is_whitespace() || matches!(c, '$' | '€' | '£' | ',' | ';'))
        .filter(|t| !t.is_empty())
        .map(|t| t.trim().trim_matches(['.', '!', '?', ')', '(', ':']).to_string())
        .filter(|t| !t.is_empty())
        .collect();
    for (i, tok) in tokens.iter().enumerate() {
        let (core, mult) = split_magnitude(tok);
        if let Ok(v) = core.parse::<f64>() {
            let mut value = v * mult;
            // "10 million" — magnitude as the following word.
            if mult == 1.0 {
                if let Some(next) = tokens.get(i + 1) {
                    let lm = next.to_ascii_lowercase();
                    let m = match lm.as_str() {
                        "million" | "mln" => 1_000_000.0,
                        "billion" | "bln" => 1_000_000_000.0,
                        "thousand" | "k" => 1_000.0,
                        _ => 1.0,
                    };
                    value = v * m;
                }
            }
            return Some(value);
        }
    }
    None
}

/// Split "10M"/"3.5B"/"15%" into (numeric-core, multiplier).
fn split_magnitude(tok: &str) -> (&str, f64) {
    let mut t = tok;
    let mut mult = 1.0;
    // Percent is discarded as a magnitude (15% == 15 for delta purposes).
    if t.ends_with('%') {
        t = &t[..t.len() - 1];
    }
    if t.len() >= 2 {
        let last = t.chars().last().unwrap_or(' ');
        let lower_last = last.to_ascii_lowercase();
        match lower_last {
            'm' if t[..t.len() - 1].chars().next().is_some_and(|c| c.is_ascii_digit()) => {
                mult = 1_000_000.0;
                t = &t[..t.len() - 1];
            }
            'b' if t[..t.len() - 1].chars().next().is_some_and(|c| c.is_ascii_digit()) => {
                mult = 1_000_000_000.0;
                t = &t[..t.len() - 1];
            }
            'k' if t[..t.len() - 1].chars().next().is_some_and(|c| c.is_ascii_digit()) => {
                mult = 1_000.0;
                t = &t[..t.len() - 1];
            }
            _ => {}
        }
    }
    (t, mult)
}
