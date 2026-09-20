//! Entity resolution and canonicalization.
//!
//! Deterministic two-phase approach (see `docs/ENTITIES.md`):
//! 1. **Canonicalization**: surface -> normalized key (lowercase, strip legal
//!    suffixes and punctuation, collapse whitespace).
//! 2. **Alias merging**: surfaces sharing a canonical key map to one entity;
//!    the longest/most frequent surface becomes the display name. User-supplied
//!    aliases can be added via the engine API and are respected from then on.
//!
//! Scope note: v0.1 deliberately does NOT do embedding-based or LLM-based
//! linking — it is conservative and labeled (`confidence` values are heuristic).
//! See ADR-006 for the reasoning and the 0.4 roadmap.

use crate::error::Result;
use crate::storage::dao;
use rusqlite::Connection;

/// Legal-suffix and noise tokens stripped during canonicalization.
const LEGAL_SUFFIXES: &[&str] = &[
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "llc", "llp", "gmbh", "ag",
    "plc", "co", "company", "group", "holdings", "sa", "sas", "bv", "oy",
];

/// Canonical key for a surface form.
pub fn canonical_key(surface: &str) -> String {
    let lower = surface.to_lowercase();
    let mut words: Vec<String> = Vec::new();
    for w in lower.split(|c: char| !c.is_alphanumeric()) {
        if w.is_empty() {
            continue;
        }
        if LEGAL_SUFFIXES.contains(&w) && !words.is_empty() {
            continue;
        }
        words.push(w.to_string());
    }
    if words.is_empty() {
        lower
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
    } else {
        words.join(" ")
    }
}

/// Persist entity candidates from a KnowledgeObject scan of one chunk.
///
/// Returns the list of (entity_id, candidate) that were recorded.
pub fn persist_entities(
    conn: &Connection,
    document_id: i64,
    chunk_id: i64,
    candidates: &[crate::knowledge::EntityCandidate],
) -> Result<Vec<(i64, crate::knowledge::EntityCandidate)>> {
    let mut out = Vec::new();
    for cand in candidates {
        // Skip weak types for graph indexing (keep in KO payload though).
        if cand.entity_type == crate::knowledge::T_URL || cand.entity_type == crate::knowledge::T_PERCENT {
            continue;
        }
        let key = canonical_key(&cand.surface);
        if key.len() < 2 {
            continue;
        }
        let entity_id = dao::upsert_entity(conn, &key, &cand.surface, &cand.entity_type)?;
        dao::add_alias(conn, entity_id, &cand.surface)?;
        dao::insert_mention(
            conn,
            entity_id,
            document_id,
            chunk_id,
            &cand.surface,
            &cand.entity_type,
            cand.start as i64,
            cand.end as i64,
            cand.confidence,
            crate::knowledge::ENTITY_EXTRACTOR,
        )?;
        dao::insert_provenance(
            conn,
            "entity_mention",
            &format!("{chunk_id}:{key}"),
            Some(document_id),
            Some(chunk_id),
            Some((cand.start as i64, cand.end as i64)),
            crate::knowledge::ENTITY_EXTRACTOR,
            crate::knowledge::KNOWLEDGE_VERSION,
        )?;
        out.push((entity_id, cand.clone()));
    }
    // One mention-count bump per (entity, chunk) — mention_count counts chunks
    // mentioning the entity, not raw regex hits (stabilizes ranking).
    let mut seen: Vec<i64> = Vec::new();
    for (id, _) in &out {
        if !seen.contains(id) {
            seen.push(*id);
            dao::bump_entity_mentions(conn, *id, 1)?;
        }
    }
    Ok(out)
}

/// Build co-occurrence relationships among entities mentioned in the same chunk.
pub fn link_cooccurrences(
    conn: &Connection,
    document_id: i64,
    mentioned: &[(i64, crate::knowledge::EntityCandidate)],
) -> Result<usize> {
    let mut ids: Vec<i64> = mentioned.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    ids.dedup();
    let mut links = 0usize;
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            dao::upsert_relationship(conn, ids[i], ids[j], "CO_OCCURS_WITH", document_id)?;
            links += 1;
        }
    }
    Ok(links)
}
