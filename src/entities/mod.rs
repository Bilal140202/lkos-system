//! Entity resolution and canonicalization (multi-stage).
//!
//! Resolution pipeline (deterministic, in priority order):
//! 1. **Exact canonical key** — lowercase, strip legal suffixes/punctuation.
//! 2. **Alias lookup** — user-supplied or previously seen surface forms
//!    (`entity_aliases`), including the public [`crate::Lkos::add_entity_alias`]
//!    API (v0.1 documented an alias API that did not exist; it does now).
//! 3. **Type-guarded fuzzy match** — Jaro–Winkler ≥ 0.93 on canonical keys
//!    with the same entity type, using a first-letter blocking pass. False
//!    merges poison a whole graph, so the threshold is deliberately strict
//!    and the match is recorded with reduced confidence.
//!
//! Scope note: embedding-based or LLM-based linking remains out of scope
//! (see NON_GOALS.md) — every stage here is inspectable and reversible
//! (merges write to `entity_merge_log`).

use crate::error::Result;
use crate::storage::dao;
use rusqlite::{Connection, OptionalExtension};

/// Legal-suffix and noise tokens stripped during canonicalization.
const LEGAL_SUFFIXES: &[&str] = &[
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "llc", "llp", "gmbh", "ag",
    "plc", "co", "company", "group", "holdings", "sa", "sas", "bv", "oy",
];

/// Jaro–Winkler similarity threshold for fuzzy entity linking.
pub const FUZZY_LINK_THRESHOLD: f64 = 0.93;

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

/// Jaro–Winkler similarity in [0, 1] (Winkler's 1989 refinement of Jaro;
/// prefix scale 0.1, max prefix 4 — the standard parameterization).
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    let j = jaro(a, b);
    if j <= 0.7 {
        return j;
    }
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    let prefix = ac
        .iter()
        .zip(bc.iter())
        .take(4)
        .take_while(|(x, y)| x == y)
        .count() as f64;
    (j + 0.1 * prefix * (1.0 - j)).min(1.0)
}

fn jaro(a: &str, b: &str) -> f64 {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    if ac.is_empty() && bc.is_empty() {
        return 1.0;
    }
    if ac.is_empty() || bc.is_empty() {
        return 0.0;
    }
    if ac == bc {
        return 1.0;
    }
    let window = (ac.len().max(bc.len()) / 2).saturating_sub(1);
    let mut a_matched = vec![false; ac.len()];
    let mut b_matched = vec![false; bc.len()];
    let mut matches = 0usize;
    for (i, ch) in ac.iter().enumerate() {
        let lo = i.saturating_sub(window);
        let hi = (i + window + 1).min(bc.len());
        for k in lo..hi {
            if !b_matched[k] && bc[k] == *ch {
                a_matched[i] = true;
                b_matched[k] = true;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }
    // Half transpositions.
    let mut transpositions = 0usize;
    let mut k = 0usize;
    for (i, m) in a_matched.iter().enumerate() {
        if *m {
            while !b_matched[k] {
                k += 1;
            }
            if ac[i] != bc[k] {
                transpositions += 1;
            }
            k += 1;
        }
    }
    let m = matches as f64;
    (m / ac.len() as f64 + m / bc.len() as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0
}

/// Resolve a candidate surface to an entity id via the multi-stage pipeline.
/// `entity_type` guards fuzzy linking (never merge a person with an org).
pub(crate) fn resolve_candidate(
    conn: &Connection,
    surface: &str,
    entity_type: &str,
) -> Result<Option<(i64, f32)>> {
    let key = canonical_key(surface);
    // Stage 1: exact canonical key.
    if let Some(id) = dao::resolve_surface(conn, &key)? {
        return Ok(Some((id, 1.0)));
    }
    // Stage 2: alias table (exact, case-insensitive).
    let alias_hit: Option<i64> = conn
        .query_row(
            "SELECT entity_id FROM entity_aliases WHERE lower(alias) = lower(?1)",
            rusqlite::params![surface],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(id) = alias_hit {
        return Ok(Some((id, 0.95)));
    }
    // Stage 3: fuzzy match within the same type, blocked by first letter.
    let first = key.chars().next().unwrap_or('\0');
    if first != '\0' && key.len() >= 3 {
        let mut stmt = conn.prepare(
            "SELECT id, canonical_key FROM entities
             WHERE entity_type = ?1 AND canonical_key LIKE ?2
             LIMIT 256",
        )?;
        let pattern = format!("{first}%");
        let rows = stmt.query_map(rusqlite::params![entity_type, pattern], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?;
        let mut best: Option<(i64, f64)> = None;
        for row in rows.filter_map(|r| r.ok()) {
            let sim = jaro_winkler(&key, &row.1);
            if sim >= FUZZY_LINK_THRESHOLD && best.as_ref().map(|b| sim > b.1).unwrap_or(true) {
                best = Some((row.0, sim));
            }
        }
        if let Some((id, sim)) = best {
            // Deterministic tie-break handled by >= strictness above.
            return Ok(Some((id, sim as f32 * 0.9)));
        }
    }
    Ok(None)
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
        // Multi-stage resolution (exact → alias → type-guarded fuzzy).
        let (entity_id, resolution_conf) =
            match resolve_candidate(conn, &cand.surface, &cand.entity_type)? {
                Some((id, conf)) => {
                    // Record the new surface as an alias of the resolved entity.
                    dao::add_alias(conn, id, &cand.surface)?;
                    (id, conf)
                }
                None => (
                    dao::upsert_entity(conn, &key, &cand.surface, &cand.entity_type)?,
                    1.0,
                ),
            };
        let confidence = cand.confidence * resolution_conf;
        dao::insert_mention(
            conn,
            entity_id,
            document_id,
            chunk_id,
            &cand.surface,
            &cand.entity_type,
            cand.start as i64,
            cand.end as i64,
            confidence,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_key_strips_suffixes_and_case() {
        assert_eq!(canonical_key("Acme Corp"), canonical_key("acme corporation"));
        assert_eq!(canonical_key("Microsoft Inc."), "microsoft");
    }

    #[test]
    fn jaro_winkler_merges_typos_not_distinct_names() {
        // Fuzzy linking operates on canonical keys (suffix-stripped), so the
        // realistic comparisons are key-vs-key.
        let a = canonical_key("Google Inc");
        let b = canonical_key("Google Incorporated");
        assert_eq!(a, b, "suffix stripping already resolves this case");
        assert!(jaro_winkler("acme systems", "acme system") >= 0.93);
        // Distinct names sharing a long prefix must stay below the threshold.
        assert!(jaro_winkler("microsoft", "microscope") < FUZZY_LINK_THRESHOLD);
    }

    #[test]
    fn jaro_winkler_distinguishes_same_prefix_people() {
        // Same first token, different people — must not cross the threshold.
        assert!(jaro_winkler("john smith", "john structures") < FUZZY_LINK_THRESHOLD);
        assert!(jaro_winkler("alice jones", "alice jackson") < FUZZY_LINK_THRESHOLD);
    }
}
