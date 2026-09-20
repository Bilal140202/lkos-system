//! Claim persistence and the contradiction engine.
//!
//! Claims are extracted deterministically (see `knowledge::extract_claims`) and
//! persisted here with provenance. The contradiction engine detects numeric
//! disagreements between claims sharing a (subject_key, predicate_key):
//! e.g. "Acme revenue $10M" vs "Acme revenue $12M". Conflicts are *preserved*,
//! never silently merged (the engine keeps disagreement with sources+dates).

use crate::error::Result;
use crate::storage::dao;
use rusqlite::Connection;

/// Persist extracted claims for one chunk + run conflict detection against
/// previously stored numeric claims. Returns number of claims stored and
/// number of conflicts (new) detected.
pub fn persist_claims(
    conn: &Connection,
    document_id: i64,
    chunk_id: i64,
    claims: &[crate::knowledge::ExtractedClaim],
) -> Result<(usize, usize)> {
    let mut stored = 0usize;
    let mut new_conflicts = 0usize;

    for claim in claims {
        let subject_key = crate::entities::canonical_key(&claim.subject);
        let predicate_key = claim.predicate.to_lowercase();
        let claim_id = dao::insert_claim(
            conn,
            &claim.subject,
            &claim.predicate,
            &claim.object,
            &claim.sentence,
            document_id,
            chunk_id,
            claim.confidence,
            crate::knowledge::CLAIM_EXTRACTOR,
            &subject_key,
            &predicate_key,
            claim.valid_from.as_deref(),
            claim.valid_until.as_deref(),
        )?;
        dao::insert_provenance(
            conn,
            "claim",
            &claim_id.to_string(),
            Some(document_id),
            Some(chunk_id),
            None,
            crate::knowledge::CLAIM_EXTRACTOR,
            crate::knowledge::KNOWLEDGE_VERSION,
        )?;
        stored += 1;

        // Numeric conflict detection against existing claims with same keys.
        if let Some(value) = dao::parse_first_number(&claim.object) {
            let existing = dao::numeric_claims(conn)?;
            for (other_id, os, op, ovalue, oyear, odoc) in existing {
                if other_id == claim_id || os != subject_key || op != predicate_key {
                    continue;
                }
                if ovalue <= 0.0 || value <= 0.0 {
                    continue;
                }
                let delta = ((value - ovalue).abs()) / ovalue.max(value);
                if delta > 0.05 {
                    // Temporal explanation if periods differ.
                    let this_year = claim
                        .valid_from
                        .as_deref()
                        .map(|s| s[..4.min(s.len())].to_string());
                    let other_year = oyear
                        .as_deref()
                        .map(|s| s[..4.min(s.len())].to_string());
                    let explanation = match (&this_year, &other_year) {
                        (Some(a), Some(b)) if a == b => format!(
                            "same metric differs by {:.0}% within the same period ({a}); disagreement preserved",
                            delta * 100.0
                        ),
                        (Some(a), Some(b)) => format!(
                            "same metric differs by {:.0}% across periods: {a} vs {b}",
                            delta * 100.0
                        ),
                        _ => format!(
                            "same metric differs by {:.0}% between documents {} and {}",
                            delta * 100.0,
                            odoc,
                            document_id
                        ),
                    };
                    // Avoid duplicate conflicts for the same pair.
                    let dup: bool = conflicts_pair_exists(conn, claim_id, other_id)?;
                    if !dup {
                        dao::insert_conflict(
                            conn,
                            &subject_key,
                            &predicate_key,
                            other_id,
                            claim_id,
                            delta as f32,
                            &explanation,
                        )?;
                        new_conflicts += 1;
                    }
                }
            }
        }
    }
    Ok((stored, new_conflicts))
}

fn conflicts_pair_exists(conn: &Connection, a: i64, b: i64) -> Result<bool> {
    let n: i64 = conn.query_row(
        "SELECT COUNT(*) FROM claim_conflicts
         WHERE (claim_a = ?1 AND claim_b = ?2) OR (claim_a = ?2 AND claim_b = ?1)",
        rusqlite::params![a, b],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}
