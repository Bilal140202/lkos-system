//! Claim persistence and the contradiction engine (v0.2 evidence layer).
//!
//! Claims are extracted deterministically (see `knowledge::extract_claims`)
//! and persisted here with provenance *including character offsets*. The
//! contradiction engine compares claims sharing a (subject_key, predicate_key)
//! after unit/magnitude normalization ("$10 million" ≡ "10M" ≡ 10_000_000).
//!
//! Conflict taxonomy (preserved, never silently merged):
//! - `same-period-disagreement` — same metric, same validity period, >5% delta.
//! - `cross-period` — same metric, different validity periods (often NOT an
//!   error: revenue changes over time; surfaced for context).
//! - `undated-disagreement` — no periods parseable.
//! - `negation-conflict` — one source asserts, another negates.
//!
//! Detection cost is O(log N + K) via the `(subject_key, predicate_key)`
//! index — v0.1 rescanned every numeric claim per insert (O(N²)).

use crate::error::Result;
use crate::storage::dao;
use rusqlite::Connection;

/// Relative numeric delta above which two claims of the same metric are
/// considered to disagree (5% — configurable per deployment later).
pub const CONFLICT_DELTA_THRESHOLD: f32 = 0.05;

/// Maximum prior claims compared per new claim (conflict-detection window).
pub const MAX_COMPARISONS: usize = 40;
/// Maximum conflict rows materialized per claim insert (pair explosion cap).
pub const MAX_CONFLICTS_PER_CLAIM: usize = 4;

/// Persist extracted claims for one chunk + run conflict detection against
/// prior claims with identical (subject_key, predicate_key). Returns number
/// of claims stored and number of new conflicts detected.
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
            Some(claim.start_offset as i64),
            Some(claim.end_offset as i64),
        )?;
        dao::insert_provenance(
            conn,
            "claim",
            &claim_id.to_string(),
            Some(document_id),
            Some(chunk_id),
            Some((claim.start_offset as i64, claim.end_offset as i64)),
            crate::knowledge::CLAIM_EXTRACTOR,
            crate::knowledge::KNOWLEDGE_VERSION,
        )?;
        stored += 1;

        // Negation conflicts: same subject+predicate, opposite polarity.
        if claim.negated {
            let existing = dao::claims_for_subject(conn, &subject_key)?;
            for other in &existing {
                if other.id == claim_id || other.predicate.to_lowercase() != predicate_key {
                    continue;
                }
                let other_negated = sentence_is_negative(&other.sentence);
                if !other_negated && !other.object.is_empty() {
                    let dup = conflicts_pair_exists(conn, claim_id, other.id)?;
                    if !dup {
                        dao::insert_conflict(
                            conn,
                            &subject_key,
                            &predicate_key,
                            other.id,
                            claim_id,
                            1.0,
                            "one source asserts, another negates the same predicate; disagreement preserved",
                            "negation-conflict",
                        )?;
                        new_conflicts += 1;
                    }
                }
            }
            continue;
        }

        // Numeric conflict detection against prior claims with the same keys.
        let Some(value) = dao::parse_first_number(&claim.object) else {
            continue;
        };
        if value <= 0.0 {
            continue;
        }
        // Indexed lookup (v0.1 rescanned every numeric claim per insert).
        // Only the most recent MAX_COMPARISONS claims of the same metric are
        // compared and at most MAX_CONFLICTS_PER_CLAIM conflict rows are
        // materialized per claim: disagreement *existence* is the signal;
        // enumerating every pair is quadratic and useless at corpus scale
        // (measured: 2.4k claims -> ~500k pair rows, ingestion stall).
        let existing = dao::numeric_claims_for_keys(conn, &subject_key, &predicate_key)?;
        let existing: Vec<_> = existing.iter().rev().take(MAX_COMPARISONS).collect();
        let mut conflicts_for_this_claim = 0usize;
        for other in existing {
            let (other_id, ovalue, oyear) = (other.0, other.1, other.2.clone());
            if other_id == claim_id || ovalue <= 0.0 {
                continue;
            }
            if conflicts_for_this_claim >= MAX_CONFLICTS_PER_CLAIM {
                break;
            }
            let delta = ((value - ovalue).abs()) / ovalue.max(value);
            if delta <= CONFLICT_DELTA_THRESHOLD as f64 {
                continue;
            }
            let this_year = claim
                .valid_from
                .as_deref()
                .map(|s| s[..4.min(s.len())].to_string());
            let other_year = oyear.as_deref().map(|s| s[..4.min(s.len())].to_string());
            let (explanation, kind) = match (&this_year, &other_year) {
                (Some(a), Some(b)) if a == b => (
                    format!(
                        "same metric differs by {:.0}% within the same period ({a}); disagreement preserved",
                        delta * 100.0
                    ),
                    "same-period-disagreement",
                ),
                (Some(a), Some(b)) => (
                    format!(
                        "same metric differs by {:.0}% across periods: {a} vs {b}",
                        delta * 100.0
                    ),
                    "cross-period",
                ),
                _ => (
                    format!(
                        "same metric differs by {:.0}% between sources; no validity periods parseable",
                        delta * 100.0
                    ),
                    "undated-disagreement",
                ),
            };
            let dup = conflicts_pair_exists(conn, claim_id, other_id)?;
            if !dup {
                dao::insert_conflict(
                    conn,
                    &subject_key,
                    &predicate_key,
                    other_id,
                    claim_id,
                    delta as f32,
                    &explanation,
                    kind,
                )?;
                new_conflicts += 1;
                conflicts_for_this_claim += 1;
            }
        }
    }
    Ok((stored, new_conflicts))
}

/// Heuristic: does a sentence negate its predicate?
pub fn sentence_is_negative(sentence: &str) -> bool {
    let lower = sentence.to_lowercase();
    for w in [" not ", "n't ", " never ", " no longer ", " without "] {
        if lower.contains(w) {
            return true;
        }
    }
    // Sentence-final / contraction-adjacent forms ("is not", "wasn't").
    lower.contains("n't") || lower.contains("not")
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
