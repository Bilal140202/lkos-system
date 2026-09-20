//! Hybrid retrieval: dense vector + lexical BM25, fused with Reciprocal Rank
//! Fusion, authority-weighted, diversified with MMR, fully explainable.
//!
//! Pipeline (see `docs/RETRIEVAL.md` and ADR-004):
//!
//! ```text
//! query ──┬─► embed ──► cosine top-N ─────┐
//!         └─► FTS5 MATCH ──► bm25 top-N ──┤
//!                                         ▼
//!                     RRF(k=60, w_dense, w_lexical)
//!                                   ▼
//!                          × authority_score
//!                                   ▼
//!                    + entity/phrase/section boosts
//!                                   ▼
//!                    MMR diversification (λ)
//!                                   ▼
//!                            top-k hits
//! ```
//!
//! Every hit carries `matched_by` explanations — the "no magic" rule.

use crate::embeddings::{cosine, EmbeddingProvider};
use crate::error::{LkosError, Result};
use crate::storage::dao::{self, HitRow};
use crate::types::*;
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};

/// Sanitize a query for FTS5 MATCH: quote each token with prefix matching,
/// join with AND. Deterministic and injection-safe (tokens are alphanumeric).
pub fn fts_escape(query: &str) -> String {
    let tokens = crate::embeddings::tokenize(query);
    if tokens.is_empty() {
        return String::new();
    }
    tokens
        .iter()
        .map(|t| format!("\"{}\"*", t))
        .collect::<Vec<_>>()
        .join(" AND ")
}

/// Compute the candidate id set for filters (None = no restriction).
fn candidate_ids(conn: &Connection, filters: Option<&Filters>) -> Result<Option<Vec<i64>>> {
    let Some(f) = filters else {
        return Ok(None);
    };
    let needs_set = f.document_ids.is_some()
        || f.doc_types.is_some()
        || f.must_contain.is_some()
        || f.as_of.is_some()
        || f.entities.is_some();
    if !needs_set {
        return Ok(None);
    }

    let mut sql = format!(
        "SELECT DISTINCT c.id FROM chunks c JOIN documents d ON d.id = c.document_id WHERE 1=1 {}",
        dao::filter_sql_parts(f)
    );
    let mut extra_params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    if let Some(entities) = &f.entities {
        if !entities.is_empty() {
            sql.push_str(
                " AND c.document_id IN (SELECT m.document_id FROM entity_mentions m \
                 JOIN entities e ON e.id = m.entity_id WHERE (",
            );
            for (i, name) in entities.iter().enumerate() {
                if i > 0 {
                    sql.push_str(" OR ");
                }
                sql.push_str(" lower(e.display_name) = lower(?) OR e.canonical_key = lower(?) ");
                extra_params.push(Box::new(name.clone()));
                extra_params.push(Box::new(crate::entities::canonical_key(name)));
            }
            sql.push_str("))");
        }
    }

    let mut stmt = conn.prepare(&sql)?;
    let mut all_params = dao::filter_params(f);
    all_params.extend(extra_params);
    dao::bind_params(&mut stmt, &all_params)?;
    let rows = stmt.raw_query().mapped(|r| r.get::<_, i64>(0));
    let mut ids = Vec::new();
    for r in rows {
        ids.push(r?);
    }
    Ok(Some(ids))
}

/// Run lexical (FTS5/BM25) search. Returns (chunk_id, rank, bm25-rank-score).
pub fn lexical_search(
    conn: &Connection,
    query: &str,
    top_n: usize,
    filters: Option<&Filters>,
) -> Result<Vec<(i64, usize, f32)>> {
    let match_expr = fts_escape(query);
    if match_expr.is_empty() {
        return Ok(Vec::new());
    }
    let candidates = candidate_ids(conn, filters)?;
    let limit_fetch = if candidates.is_some() {
        50_000
    } else {
        (top_n.max(50) * 8).min(5000)
    };

    let sql = if candidates.is_some() {
        "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ?1
         AND rowid IN (SELECT value FROM json_each(?2))
         ORDER BY rank LIMIT ?3"
    } else {
        "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ?1 ORDER BY rank LIMIT ?2"
    };
    let mut stmt = conn.prepare(sql)?;
    let mut rows = if let Some(ids) = candidates {
        stmt.query(rusqlite::params![
            match_expr,
            serde_json::to_string(&ids).unwrap_or_else(|_| "[]".into()),
            limit_fetch as i64
        ])?
    } else {
        stmt.query(rusqlite::params![match_expr, limit_fetch as i64])?
    };
    let mut out = Vec::new();
    let mut rank = 1usize;
    while let Some(row) = rows.next()? {
        let id: i64 = row.get(0)?;
        let _bm: f64 = row.get(1)?;
        out.push((id, rank, -(rank as f32)));
        rank += 1;
        if out.len() >= top_n {
            break;
        }
    }
    Ok(out)
}

/// Dense (cosine) search over all embeddings. Brute force; capped by `max_scan`.
pub fn vector_search(
    conn: &Connection,
    provider: &dyn EmbeddingProvider,
    query: &str,
    top_n: usize,
    filters: Option<&Filters>,
    max_scan: usize,
) -> Result<Vec<(i64, usize, f32)>> {
    let qv = provider.embed(query)?;
    let corpus = dao::all_embeddings(conn, filters)?;
    if corpus.len() > max_scan {
        return Err(LkosError::Other(format!(
            "dense scan would cover {} chunks (cap {max_scan}); \
             reduce the library, use filters, or enable an ANN index (roadmap 0.3)",
            corpus.len()
        )));
    }
    let mut scored: Vec<(i64, f32)> = corpus
        .into_iter()
        .map(|(id, v)| (id, cosine(&qv, &v)))
        .filter(|(_, s)| *s > 0.0)
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored
        .into_iter()
        .take(top_n)
        .enumerate()
        .map(|(i, (id, s))| (id, i + 1, s))
        .collect())
}

/// Core hybrid search. Returns fused+boosted+MMR-diversified hits.
#[allow(clippy::too_many_arguments)]
pub fn hybrid_search(
    conn: &Connection,
    provider: &dyn EmbeddingProvider,
    query: &str,
    top_k: usize,
    mode: RetrievalMode,
    filters: Option<&Filters>,
    rrf_k: usize,
    w_vector: f32,
    w_fts: f32,
    mmr_lambda: f32,
    max_scan: usize,
    known_entities: &[(String, i64)],
) -> Result<Vec<SearchHit>> {
    let top_n = (top_k * 6).clamp(24, 200);

    // Channel execution.
    let (vec_res, fts_res) = match mode {
        RetrievalMode::LexicalOnly => (Vec::new(), lexical_search(conn, query, top_n, filters)?),
        RetrievalMode::VectorOnly => (
            vector_search(conn, provider, query, top_n, filters, max_scan)?,
            Vec::new(),
        ),
        RetrievalMode::EntityLookup => (Vec::new(), Vec::new()),
        // Hybrid and Auto both run both channels (planner picks mode upstream).
        _ => {
            let v = vector_search(conn, provider, query, top_n, filters, max_scan)?;
            let f = lexical_search(conn, query, top_n, filters)?;
            (v, f)
        }
    };

    // Entity-intersection channel.
    let mut entity_chunk_name: HashMap<i64, String> = HashMap::new();
    if matches!(mode, RetrievalMode::EntityLookup) {
        let rank = 1usize;
        for (name, eid) in known_entities {
            for cid in dao::chunks_for_entity(conn, *eid)? {
                entity_chunk_name.entry(cid).or_insert_with(|| name.clone());
                let _ = rank;
            }
        }
    }

    // RRF fusion.
    let mut fusion: HashMap<i64, (f32, Vec<MatchSource>)> = HashMap::new();
    let bump =
        |id: i64, weight: f32, rank: usize, src: MatchSource, fusion: &mut HashMap<i64, (f32, Vec<MatchSource>)>| {
            let contribution = weight / ((rrf_k + rank) as f32);
            let entry = fusion.entry(id).or_insert((0.0, Vec::new()));
            entry.0 += contribution;
            entry.1.push(src);
        };
    for (id, rank, cos) in &vec_res {
        bump(
            *id,
            w_vector,
            *rank,
            MatchSource::Vector { rank: *rank, cosine: *cos },
            &mut fusion,
        );
    }
    for (id, rank, _) in &fts_res {
        bump(*id, w_fts, *rank, MatchSource::Fts { rank: *rank, bm25: 0.0 }, &mut fusion);
    }
    if matches!(mode, RetrievalMode::EntityLookup) {
        for (rank, (cid, name)) in entity_chunk_name.iter().enumerate() {
            bump(
                *cid,
                0.8,
                rank + 1,
                MatchSource::Entity { name: name.clone() },
                &mut fusion,
            );
        }
    }

    // Boosts: authority, phrase, entity, section title.
    let lower_query = query.to_lowercase();
    let query_tokens: HashSet<String> =
        crate::embeddings::tokenize(&lower_query).into_iter().collect();

    let mut scored_rows: Vec<(i64, f32, Vec<MatchSource>, HitRow)> = Vec::new();
    for (id, (mut score, mut sources)) in fusion {
        let row = dao::get_hit_row(conn, id)?;
        score *= row.authority_score.max(0.1);
        let lower_text = row.text.to_lowercase();
        if !lower_query.trim().is_empty() && lower_text.contains(&lower_query) {
            score += 0.08;
            sources.push(MatchSource::Phrase);
        }
        if !matches!(mode, RetrievalMode::EntityLookup) {
            for (name, _) in known_entities {
                if lower_text.contains(&name.to_lowercase()) {
                    score += 0.03;
                    sources.push(MatchSource::Entity { name: name.clone() });
                    break;
                }
            }
        }
        if let Some(sec) = &row.section_title {
            let sec_tokens: HashSet<String> =
                crate::embeddings::tokenize(&sec.to_lowercase()).into_iter().collect();
            let overlap = sec_tokens.intersection(&query_tokens).count();
            if overlap > 0 {
                score += 0.02 * overlap as f32;
                sources.push(MatchSource::SectionTitle);
            }
        }
        scored_rows.push((id, score, sources, row));
    }

    // Sort by score desc.
    scored_rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // MMR diversification on chunk embeddings.
    let selected = mmr_select(conn, scored_rows, top_k, mmr_lambda)?;

    // Render hits.
    let mut hits = Vec::new();
    for (rank, (id, score, sources, row)) in selected.into_iter().enumerate() {
        hits.push(SearchHit {
            chunk_id: id,
            document_id: row.document_id,
            document: row.filename.clone(),
            section: row.section_title.clone(),
            text: row.text.clone(),
            score,
            rank: rank + 1,
            matched_by: sources,
            authority_score: row.authority_score,
        });
    }
    Ok(hits)
}

/// Maximal Marginal Relevance selection with embedding lookups (cached).
type ScoredHit = (i64, f32, Vec<MatchSource>, HitRow);

fn mmr_select(
    conn: &Connection,
    mut scored: Vec<ScoredHit>,
    k: usize,
    lambda: f32,
) -> Result<Vec<ScoredHit>> {
    if k >= scored.len() || lambda >= 0.999 {
        scored.truncate(k);
        return Ok(scored);
    }
    let mut embeddings: HashMap<i64, Vec<f32>> = HashMap::new();
    let prefetch = (k + 16).min(scored.len());
    for (id, _, _, _) in scored.iter().take(prefetch) {
        if !embeddings.contains_key(id) {
            embeddings.insert(*id, dao::chunk_embedding(conn, *id)?.unwrap_or_default());
        }
    }
    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut remaining: Vec<usize> = (0..scored.len()).collect();

    let sim = |a: usize, b: usize| -> f32 {
        match (embeddings.get(&scored[a].0), embeddings.get(&scored[b].0)) {
            (Some(x), Some(y)) => cosine(x, y),
            _ => 0.0,
        }
    };

    while selected.len() < k && !remaining.is_empty() {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (ri, &cand) in remaining.iter().enumerate() {
            let relevance = scored[cand].1;
            let max_sim = selected
                .iter()
                .map(|&s| sim(s, cand))
                .fold(0.0f32, f32::max);
            let mmr = lambda * relevance - (1.0 - lambda) * max_sim;
            if mmr > best_val {
                best_val = mmr;
                best_idx = ri;
            }
        }
        let chosen = remaining.remove(best_idx);
        selected.push(chosen);
    }
    Ok(selected.into_iter().map(|i| scored[i].clone()).collect())
}
