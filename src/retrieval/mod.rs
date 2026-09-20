//! Hybrid retrieval: dense vector + lexical BM25, fused with Reciprocal Rank
//! Fusion, reranked, authority-weighted, diversified with MMR, explainable.
//!
//! Pipeline (see `docs/RETRIEVAL.md` and ADR-004):
//!
//! ```text
//! query ──┬─► embed(model-filtered) ─► cosine top-N ─┐
//!         └─► FTS5 MATCH ─► bm25 top-N ─────────────┤
//!                                                    ▼
//!                     RRF(k=60, w_dense, w_lexical)
//!                                   ▼
//!                          × authority_score
//!                                   ▼
//!              + entity / phrase / section / freshness boosts
//!                                   ▼
//!              lexical-overlap rerank (top-N candidates, deterministic)
//!                                   ▼
//!                    MMR diversification (λ)
//!                                   ▼
//!                            top-k hits
//! ```
//!
//! Changes vs v0.1 (evidence in docs/RERANKING.md):
//! - BM25 scores are **kept** (v0.1 discarded them; `bm25` was always 0.0),
//!   so hits explain their true lexical evidence.
//! - The dense channel only compares chunks embedded by the same model
//!   (vector spaces are not comparable across models).
//! - The entity channel is deterministic (v0.1 ranked by HashMap order).
//! - MMR prefetches embeddings for ALL candidates up to a documented cap
//!   (v0.1 silently degraded to relevance-only beyond k+16).
//! - A deterministic lexical-overlap reranker re-scores the fused top-N
//!   before MMR; ablations in docs/EVALUATION.md quantify its effect.

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

/// Run lexical (FTS5/BM25) search. Returns (chunk_id, rank, bm25-score)
/// where score = -bm25 (FTS5 reports lower-is-better; we normalize to
/// higher-is-better). v0.1 returned `-(rank)` and threw the score away.
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
        let bm: f64 = row.get(1)?;
        // FTS5 rank = bm25 (more negative = better). Report higher-better.
        out.push((id, rank, -(bm as f32)));
        rank += 1;
        if out.len() >= top_n {
            break;
        }
    }
    Ok(out)
}

/// Dense (cosine) search restricted to chunks embedded by the provider's
/// model. Single-pass blob scan (v0.1 issued one SELECT per chunk).
pub fn vector_search(
    conn: &Connection,
    provider: &dyn EmbeddingProvider,
    query: &str,
    top_n: usize,
    filters: Option<&Filters>,
    max_scan: usize,
) -> Result<Vec<(i64, usize, f32)>> {
    let qv = provider.embed(query)?;
    let corpus = dao::all_embeddings(conn, filters, &provider.name())?;
    if corpus.len() > max_scan {
        return Err(LkosError::Other(format!(
            "dense scan would cover {} chunks (cap {max_scan}); \
             reduce the library, use filters, or shard the index",
            corpus.len()
        )));
    }
    let mut scored: Vec<(i64, f32)> = corpus
        .into_iter()
        .map(|(id, v)| (id, cosine(&qv, &v)))
        .filter(|(_, s)| *s > 0.0)
        .collect();
    // Deterministic order: score desc, then chunk id asc (tie-break).
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    Ok(scored
        .into_iter()
        .take(top_n)
        .enumerate()
        .map(|(i, (id, s))| (id, i + 1, s))
        .collect())
}

/// Core hybrid search. Returns fused+reranked+boosted+MMR-diversified hits.
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
    enable_reranking: bool,
    rerank_top_n: usize,
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

    // Entity-intersection channel — deterministic order (by entity id, then
    // chunk id). v0.1 ranked these by HashMap iteration order.
    let mut entity_chunk_name: HashMap<i64, String> = HashMap::new();
    let mut entity_chunk_ids: Vec<i64> = Vec::new();
    if matches!(mode, RetrievalMode::EntityLookup) {
        let mut by_id: Vec<(i64, &str)> = known_entities
            .iter()
            .map(|(name, eid)| (*eid, name.as_str()))
            .collect();
        by_id.sort_unstable();
        for (eid, name) in by_id {
            for cid in dao::chunks_for_entity(conn, eid)? {
                if entity_chunk_name.insert(cid, name.to_string()).is_none() {
                    entity_chunk_ids.push(cid);
                }
            }
        }
        entity_chunk_ids.sort_unstable();
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
    for (id, rank, bm25_score) in &fts_res {
        bump(
            *id,
            w_fts,
            *rank,
            MatchSource::Fts { rank: *rank, bm25: *bm25_score },
            &mut fusion,
        );
    }
    if matches!(mode, RetrievalMode::EntityLookup) {
        for (rank, cid) in entity_chunk_ids.iter().enumerate() {
            let name = entity_chunk_name
                .get(cid)
                .cloned()
                .unwrap_or_default();
            bump(*cid, 0.8, rank + 1, MatchSource::Entity { name }, &mut fusion);
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

    // Sort by score desc, tie-break by chunk id (deterministic).
    scored_rows.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // Lexical-overlap rerank over the fused top-N (deterministic).
    if enable_reranking && !matches!(mode, RetrievalMode::EntityLookup) {
        rerank_lexical_overlap(&mut scored_rows, rerank_top_n, &query_tokens);
    }

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

/// Deterministic lexical-overlap reranker.
///
/// Re-scores the top `window` fused candidates with a BM25-flavored term
/// coverage signal: matched query terms (weighted by inverse document
/// length) over the candidate text. The reranker only REORDERS candidates
/// the channels already surfaced — it cannot inject unseen chunks — so it
/// is safe to apply without a second candidate pass. The fused score and
/// rerank score are combined 50/50 after min-max normalization within the
/// window, preserving fusion evidence in `matched_by`.
fn rerank_lexical_overlap(
    scored: &mut [(i64, f32, Vec<MatchSource>, HitRow)],
    window: usize,
    query_tokens: &HashSet<String>,
) {
    if window == 0 || query_tokens.is_empty() || scored.is_empty() {
        return;
    }
    let window = window.min(scored.len());

    // Pass 1: compute raw rerank scores for the window.
    let mut raw: Vec<f32> = Vec::with_capacity(window);
    for entry in scored.iter().take(window) {
        let text_tokens: HashSet<String> =
            crate::embeddings::tokenize(&entry.3.text).into_iter().collect();
        let overlap = query_tokens.intersection(&text_tokens).count();
        // Length-normalized coverage: avoids biasing toward long chunks.
        let cov = overlap as f32 / query_tokens.len() as f32;
        let len_penalty = (text_tokens.len() as f32 + 32.0).ln();
        raw.push(cov / len_penalty.sqrt());
    }
    let min = raw.iter().cloned().fold(f32::MAX, f32::min);
    let max = raw.iter().cloned().fold(f32::MIN, f32::max);
    let span = (max - min).max(f32::EPSILON);

    // Pass 2: blend normalized rerank score with the fused score (50/50).
    for (i, entry) in scored.iter_mut().take(window).enumerate() {
        let rr = (raw[i] - min) / span;
        entry.1 = 0.5 * entry.1 + 0.5 * rr;
    }

    // Pass 3: re-sort the window (deterministic tie-break by chunk id).
    scored[..window].sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
}

/// Maximal Marginal Relevance selection with embedding lookups (cached for
/// the full candidate window up to 64 — v0.1 prefetched only k+16 and
/// silently degraded to relevance-only beyond that).
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
    let prefetch = scored.len().min(64);
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
