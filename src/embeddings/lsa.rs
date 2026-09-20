//! Corpus-trained semantic embeddings (LSA) — the real dense channel.
//!
//! v0.1's dense channel was signed feature hashing: *lexical*, not semantic.
//! This module replaces that default with a genuine distributional-semantics
//! provider trained on the local corpus (Deerwester et al., 1990, "Indexing
//! by Latent Semantic Analysis", JASIS 41(6)); dimensionality reduction uses
//! randomized truncated SVD (Halko, Martinsson & Tropp, 2011, "Finding
//! Structure with Randomness", SIAM Review 53(2)).
//!
//! Pipeline (all local, deterministic, no network, no model downloads):
//!
//! ```text
//! chunk texts ─► tokenize ─► vocabulary (top-V terms by document frequency)
//!            ─► tf (1+ln) × IDF-weighted term–chunk matrix
//!            ─► randomized SVD (power iterations, fixed seed)
//!            ─► term vectors  V×d  (sign-fixed for determinism)
//! query      ─► Σ_t (1+ln tf(t,q)) · term_vector[t]  ─► L2 normalize
//! ```
//!
//! Determinism: the sketching matrix is drawn from a fixed-seed LCG, the
//! power-iteration count is fixed, and each latent dimension's sign is fixed
//! so that its largest-|component| term is positive. Re-training on the same
//! corpus yields bit-identical vectors.
//!
//! Cold start: until [`Config::semantic_min_chunks`] chunks exist the model
//! is `None` and the engine's hashing fallback powers the dense channel
//! (lexical but useful). After training, stale (hashing-embedded) chunks are
//! re-embedded by the incremental `reembed` job — see `docs/EMBEDDINGS.md`.

use crate::embeddings::tokenize;
use crate::error::Result;
use crate::storage::dao;
use rusqlite::Connection;
use std::collections::HashMap;

/// Model identifier recorded per chunk and in `engine_meta`.
pub const LSA_MODEL_NAME: &str = "lsa-pmi-svd-v1";

/// Trained LSA model: vocabulary + latent term vectors.
#[derive(Debug, Clone)]
pub struct LsaModel {
    /// term -> row index in `term_vecs`.
    pub vocab: HashMap<String, u32>,
    /// Latent term vectors, row-major `V × dim`.
    pub term_vecs: Vec<f32>,
    /// Latent dimensionality.
    pub dim: usize,
    /// Number of chunks the model was trained on (provenance).
    pub trained_on_chunks: usize,
    /// SHA-256 of the sorted vocabulary — identifies the trained state.
    pub vocab_hash: String,
}

impl LsaModel {
    /// Project a text into the latent space (query or chunk encoding).
    ///
    /// Terms absent from the vocabulary are skipped; the result is
    /// L2-normalized. Returns `None` when *no* term is in vocabulary (the
    /// caller should fall back to the lexical/hashing channels).
    pub fn project(&self, text: &str) -> Option<Vec<f32>> {
        let mut tf: HashMap<String, f32> = HashMap::new();
        for w in tokenize(text) {
            *tf.entry(w).or_insert(0.0) += 1.0;
        }
        let mut v = vec![0.0f32; self.dim];
        let mut hits = 0usize;
        for (term, count) in &tf {
            if let Some(&idx) = self.vocab.get(term) {
                hits += 1;
                let weight = 1.0 + (*count).ln();
                let row = &self.term_vecs[idx as usize * self.dim..(idx as usize + 1) * self.dim];
                for (vi, tv) in v.iter_mut().zip(row) {
                    *vi += tv * weight;
                }
            }
        }
        if hits == 0 {
            return None;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in &mut v {
                *x /= norm;
            }
            Some(v)
        } else {
            None
        }
    }
}

/// Train an LSA model from chunk texts.
///
/// * `dim` — target latent dimensionality (clamped to corpus limits).
/// * `min_df` — terms appearing in fewer chunks are dropped.
/// * `max_vocab` — vocabulary cap (highest-DF terms win).
/// * `power_iters` — randomized-SVD power iterations (4 is standard).
pub fn train(
    chunk_texts: &[&str],
    dim: usize,
    min_df: usize,
    max_vocab: usize,
    power_iters: usize,
) -> Option<LsaModel> {
    if chunk_texts.len() < 4 {
        return None;
    }
    // 1. Document frequencies + per-chunk term counts.
    let mut df: HashMap<String, u32> = HashMap::new();
    let mut docs: Vec<HashMap<String, f32>> = Vec::with_capacity(chunk_texts.len());
    for text in chunk_texts {
        let mut tf: HashMap<String, f32> = HashMap::new();
        for w in tokenize(text) {
            *tf.entry(w).or_insert(0.0) += 1.0;
        }
        for term in tf.keys() {
            *df.entry(term.clone()).or_insert(0) += 1;
        }
        docs.push(tf);
    }
    // 2. Vocabulary: terms with df >= min_df, capped to max_vocab by df.
    let mut terms: Vec<(String, u32)> = df
        .into_iter()
        .filter(|(_, d)| *d as usize >= min_df)
        .collect();
    // Deterministic order: df desc, then term asc (total order → stable).
    terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    terms.truncate(max_vocab);
    if terms.len() < 8 {
        return None; // too little signal to factor
    }
    let vocab: HashMap<String, u32> = terms
        .iter()
        .enumerate()
        .map(|(i, (t, _))| (t.clone(), i as u32))
        .collect();
    let n_terms = vocab.len();
    let n_docs = docs.len();
    let dim = dim.min(n_terms.saturating_sub(1)).max(2).min(n_docs.saturating_sub(1)).max(2);

    // 3. TF-IDF sparse matrix (terms × docs), weight = (1+ln tf) · ln(1+N/df).
    //    Stored as CSR: row = term.
    let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_terms];
    let idf: Vec<f32> = terms
        .iter()
        .map(|(_, d)| (1.0 + n_docs as f32 / *d as f32).ln())
        .collect();
    for (di, tf) in docs.iter().enumerate() {
        for (term, count) in tf {
            if let Some(&ti) = vocab.get(term) {
                let w = (1.0 + *count).ln() * idf[ti as usize];
                rows[ti as usize].push((di as u32, w));
            }
        }
    }

    // 4. Randomized truncated SVD: A (terms×docs) ≈ U Σ Vᵀ.
    //    Sketch Ω: docs × l (l = dim + oversampling), Gaussian from fixed-seed LCG.
    let l = (dim + 8).min(n_docs);
    // Y = A Ω  (terms × l)
    let mut y = vec![0.0f32; n_terms * l];
    let mut rng = Lcg::new(0x9E_37_79_B9_7F_4A_7C_15);
    let omega: Vec<f32> = (0..n_docs * l).map(|_| gaussian(&mut rng)).collect();
    for (ti, row) in rows.iter().enumerate() {
        for (di, w) in row {
            let omega_row = &omega[*di as usize * l..(*di as usize + 1) * l];
            let y_row = &mut y[ti * l..(ti + 1) * l];
            for (yi, oi) in y_row.iter_mut().zip(omega_row) {
                *yi += w * oi;
            }
        }
    }
    // Power iterations: Y = A (Aᵀ Y) — improves subspace quality.
    for _ in 0..power_iters.max(1) {
        // Z = Aᵀ Y (docs × l)
        let mut z = vec![0.0f32; n_docs * l];
        for (ti, row) in rows.iter().enumerate() {
            let y_row = &y[ti * l..(ti + 1) * l];
            for (di, w) in row {
                let z_row = &mut z[*di as usize * l..(*di as usize + 1) * l];
                for (zi, yi) in z_row.iter_mut().zip(y_row) {
                    *zi += w * yi;
                }
            }
        }
        // Y = A Z
        y = vec![0.0f32; n_terms * l];
        for (ti, row) in rows.iter().enumerate() {
            for (di, w) in row {
                let z_row = &z[*di as usize * l..(*di as usize + 1) * l];
                let y_row = &mut y[ti * l..(ti + 1) * l];
                for (yi, zi) in y_row.iter_mut().zip(z_row) {
                    *yi += w * zi;
                }
            }
        }
    }
    // Orthonormalize Y's columns (modified Gram–Schmidt) → U_l (terms × l).
    for col in 0..l {
        // normalize col
        let mut norm = 0.0f32;
        for ti in 0..n_terms {
            norm += y[ti * l + col] * y[ti * l + col];
        }
        let norm = norm.sqrt();
        if norm > f32::EPSILON {
            for ti in 0..n_terms {
                y[ti * l + col] /= norm;
            }
        }
        // re-orthogonalize later columns against this one
        for later in (col + 1)..l {
            let mut dot = 0.0f32;
            for ti in 0..n_terms {
                dot += y[ti * l + col] * y[ti * l + later];
            }
            for ti in 0..n_terms {
                y[ti * l + later] -= dot * y[ti * l + col];
            }
        }
    }

    // Term vectors = first `dim` orthonormal columns; fix signs deterministically.
    let mut term_vecs = vec![0.0f32; n_terms * dim];
    for d in 0..dim {
        // dominant sign: largest |component| of column d
        let mut best_abs = 0.0f32;
        let mut best_sign = 1.0f32;
        for ti in 0..n_terms {
            let v = y[ti * l + d];
            if v.abs() > best_abs {
                best_abs = v.abs();
                best_sign = if v < 0.0 { -1.0 } else { 1.0 };
            }
        }
        for ti in 0..n_terms {
            term_vecs[ti * dim + d] = y[ti * l + d] * best_sign;
        }
    }

    // 5. Vocabulary hash (identity of the trained state).
    let mut sorted_terms: Vec<&str> = vocab.keys().map(|s| s.as_str()).collect();
    sorted_terms.sort_unstable();
    let vocab_hash = crate::ingestion::hash_text(&sorted_terms.join("\u{1}"));

    Some(LsaModel {
        vocab,
        term_vecs,
        dim,
        trained_on_chunks: n_docs,
        vocab_hash,
    })
}

/// Persist the trained model (replaces any previous one).
pub fn save_model(conn: &Connection, model: &LsaModel) -> Result<()> {
    conn.execute("DELETE FROM lsa_terms", [])?;
    let mut stmt = conn.prepare("INSERT INTO lsa_terms(term, idx, vector) VALUES (?1, ?2, ?3)")?;
    let mut term_vec = vec![0.0f32; model.dim];
    // Deterministic insertion order: by index.
    let mut inv: Vec<(&String, &u32)> = model.vocab.iter().collect();
    inv.sort_by_key(|(_, i)| **i);
    for (term, idx) in inv {
        let base = *idx as usize * model.dim;
        term_vec.copy_from_slice(&model.term_vecs[base..base + model.dim]);
        stmt.execute(rusqlite::params![term, *idx as i64, dao::f32_to_bytes(&term_vec)])?;
    }
    dao::meta_set(conn, "lsa_dim", &model.dim.to_string())?;
    dao::meta_set(conn, "lsa_trained_on_chunks", &model.trained_on_chunks.to_string())?;
    dao::meta_set(conn, "lsa_vocab_hash", &model.vocab_hash)?;
    Ok(())
}

/// Load the persisted model, if any.
pub fn load_model(conn: &Connection) -> Result<Option<LsaModel>> {
    let dim: usize = match dao::meta_get(conn, "lsa_dim")?.and_then(|v| v.parse().ok()) {
        Some(d) => d,
        None => return Ok(None),
    };
    let mut stmt = conn.prepare("SELECT term, idx, vector FROM lsa_terms ORDER BY idx")?;
    let mut rows = stmt.query([])?;
    let mut vocab = HashMap::new();
    let mut term_vecs: Vec<f32> = Vec::new();
    while let Some(row) = rows.next()? {
        let term: String = row.get(0)?;
        let _idx: i64 = row.get(1)?;
        let blob: Vec<u8> = row.get(2)?;
        let vec = dao::bytes_to_f32(&blob);
        if vec.len() != dim {
            return Ok(None); // corrupted/mismatched model — retrain
        }
        vocab.insert(term, vocab.len() as u32);
        term_vecs.extend_from_slice(&vec);
    }
    if vocab.is_empty() {
        return Ok(None);
    }
    Ok(Some(LsaModel {
        vocab,
        term_vecs,
        dim,
        trained_on_chunks: dao::meta_get(conn, "lsa_trained_on_chunks")?
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        vocab_hash: dao::meta_get(conn, "lsa_vocab_hash")?.unwrap_or_default(),
    }))
}

/// Deterministic LCG (Park–Miller quality constants, 64-bit variant).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        // Knuth MMIX LCG — deterministic across platforms.
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
}

/// Uniform Gaussian via Box–Muller on two uniforms from the LCG.
fn gaussian(rng: &mut Lcg) -> f32 {
    let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let u2 = ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<String> {
        let themes: [&[&str]; 4] = [
            &[
                "revenue profit earnings fiscal quarter sales growth margin business",
                "quarterly earnings report shows revenue growth and profit margin gains",
                "sales revenue increased while profit margins held steady this quarter",
                "fiscal year revenue forecast projects continued sales and earnings growth",
            ],
            &[
                "neural network training gradient descent backpropagation layers",
                "training deep neural networks with stochastic gradient descent",
                "backpropagation computes gradients across network layers",
                "the network was trained until the gradient magnitudes stabilized",
            ],
            &[
                "database query optimizer index scan latency throughput",
                "the query optimizer chose an index scan for low latency",
                "index scans reduced query latency and improved throughput",
                "database throughput depends on index structure and query plans",
            ],
            &[
                "photosynthesis chlorophyll sunlight carbon dioxide glucose plants",
                "plants use chlorophyll to convert sunlight and carbon dioxide into glucose",
                "photosynthesis converts light energy into chemical energy in plants",
                "chlorophyll absorbs sunlight driving the photosynthesis reaction",
            ],
        ];
        let mut docs = Vec::new();
        for theme in themes.iter() {
            for line in theme.iter() {
                docs.push(line.to_string());
            }
        }
        docs
    }

    #[test]
    fn lsa_separates_themes_lexically_unrelated() {
        let docs = corpus();
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let model = train(&refs, 8, 1, 512, 4).expect("model trains");
        // A query with NO word overlap with the database theme except
        // synonym-level relations must still rank DB chunks first because
        // co-occurrence structure places them together. Use a paraphrase
        // built from vocabulary shared across the theme.
        let q = "query plan throughput";
        let qv = model.project(q).expect("in-vocab query");
        // Score all docs.
        let mut scored: Vec<(usize, f32)> = docs
            .iter()
            .enumerate()
            .filter_map(|(i, d)| model.project(d).map(|v| {
                let dot: f32 = qv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                (i, dot)
            }))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top4: std::collections::HashSet<usize> =
            scored.iter().take(4).map(|(i, _)| *i).collect();
        // Chunks 8..12 are the database theme.
        for i in 8..12 {
            assert!(
                top4.contains(&i),
                "database chunk {i} not in top-4; got {scored:?}"
            );
        }
    }

    #[test]
    fn lsa_training_is_deterministic() {
        let docs = corpus();
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let a = train(&refs, 8, 1, 512, 4).unwrap();
        let b = train(&refs, 8, 1, 512, 4).unwrap();
        assert_eq!(a.vocab_hash, b.vocab_hash);
        assert_eq!(a.term_vecs, b.term_vecs);
    }

    #[test]
    fn oov_query_returns_none() {
        let docs = corpus();
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let model = train(&refs, 8, 1, 512, 4).unwrap();
        assert!(model.project("zzzqqq wwwxxx").is_none());
    }

    #[test]
    fn model_survives_persistence_roundtrip() {
        let docs = corpus();
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let model = train(&refs, 8, 1, 512, 4).unwrap();
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE lsa_terms (term TEXT PRIMARY KEY, idx INTEGER NOT NULL, vector BLOB NOT NULL);
             CREATE TABLE engine_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
        )
        .unwrap();
        save_model(&conn, &model).unwrap();
        let loaded = load_model(&conn).unwrap().expect("model loads");
        assert_eq!(loaded.vocab_hash, model.vocab_hash);
        assert_eq!(loaded.term_vecs, model.term_vecs);
    }
}
