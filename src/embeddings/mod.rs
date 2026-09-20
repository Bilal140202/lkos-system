//! Embedding abstraction.
//!
//! LKOS is model-agnostic (ADR-005): anything implementing
//! [`EmbeddingProvider`] can power the dense channel. v0.1 ships a fully
//! offline, dependency-free provider — the *feature-hashing embedder*
//! (Weinberger et al., ICML 2009) over word unigrams + bigrams + char
//! trigrams. It is deterministic, allocation-cheap, and needs no model files,
//! but it is **lexical**, not semantic: it cannot match paraphrases. That is a
//! documented, deliberate v0.1 trade-off — semantic providers
//! (fastembed/ONNX) plug into the same trait with zero engine changes
//! (ROADMAP 0.2), and the hybrid BM25 channel carries most precision load.

use crate::error::Result;

/// A local embedding provider.
pub trait EmbeddingProvider: Send + Sync {
    /// Provider/model name recorded in the index (used for reindex checks).
    fn name(&self) -> &str;
    /// Dimensionality of produced vectors.
    fn dim(&self) -> usize;
    /// Embed a batch of texts (batching allows providers to amortize work).
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Embed a single text.
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed_batch(std::slice::from_ref(&text))?;
        Ok(v.pop().unwrap_or_default())
    }
}

/// Deterministic feature-hashing embedder (offline, no model files).
pub struct HashingEmbedder {
    dim: usize,
    seed: u64,
}

impl HashingEmbedder {
    /// Create with a dimension (default 256) and fixed seed.
    pub fn new(dim: usize) -> Self {
        HashingEmbedder { dim: dim.max(64), seed: 0x9E_37_79_B9_7F_4A_7C_15 }
    }

    /// Override seed (rarely needed; documented for reproducibility).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn hash(&self, token: &str, salt: u64) -> (usize, f32) {
        // FNV-1a 64 with seed + salt.
        let mut h: u64 = 0xcb_f2_9c_e4_84_22_23_25 ^ self.seed ^ salt.wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        for b in token.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x10_00_00_00_01_b3);
        }
        let idx = (h % self.dim as u64) as usize;
        let sign = if h & (1 << 63) == 0 { 1.0 } else { -1.0 };
        (idx, sign)
    }
}

impl EmbeddingProvider for HashingEmbedder {
    fn name(&self) -> &str {
        "hashing-lex-v1"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }
}

impl HashingEmbedder {
    fn embed_one(&self, text: &str) -> Vec<f32> {
        let lower = text.to_lowercase();
        let words = tokenize(&lower);
        let mut v = vec![0.0f32; self.dim];

        // Word unigrams: weight 1 + ln(tf).
        let mut tf: Vec<(String, f32)> = Vec::new();
        for w in &words {
            match tf.iter_mut().find(|(s, _)| s == w) {
                Some((_, c)) => *c += 1.0,
                None => tf.push((w.clone(), 1.0)),
            }
        }
        for (w, c) in &tf {
            let weight = 1.0 + c.ln();
            let (i, s) = self.hash(w, 1);
            v[i] += s * weight;
            // char trigram smoothing for morphological robustness
            add_trigrams(self, &mut v, w, 0.3 * weight);
        }

        // Adjacent bigrams: weight 0.5.
        for pair in words.windows(2) {
            let bigram = format!("{} {}", pair[0], pair[1]);
            let (i, s) = self.hash(&bigram, 2);
            v[i] += s * 0.5;
        }

        // L2 normalize.
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

fn add_trigrams(e: &HashingEmbedder, v: &mut [f32], word: &str, weight: f32) {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 3 {
        return;
    }
    let mut tri = String::new();
    for w in chars.windows(3) {
        tri.clear();
        tri.extend(w.iter());
        let (i, s) = e.hash(&tri, 3);
        v[i] += s * weight;
    }
}

/// Tokenizer: lowercase alphanumeric runs (digits kept for identifiers/dates).
pub fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Cosine similarity of two vectors (assumes L2-normalized inputs; falls back
/// to full cosine when not normalized).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        0.0
    } else {
        dot / (na * nb)
    }
}
