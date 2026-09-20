//! Engine configuration.
//!
//! Defaults are chosen to be conservative for local desktop use
//! (see `docs/architecture/PERFORMANCE.md` for measured trade-offs).

use serde::{Deserialize, Serialize};

/// Tunable engine configuration. All fields have sane defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Maximum characters per chunk (approx. 250-400 tokens).
    pub chunk_max_chars: usize,
    /// Minimum characters for a chunk to stand alone (else merged forward).
    pub chunk_min_chars: usize,
    /// Overlap characters between prose chunks (0 = none; structural chunking
    /// makes overlap mostly unnecessary and it duplicates evidence).
    pub chunk_overlap: usize,
    /// Embedding dimension for the default hashing embedder.
    pub embedding_dim: usize,
    /// Reciprocal Rank Fusion constant `k` (Cormack et al. use 60).
    pub rrf_k: usize,
    /// Weight of the dense channel in RRF.
    pub weight_vector: f32,
    /// Weight of the lexical channel in RRF.
    pub weight_fts: f32,
    /// MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    pub mmr_lambda: f32,
    /// Maximum chunks scanned by brute-force dense search before refusing
    /// (documented threshold; HNSW is on the roadmap).
    pub max_dense_scan: usize,
    /// Number of background worker threads.
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
    /// Default LLM summary temperature.
    pub summary_temperature: f32,
    /// Default LLM summary max tokens.
    pub summary_max_tokens: u32,
    /// Timeout for LLM invocations, seconds.
    pub llm_timeout_secs: u64,
    /// Run the full pipeline synchronously inside `ingest_*` calls
    /// (true = simpler determinism; false = background job queue).
    #[serde(default = "default_true")]
    pub synchronous_ingestion: bool,
    /// Disable automatic knowledge enrichment (entities/claims/graph) if desired.
    #[serde(default = "default_true")]
    pub enable_knowledge_extraction: bool,
    /// Dense-channel embedding provider: `"hashing"` (lexical fallback) or
    /// `"lsa"` (corpus-trained semantic, see `docs/EMBEDDINGS.md`).
    #[serde(default = "default_embedding_provider")]
    pub embedding_provider: String,
    /// Latent dimensionality of the LSA model.
    #[serde(default = "default_lsa_dim")]
    pub lsa_dim: usize,
    /// Minimum document frequency for an LSA vocabulary term.
    #[serde(default = "default_lsa_min_df")]
    pub lsa_min_df: usize,
    /// Vocabulary cap for the LSA model.
    #[serde(default = "default_lsa_max_vocab")]
    pub lsa_max_vocab: usize,
    /// Minimum chunks before LSA training is attempted (cold-start guard).
    #[serde(default = "default_semantic_min_chunks")]
    pub semantic_min_chunks: usize,
    /// Retrain LSA when the share of stale-model chunks exceeds this fraction.
    #[serde(default = "default_retrain_fraction")]
    pub retrain_fraction: f32,
    /// Apply the deterministic lexical-overlap reranker over the fused
    /// candidates (evidence-gated: see docs/RERANKING.md ablations).
    #[serde(default = "default_true")]
    pub enable_reranking: bool,
    /// Number of fused candidates the reranker examines.
    #[serde(default = "default_rerank_top_n")]
    pub rerank_top_n: usize,
    /// Boost weight for freshness under `Latest` temporal intent (0 = off).
    #[serde(default = "default_freshness_boost")]
    pub freshness_boost: f32,
    /// Base delay (seconds) for exponential job backoff: base * 2^(attempt-1).
    #[serde(default = "default_backoff_base")]
    pub job_backoff_base_secs: u64,
    /// Maximum backoff delay cap (seconds).
    #[serde(default = "default_backoff_cap")]
    pub job_backoff_max_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            chunk_max_chars: 1100,
            chunk_min_chars: 200,
            chunk_overlap: 0,
            embedding_dim: 256,
            rrf_k: 60,
            weight_vector: 0.5,
            weight_fts: 0.5,
            mmr_lambda: 0.7,
            max_dense_scan: 250_000,
            worker_threads: 1,
            summary_temperature: 0.2,
            summary_max_tokens: 256,
            llm_timeout_secs: 300,
            synchronous_ingestion: true,
            enable_knowledge_extraction: true,
            embedding_provider: "lsa".into(),
            lsa_dim: 128,
            lsa_min_df: 2,
            lsa_max_vocab: 8192,
            semantic_min_chunks: 24,
            retrain_fraction: 0.2,
            enable_reranking: true,
            rerank_top_n: 24,
            freshness_boost: 0.05,
            job_backoff_base_secs: 2,
            job_backoff_max_secs: 300,
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_embedding_provider() -> String {
    "lsa".into()
}
fn default_lsa_dim() -> usize {
    128
}
fn default_lsa_min_df() -> usize {
    2
}
fn default_lsa_max_vocab() -> usize {
    8192
}
fn default_semantic_min_chunks() -> usize {
    24
}
fn default_retrain_fraction() -> f32 {
    0.2
}
fn default_rerank_top_n() -> usize {
    24
}
fn default_freshness_boost() -> f32 {
    0.05
}
fn default_backoff_base() -> u64 {
    2
}
fn default_backoff_cap() -> u64 {
    300
}
fn default_worker_threads() -> usize {
    1
}
