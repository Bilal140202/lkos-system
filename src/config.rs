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
    pub worker_threads: usize,
    /// Default LLM summary temperature.
    pub summary_temperature: f32,
    /// Default LLM summary max tokens.
    pub summary_max_tokens: u32,
    /// Timeout for LLM invocations, seconds.
    pub llm_timeout_secs: u64,
    /// Run the full pipeline synchronously inside `ingest_*` calls
    /// (true = simpler determinism; false = background job queue).
    pub synchronous_ingestion: bool,
    /// Disable automatic knowledge enrichment (entities/claims/graph) if desired.
    pub enable_knowledge_extraction: bool,
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
        }
    }
}
