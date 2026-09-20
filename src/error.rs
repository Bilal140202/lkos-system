//! Structured error types for LKOS.
//!
//! Every fallible operation in LKOS returns [`Result`] with a concrete
//! [`LkosError`] variant so callers can match on failure modes instead of
//! string matching. Errors never panic across the public API.

use thiserror::Error;

/// Concrete error type for every LKOS operation.
#[derive(Debug, Error)]
pub enum LkosError {
    /// Underlying SQLite failure (including FTS5 errors).
    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),

    /// Filesystem failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization failure.
    #[error("serialization error: {0}")]
    Json(#[from] serde_json::Error),

    /// A document id was requested but does not exist.
    #[error("document not found: {0}")]
    DocumentNotFound(i64),

    /// An entity id was requested but does not exist.
    #[error("entity not found: {0}")]
    EntityNotFound(i64),

    /// The file type has no extractor.
    #[error("unsupported file type: {0}")]
    UnsupportedFileType(String),

    /// Input that cannot be processed (empty, binary garbage, oversized).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// A structured query was malformed.
    #[error("invalid query: {0}")]
    InvalidQuery(String),

    /// Local LLM invocation failed.
    #[error("llm error: {0}")]
    Llm(String),

    /// A background job was cancelled.
    #[error("job cancelled")]
    Cancelled,

    /// Embedding dimension mismatch between index and provider.
    #[error(
        "embedding mismatch: index was built with model '{index_model}' (dim {index_dim}), \
         configured provider is '{provider_model}' (dim {provider_dim}). Reindex required."
    )]
    EmbeddingMismatch {
        /// Model name recorded in the index.
        index_model: String,
        /// Dimension recorded in the index.
        index_dim: usize,
        /// Configured provider name.
        provider_model: String,
        /// Configured provider dimension.
        provider_dim: usize,
    },

    /// Any other condition with a human-readable message.
    #[error("{0}")]
    Other(String),
}

/// Result alias used across the crate.
pub type Result<T> = std::result::Result<T, LkosError>;
