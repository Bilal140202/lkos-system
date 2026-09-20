//! # LKOS — Local Knowledge Object System
//!
//! An offline, privacy-first hybrid retrieval and knowledge engine for
//! local-first applications. LKOS transforms raw documents into structured,
//! provenance-aware knowledge: chunks, embeddings, entities, claims,
//! relationships — all stored in a single portable SQLite file.
//!
//! Design invariants:
//! - **Local-first**: zero network, zero telemetry, zero cloud dependencies.
//! - **LLM-optional**: the engine is fully useful without any model.
//! - **Provenance-first**: every derived artifact traces back to evidence.
//! - **Explainable**: every result explains *why* it was retrieved.
//! - **Honest**: documented limitations are part of the contract
//!   (see `docs/NON_GOALS.md`).
//!
//! Quick start:
//! ```no_run
//! use lkos::{Config, Lkos, QueryRequest};
//!
//! # fn main() -> lkos::Result<()> {
//! let engine = Lkos::open("library.lkos", Config::default())?;
//! engine.ingest_file("handbook.md")?;
//! let resp = engine.query(QueryRequest::new("vacation policy").top_k(5))?;
//! for hit in &resp.hits {
//!     println!("{} :: {} :: {:.4}", hit.rank, hit.document, hit.score);
//! }
//! # Ok(())
//! # }
//! ```

#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

pub mod chunking;
pub mod claims;
pub mod config;
pub mod embeddings;
pub mod engine;
pub mod entities;
pub mod error;
pub mod events;
pub mod graph;
pub mod ingestion;
pub mod jobs;
pub mod knowledge;
pub mod llm;
pub mod provenance;
pub mod query;
pub mod retrieval;
pub mod storage;
pub mod temporal;
pub mod types;

pub use config::Config;
pub use engine::Lkos;
pub use error::{LkosError, Result};
pub use types::*;

/// Crate version (recorded in engine_meta for auditability).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
