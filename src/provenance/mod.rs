//! Provenance utilities: the evidence graph in exported form.
//!
//! Every derived artifact in LKOS (chunk, entity mention, claim,
//! relationship, summary) has provenance rows linking it to its source
//! document, chunk, offsets, extractor identity, and time. This module adds
//! bulk export so knowledge never becomes detached from evidence.

use crate::error::Result;
use crate::storage::dao;
use crate::storage::Store;
use serde::Serialize;
use std::collections::BTreeMap;

/// A full provenance export for one document.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentProvenance {
    /// Document id.
    pub document_id: i64,
    /// Filename.
    pub filename: String,
    /// Content hash of the original bytes.
    pub content_hash: String,
    /// Provenance rows grouped by artifact type.
    pub artifacts: BTreeMap<String, Vec<crate::types::ProvenanceRecord>>,
}

/// Export all provenance records for a document.
pub fn export_document(store: &Store, document_id: i64) -> Result<Option<DocumentProvenance>> {
    let doc = match dao::get_document(store.read(), document_id) {
        Ok(d) => d,
        Err(crate::LkosError::DocumentNotFound(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let rows = all_for_document(store, document_id)?;
    let mut artifacts: BTreeMap<String, Vec<crate::types::ProvenanceRecord>> = BTreeMap::new();
    for r in rows {
        artifacts.entry(r.artifact_type.clone()).or_default().push(r);
    }
    Ok(Some(DocumentProvenance {
        document_id,
        filename: doc.filename,
        content_hash: doc.content_hash,
        artifacts,
    }))
}

/// All provenance rows touching a document (artifact rows carry document_id).
pub fn all_for_document(store: &Store, document_id: i64) -> Result<Vec<crate::types::ProvenanceRecord>> {
    let mut stmt = store.read().prepare(
        "SELECT id, artifact_type, artifact_id, document_id, chunk_id, start_offset, end_offset, \
         extractor, extractor_version, created_at
         FROM provenance WHERE document_id = ?1 ORDER BY id",
    )?;
    let rows = stmt.query_map(rusqlite::params![document_id], |r| {
        let start: Option<i64> = r.get(5)?;
        let end: Option<i64> = r.get(6)?;
        Ok(crate::types::ProvenanceRecord {
            id: r.get(0)?,
            artifact_type: r.get(1)?,
            artifact_id: r.get(2)?,
            document_id: r.get(3)?,
            chunk_id: r.get(4)?,
            offsets: match (start, end) {
                (Some(s), Some(e)) => Some((s, e)),
                _ => None,
            },
            extractor: r.get(7)?,
            extractor_version: r.get(8)?,
            created_at: r.get(9)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}
