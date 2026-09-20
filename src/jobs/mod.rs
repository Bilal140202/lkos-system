//! Background job system: persistence, claiming, retry/backoff, recovery.
//!
//! Jobs live in the `jobs` table (crash-safe). On engine open, any job left in
//! `running` by a crash is reset to `pending` (recovery, see docs/DATABASE.md
//! and the crash-safety tests). Workers are plain OS threads that claim one
//! job at a time — SQLite serializes writers, so a single worker is optimal
//! for v0.1 (documented in PERFORMANCE.md).

use crate::error::Result;
use crate::storage::dao;
use rusqlite::Connection;

/// A claimed job handed to a worker.
#[derive(Debug, Clone)]
pub struct Job {
    /// Job row id.
    pub id: i64,
    /// Job kind (e.g. `process_document`).
    pub kind: String,
    /// JSON payload (e.g. `{"document_id": 3}`).
    pub payload: String,
}

/// Enqueue a `process_document` job for a document id.
pub fn enqueue_process_document(conn: &Connection, document_id: i64) -> Result<i64> {
    let payload = serde_json::json!({ "document_id": document_id }).to_string();
    dao::enqueue_job(conn, "process_document", &payload, 5)
}

/// Enqueue a document deletion cleanup job.
pub fn enqueue_delete_document(conn: &Connection, document_id: i64) -> Result<i64> {
    let payload = serde_json::json!({ "document_id": document_id }).to_string();
    dao::enqueue_job(conn, "delete_document", &payload, 1)
}

/// Claim the next job (marks it running atomically enough for a single worker).
pub fn claim_next(conn: &Connection) -> Result<Option<Job>> {
    if let Some((id, kind, payload)) = dao::next_job(conn)? {
        dao::update_job(conn, id, "running", 0, None)?;
        Ok(Some(Job { id, kind, payload }))
    } else {
        Ok(None)
    }
}

/// Mark a job done.
pub fn complete(conn: &Connection, job: &Job) -> Result<()> {
    let attempts = job_attempts(conn, job.id)?;
    dao::update_job(conn, job.id, "done", attempts + 1, None)
}

/// Mark a job failed; re-queue for retry until max attempts.
pub fn fail_and_maybe_retry(conn: &Connection, job: &Job, error: &str) -> Result<bool> {
    let attempts = job_attempts(conn, job.id)?;
    let max: i64 = conn.query_row(
        "SELECT max_attempts FROM jobs WHERE id = ?1",
        rusqlite::params![job.id],
        |r| r.get(0),
    )?;
    let new_attempts = attempts + 1;
    if new_attempts >= max {
        dao::update_job(conn, job.id, "failed", new_attempts, Some(error))?;
        Ok(false)
    } else {
        dao::update_job(conn, job.id, "pending", new_attempts, Some(error))?;
        Ok(true)
    }
}

fn job_attempts(conn: &Connection, job_id: i64) -> Result<i64> {
    Ok(conn.query_row(
        "SELECT attempts FROM jobs WHERE id = ?1",
        rusqlite::params![job_id],
        |r| r.get(0),
    )?)
}

/// Cancel every pending job (used on shutdown with a drain flag).
pub fn cancel_pending(conn: &Connection) -> Result<usize> {
    let n = conn.execute(
        "UPDATE jobs SET status = 'cancelled', updated_at = ?2 WHERE status = 'pending'",
        rusqlite::params![dao::now()],
    )?;
    Ok(n)
}
