//! Background job system: persistence, atomic claiming, exponential backoff,
//! dead-lettering, cancellation, crash recovery.
//!
//! Improvements over v0.1 (all evidence-gated, see docs/DATABASE.md):
//! - **Atomic claim**: an immediate transaction + conditional UPDATE replaces
//!   the SELECT-then-UPDATE race; multiple workers can safely compete.
//! - **Exponential backoff**: failed jobs are re-released at
//!   `base * 2^(attempt-1)` seconds (capped), instead of immediately.
//! - **Dead-letter**: terminal failures keep their payload + error for
//!   inspection (`failed` status) instead of vanishing.
//! - **Cancellation**: `cancel_job` sets `cancelled`; workers check the flag
//!   between stages, and pending jobs can be cancelled by id.
//! - **Progress**: workers report 0–100 progress for long jobs.
//! - **Configurable worker count** (`Config::worker_threads`).

use crate::error::Result;
use crate::storage::dao;
use rusqlite::{Connection, OptionalExtension};

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

/// Enqueue a semantic re-embed sweep (re-embed chunks whose embedding model
/// is not `current`).
pub fn enqueue_reembed(conn: &Connection, current: &str) -> Result<i64> {
    let payload = serde_json::json!({ "model": current }).to_string();
    dao::enqueue_job(conn, "reembed_stale", &payload, 4)
}

/// Enqueue an LSA (re)training job.
pub fn enqueue_train_semantic(conn: &Connection) -> Result<i64> {
    dao::enqueue_job(conn, "train_semantic", "{}", 3)
}

/// Enqueue a document deletion cleanup job.
pub fn enqueue_delete_document(conn: &Connection, document_id: i64) -> Result<i64> {
    let payload = serde_json::json!({ "document_id": document_id }).to_string();
    dao::enqueue_job(conn, "delete_document", &payload, 1)
}

/// Claim the next runnable job atomically.
pub fn claim_next(conn: &mut Connection) -> Result<Option<Job>> {
    Ok(dao::claim_next_atomic(conn)?
        .map(|(id, kind, payload)| Job { id, kind, payload }))
}

/// Mark a job done.
pub fn complete(conn: &Connection, job: &Job) -> Result<()> {
    let attempts = job_attempts(conn, job.id)?;
    dao::update_job(conn, job.id, "done", attempts + 1, None)
}

/// Report job progress (0-100).
pub fn progress(conn: &Connection, job_id: i64, pct: i64) -> Result<()> {
    dao::set_job_progress(conn, job_id, pct)
}

/// Was this job cancelled while running?
pub fn is_cancelled(conn: &Connection, job_id: i64) -> Result<bool> {
    let status: Option<String> = conn
        .query_row(
            "SELECT status FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .optional()?
        .flatten();
    Ok(status.as_deref() == Some("cancelled"))
}

/// Mark a job failed; re-queue with exponential backoff until max attempts,
/// then dead-letter. Returns `true` when the job was re-queued.
pub fn fail_and_maybe_retry(
    conn: &Connection,
    job: &Job,
    error: &str,
    backoff_base_secs: u64,
    backoff_max_secs: u64,
) -> Result<bool> {
    let attempts = job_attempts(conn, job.id)?;
    let max: i64 = conn.query_row(
        "SELECT max_attempts FROM jobs WHERE id = ?1",
        rusqlite::params![job.id],
        |r| r.get(0),
    )?;
    let new_attempts = attempts + 1;
    if new_attempts >= max {
        // Dead-letter: terminal status with payload + error preserved.
        dao::update_job_sched(conn, job.id, "failed", new_attempts, Some(error), None)?;
        Ok(false)
    } else {
        // Exponential backoff: base * 2^(attempt-1), capped.
        let exp = new_attempts.saturating_sub(1).min(16) as u32;
        let delay = backoff_base_secs
            .saturating_mul(1u64 << exp)
            .min(backoff_max_secs);
        let run_at = (chrono::Utc::now() + chrono::Duration::seconds(delay as i64))
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        dao::update_job_sched(conn, job.id, "pending", new_attempts, Some(error), Some(&run_at))?;
        Ok(true)
    }
}

/// Cancel a pending job; returns whether a row was cancelled.
pub fn cancel_job(conn: &Connection, job_id: i64) -> Result<bool> {
    let n = conn.execute(
        "UPDATE jobs SET status = 'cancelled', updated_at = ?2
         WHERE id = ?1 AND status = 'pending'",
        rusqlite::params![job_id, dao::now()],
    )?;
    Ok(n > 0)
}

/// Cancel every pending job (used on shutdown with a drain flag).
pub fn cancel_pending(conn: &Connection) -> Result<usize> {
    let n = conn.execute(
        "UPDATE jobs SET status = 'cancelled', updated_at = ?2 WHERE status = 'pending'",
        rusqlite::params![dao::now()],
    )?;
    Ok(n)
}

fn job_attempts(conn: &Connection, job_id: i64) -> Result<i64> {
    Ok(conn.query_row(
        "SELECT attempts FROM jobs WHERE id = ?1",
        rusqlite::params![job_id],
        |r| r.get(0),
    )?)
}
