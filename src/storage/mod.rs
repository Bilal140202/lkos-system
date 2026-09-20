//! SQLite storage layer: connection management, WAL setup, migrations, FTS5.
//!
//! SQLite is the *canonical* store (ADR-001). Every artifact lives here:
//! documents, chunks, the FTS5 lexical index, entities, claims, relationships,
//! provenance, jobs. The database is a single portable file; WAL mode gives
//! crash safety (see SECURITY/PRIVACY docs).

pub(crate) mod dao;
pub(crate) mod schema;

use crate::error::Result;
use rusqlite::Connection;
use std::path::{Path, PathBuf};

/// Current schema version (PRAGMA user_version).
pub const SCHEMA_VERSION: i64 = 5;

/// A thin wrapper over a SQLite connection with LKOS pragmas applied.
pub struct Store {
    conn: Connection,
    path: PathBuf,
}

impl Store {
    /// Open (or create) the database at `path`, applying pragmas and migrations.
    pub fn open(path: impl AsRef<Path>) -> Result<Store> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let conn = Connection::open(&path)?;
        Self::configure(&conn)?;
        let store = Store { conn, path };
        store.migrate()?;
        Ok(store)
    }

    /// Open an in-memory database (tests, ephemeral usage).
    pub fn open_in_memory() -> Result<Store> {
        let conn = Connection::open_in_memory()?;
        Self::configure(&conn)?;
        let store = Store {
            conn,
            path: PathBuf::from(":memory:"),
        };
        store.migrate()?;
        Ok(store)
    }

    fn configure(conn: &Connection) -> Result<()> {
        // WAL: crash-safe, concurrent readers (see docs/DATABASE.md).
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        Ok(())
    }

    /// Apply all pending migrations sequentially.
    pub fn migrate(&self) -> Result<()> {
        let version: i64 = self
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version < 1 {
            self.conn.execute_batch(schema::V1_BASE)?;
        }
        if version < 2 {
            self.conn.execute_batch(schema::V2_KNOWLEDGE)?;
        }
        if version < 3 {
            self.conn.execute_batch(schema::V3_GRAPH_CLAIMS)?;
        }
        if version < 4 {
            self.conn.execute_batch(schema::V4_JOBS_META)?;
        }
        if version < 5 {
            self.conn.execute_batch(schema::V5_SEMANTIC_INCREMENTAL)?;
        }
        if version > SCHEMA_VERSION {
            return Err(crate::error::LkosError::Other(format!(
                "database schema v{version} is newer than this build (v{SCHEMA_VERSION}); \
                 upgrade LKOS instead of downgrading onto this library"
            )));
        }
        self.conn
            .pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(())
    }

    /// Read the current schema version.
    pub fn schema_version(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))?)
    }

    /// Run an integrity check; returns rows of the check output.
    pub fn integrity_check(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("PRAGMA integrity_check")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Exclusive access for transactions. Public so applications embedding
    /// LKOS can build custom tooling (backup scripts, migrations) on the
    /// same connection pool discipline the engine uses.
    pub fn conn(&mut self) -> &mut Connection {
        &mut self.conn
    }

    /// Read-only access.
    pub fn read(&self) -> &Connection {
        &self.conn
    }

    /// Vacuum the database file.
    pub fn vacuum(&self) -> Result<()> {
        self.conn.execute_batch("VACUUM")?;
        Ok(())
    }

    /// Filesystem size of the main database file (best-effort).
    pub fn file_size_bytes(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }

    /// Backup the database to `dest` using SQLite's online backup API.
    pub fn backup_to(&self, dest: impl AsRef<Path>) -> Result<()> {
        let dest_path = dest.as_ref().to_path_buf();
        if let Some(parent) = dest_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut dst = Connection::open(&dest_path)?;
        let backup = rusqlite::backup::Backup::new(&self.conn, &mut dst)?;
        backup.run_to_completion(64, std::time::Duration::from_millis(5), None)?;
        Ok(())
    }
}
