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
    ///
    /// Correctness contract (three layers):
    /// 1. **Per-batch version stamps.** `user_version` is bumped immediately
    ///    after each committed batch, so a crash between batches resumes from
    ///    the last completed batch instead of re-running already-applied DDL.
    /// 2. **Signature-column skip (self-healing).** Databases written by
    ///    builds that stamped the version only once at the end may carry
    ///    fully-applied batches under a stale stamp. Before running a batch
    ///    that contains `ALTER TABLE ADD COLUMN`, the signature column's
    ///    presence proves the batch committed — it is skipped, not re-applied
    ///    (re-applying would fail with "duplicate column name" and leave the
    ///    library permanently unopenable).
    /// 3. **Concurrent-open tolerance.** Two connections opening the same
    ///    fresh file can both observe a stale stamp; if the batch DDL then
    ///    fails with "duplicate column" while the signature column exists,
    ///    the other connection committed it and this one treats it as done.
    pub fn migrate(&self) -> Result<()> {
        /// (target version, batch SQL, (table, column) proving the batch applied)
        type Migration = (i64, &'static str, Option<(&'static str, &'static str)>);
        const MIGRATIONS: &[Migration] = &[
            (1, schema::V1_BASE, None),
            (2, schema::V2_KNOWLEDGE, Some(("chunks", "knowledge_json"))),
            (3, schema::V3_GRAPH_CLAIMS, None),
            (4, schema::V4_JOBS_META, None),
            (
                5,
                schema::V5_SEMANTIC_INCREMENTAL,
                Some(("chunks", "embedding_model")),
            ),
        ];

        let version: i64 = self
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version > SCHEMA_VERSION {
            return Err(crate::error::LkosError::Other(format!(
                "database schema v{version} is newer than this build (v{SCHEMA_VERSION}); \
                 upgrade LKOS instead of downgrading onto this library"
            )));
        }

        for &(target, sql, signature) in MIGRATIONS {
            if version >= target {
                continue;
            }
            let already_applied = match signature {
                Some((table, column)) => self.column_exists(table, column)?,
                None => false,
            };
            if !already_applied {
                if let Err(e) = self.conn.execute_batch(sql) {
                    let healed = match signature {
                        Some((table, column)) => self.column_exists(table, column)?,
                        None => false,
                    };
                    let duplicate_column = matches!(
                        &e,
                        rusqlite::Error::SqliteFailure(_, Some(msg))
                            if msg.contains("duplicate column")
                    );
                    if !(healed && duplicate_column) {
                        return Err(e.into());
                    }
                    // else: a concurrent connection committed this batch
                    // between our version read and our DDL — treat as done.
                }
            }
            self.conn.pragma_update(None, "user_version", target)?;
        }
        Ok(())
    }

    /// Whether `table` currently has a column named `column` (case-sensitive,
    /// matching SQLite's identifier semantics for our schema names).
    fn column_exists(&self, table: &str, column: &str) -> Result<bool> {
        // Table and column names come from the compile-time migration table,
        // never from user input — direct interpolation is safe here.
        let sql = format!("PRAGMA table_info({table})");
        let mut stmt = self.conn.prepare(&sql)?;
        let found = stmt
            .query_map([], |r| r.get::<_, String>(1))?
            .any(|name| name.as_deref() == Ok(column));
        Ok(found)
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
