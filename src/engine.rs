//! The LKOS engine facade — the only type most applications need.
//!
//! ```no_run
//! use lkos::{Config, Lkos, QueryRequest};
//!
//! # fn main() -> lkos::Result<()> {
//! let engine = Lkos::open("library.lkos", Config::default())?;
//! engine.ingest_bytes("report.md", b"# Report\n\nRevenue was $10M in 2024.")?;
//! let response = engine.query(QueryRequest::new("revenue 2024"))?;
//! println!("top hit: {}", response.hits[0].document);
//! # Ok(())
//! # }
//! ```

use crate::chunking;
use crate::config::Config;
use crate::embeddings::{EmbeddingProvider, HashingEmbedder};
use crate::error::{LkosError, Result};
use crate::events::EventBus;
use crate::ingestion;
use crate::llm::{grounded_prompt, LlmProvider};
use crate::storage::dao;
use crate::storage::Store;
use crate::types::*;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// The engine. Cheap to clone (shared inner state).
#[derive(Clone)]
pub struct Lkos {
    inner: Arc<Inner>,
}

struct Inner {
    path: std::path::PathBuf,
    config: Config,
    embedder: Box<dyn EmbeddingProvider>,
    bus: EventBus,
    llm: RwLock<Option<Arc<dyn LlmProvider>>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl std::fmt::Debug for Lkos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lkos")
            .field("path", &self.inner.path)
            .field("config", &self.inner.config)
            .finish_non_exhaustive()
    }
}

impl Lkos {
    // ------------------------------------------------------------------
    // lifecycle
    // ------------------------------------------------------------------

    /// Open (creating if needed) a knowledge base at `path` with `config`.
    pub fn open(path: impl AsRef<std::path::Path>, config: Config) -> Result<Lkos> {
        let path = path.as_ref().to_path_buf();
        {
            let store = Store::open(&path)?;
            // Record / verify embedding model identity.
            let meta_model = dao::meta_get(store.read(), "embedding_model")?;
            match meta_model {
                None => {
                    dao::meta_set(store.read(), "embedding_model", "hashing-lex-v1")?;
                    dao::meta_set(
                        store.read(),
                        "embedding_dim",
                        &config.embedding_dim.to_string(),
                    )?;
                }
                Some(_m) => {
                    // Hashing embedder is the only built-in; dimension is the check.
                    let meta_dim: usize = dao::meta_get(store.read(), "embedding_dim")?
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(config.embedding_dim);
                    if meta_dim != config.embedding_dim {
                        return Err(LkosError::EmbeddingMismatch {
                            index_model: _m,
                            index_dim: meta_dim,
                            provider_model: "hashing-lex-v1".into(),
                            provider_dim: config.embedding_dim,
                        });
                    }
                }
            }
            // Crash recovery: requeue interrupted jobs.
            let recovered = dao::recover_running_jobs(store.read())?;
            if recovered > 0 {
                dao::meta_set(
                    store.read(),
                    "last_recovery",
                    &format!("requeued {recovered} interrupted jobs"),
                )?;
            }
        }
        let engine = Lkos {
            inner: Arc::new(Inner {
                path,
                config,
                embedder: Box::new(HashingEmbedder::new(256)),
                bus: EventBus::new(),
                llm: RwLock::new(None),
                worker: Mutex::new(None),
                shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            }),
        };
        if !engine.inner.config.synchronous_ingestion {
            engine.start_worker();
        }
        Ok(engine)
    }

    /// Open with an in-memory database (tests/examples).
    pub fn open_in_memory(config: Config) -> Result<Lkos> {
        // Route through the file-based path with a temp file for API symmetry.
        let dir = std::env::temp_dir().join(format!("lkos-mem-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("mem-{}.lkos", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
        Self::open(path, config)
    }

    /// Install an LLM provider (optional capability).
    pub fn set_llm(&self, provider: Arc<dyn LlmProvider>) {
        *self
            .inner
            .llm
            .write()
            .expect("llm lock") = Some(provider);
    }

    /// Subscribe to knowledge events.
    pub fn subscribe(&self) -> Receiver<Event> {
        self.inner.bus.subscribe()
    }

    /// The engine's configuration.
    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    /// Stop background workers and flush pending state.
    pub fn close(&self) -> Result<()> {
        self.inner
            .shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut w) = self.inner.worker.lock() {
            if let Some(handle) = w.take() {
                let _ = handle.join();
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // ingestion
    // ------------------------------------------------------------------

    /// Ingest a document from raw bytes. Idempotent on identical content.
    pub fn ingest_bytes(&self, filename: &str, bytes: &[u8]) -> Result<DocumentInfo> {
        let extracted = ingestion::extract(filename, bytes)?;
        let mut store = Store::open(&self.inner.path)?;
        let (doc_id, created) = self.register_document(&mut store, &extracted)?;
        if !created {
            return dao::get_document(store.read(), doc_id);
        }
        dao::set_pending_text(store.conn(), doc_id, &extracted.text)?;
        if self.inner.config.synchronous_ingestion {
            drop(store);
            self.process_document(doc_id)?;
            let store = Store::open(&self.inner.path)?;
            dao::get_document(store.read(), doc_id)
        } else {
            crate::jobs::enqueue_process_document(store.conn(), doc_id)?;
            self.inner
                .bus
                .publish("document.queued", doc_id, filename);
            dao::get_document(store.read(), doc_id)
        }
    }

    /// Ingest a document from a filesystem path.
    pub fn ingest_file(&self, path: impl AsRef<std::path::Path>) -> Result<DocumentInfo> {
        let path = path.as_ref();
        let extracted = ingestion::extract_file(path)?;
        let mut store = Store::open(&self.inner.path)?;
        let (doc_id, created) = self.register_document(&mut store, &extracted)?;
        if !created {
            return dao::get_document(store.read(), doc_id);
        }
        dao::set_pending_text(store.conn(), doc_id, &extracted.text)?;
        if self.inner.config.synchronous_ingestion {
            drop(store);
            self.process_document(doc_id)?;
            let store = Store::open(&self.inner.path)?;
            dao::get_document(store.read(), doc_id)
        } else {
            crate::jobs::enqueue_process_document(store.conn(), doc_id)?;
            self.inner.bus.publish("document.queued", doc_id, &extracted.filename);
            dao::get_document(store.read(), doc_id)
        }
    }

    fn register_document(
        &self,
        store: &mut Store,
        extracted: &ingestion::ExtractedDocument,
    ) -> Result<(i64, bool)> {
        let conn = store.conn();
        let (doc_id, created) = dao::upsert_document(
            conn,
            extracted.path_for_storage.as_deref().unwrap_or(""),
            &extracted.filename,
            extracted.doc_type.as_str(),
            extracted.size,
            &extracted.content_hash,
            extracted.text.len(),
            ingestion::EXTRACTOR_VERSION,
            chunking::CHUNKER_VERSION,
            self.inner.embedder.name(),
            crate::knowledge::KNOWLEDGE_VERSION,
        )?;
        if created {
            self.inner.bus.publish("document.added", doc_id, &extracted.filename);
        }
        Ok((doc_id, created))
    }

    /// Run (or re-run) the full knowledge pipeline for a document.
    ///
    /// Public so that applications embedding LKOS can drive the pipeline
    /// themselves (e.g. after enqueueing jobs in `synchronous_ingestion=false`
    /// mode without spawning the built-in worker).
    #[allow(clippy::too_many_lines)]
    pub fn process_document(&self, doc_id: i64) -> Result<()> {
        let cfg = self.inner.config.clone();
        let mut store = Store::open(&self.inner.path)?;
        let doc = dao::get_document(store.read(), doc_id)?;
        let Some(text) = dao::take_pending_text(store.conn(), doc_id)? else {
            return Err(LkosError::Other(format!(
                "document {doc_id} has no pending text (already processed?)"
            )));
        };

        // INDEXING ---------------------------------------------------------
        dao::set_readiness(store.conn(), doc_id, ReadinessState::Indexing)?;
        self.inner
            .bus
            .publish("document.status", doc_id, "indexing");

        let doc_type = match doc.doc_type.as_str() {
            "markdown" => crate::types::DocType::Markdown,
            "code" => crate::types::DocType::Code,
            "data" => crate::types::DocType::Data,
            _ => crate::types::DocType::Text,
        };
        let proposed = chunking::chunk_document(&text, doc_type, &cfg);

        // Embed + store chunks.
        let texts: Vec<&str> = proposed.iter().map(|p| p.text.as_str()).collect();
        let embeddings = self.inner.embedder.embed_batch(&texts)?;
        let mut chunk_ids = Vec::with_capacity(proposed.len());
        let mut all_keywords: Vec<String> = Vec::new();
        let mut sections_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (i, p) in proposed.iter().enumerate() {
            let emb_bytes = dao::f32_to_bytes(&embeddings[i]);
            let ko = if cfg.enable_knowledge_extraction {
                let total_docs = dao::total_documents(store.read())?;
                crate::knowledge::extract_knowledge(
                    &p.text,
                    &|term| {
                        // df lookup happens within this connection
                        store
                            .read()
                            .query_row(
                                "SELECT df FROM term_df WHERE term = ?1",
                                rusqlite::params![term],
                                |r| r.get::<_, i64>(0),
                            )
                            .ok()
                    },
                    total_docs,
                )
            } else {
                crate::knowledge::KnowledgeObject::default()
            };
            for (kw, _) in &ko.keywords {
                all_keywords.push(kw.clone());
            }
            if let Some(sec) = &p.section_title {
                sections_seen.insert(sec.clone());
            }
            let cid = dao::insert_chunk(
                store.conn(),
                doc_id,
                i as i64,
                &p.text,
                p.section_title.as_deref(),
                p.kind.as_str(),
                p.start_offset as i64,
                p.end_offset as i64,
                p.authority,
                &crate::ingestion::hash_text(&p.text),
                &emb_bytes,
            )?;
            dao::update_chunk_artifacts(
                store.conn(),
                cid,
                &emb_bytes,
                &crate::knowledge::ko_to_json(&ko),
            )?;
            dao::insert_provenance(
                store.conn(),
                "chunk",
                &cid.to_string(),
                Some(doc_id),
                Some(cid),
                Some((p.start_offset as i64, p.end_offset as i64)),
                &format!("{} -> {} -> {}", ingestion::EXTRACTOR_VERSION, chunking::CHUNKER_VERSION, self.inner.embedder.name()),
                crate::knowledge::KNOWLEDGE_VERSION,
            )?;
            chunk_ids.push(cid);
        }

        // KNOWLEDGE ENRICHMENT ----------------------------------------------
        if cfg.enable_knowledge_extraction {
            for (i, cid) in chunk_ids.iter().enumerate() {
                let ko = crate::knowledge::extract_knowledge(
                    &proposed[i].text,
                    &|_| None, // df already applied at insert time for ranking; enrichment uses raw keywords
                    1,
                );
                let mentioned =
                    crate::entities::persist_entities(store.conn(), doc_id, *cid, &ko.entities)?;
                crate::entities::link_cooccurrences(store.conn(), doc_id, &mentioned)?;
                let (claims_stored, _conflicts) = crate::claims::persist_claims(
                    store.conn(),
                    doc_id,
                    *cid,
                    &ko.claims,
                )?;
                if claims_stored > 0 {
                    self.inner
                        .bus
                        .publish("claim.extracted", doc_id, &format!("{claims_stored} claims in chunk {cid}"));
                }
                if !mentioned.is_empty() {
                    self.inner
                        .bus
                        .publish("entity.discovered", doc_id, &format!("{} entities in chunk {cid}", mentioned.len()));
                }
            }
        }

        // Keyword document-frequency counters (after all chunks stored).
        all_keywords.sort();
        all_keywords.dedup();
        dao::bump_term_df(store.conn(), &all_keywords)?;

        let section_count = sections_seen.len() as i64;
        dao::finalize_index(store.conn(), doc_id, chunk_ids.len() as i64, section_count)?;
        dao::set_readiness(store.conn(), doc_id, ReadinessState::Ready)?;
        self.inner
            .bus
            .publish("document.ready", doc_id, &doc.filename);

        // OPTIONAL LLM SUMMARY ----------------------------------------------
        let llm = self
            .inner
            .llm
            .read()
            .expect("llm lock")
            .clone();
        if let Some(provider) = llm {
            if provider.name() != "null" && chunk_ids.len() >= 2 {
                dao::set_readiness(store.conn(), doc_id, ReadinessState::Summarizing)?;
                self.inner
                    .bus
                    .publish("document.status", doc_id, "summarizing");
                let summary_input: String = {
                    let mut s = String::new();
                    for p in proposed.iter().take(10) {
                        s.push_str(&p.text);
                        s.push('\n');
                        if s.len() > 3000 {
                            break;
                        }
                    }
                    s.chars().take(3000).collect()
                };
                let prompt = format!(
                    "Summarize the following document in 3-4 factual sentences. Plain text only.\n\nDOCUMENT:\n{summary_input}\n\nSummary:"
                );
                match provider.generate(&prompt, cfg.summary_max_tokens, cfg.summary_temperature) {
                    Ok(summary) => {
                        dao::set_summary(
                            store.conn(),
                            doc_id,
                            summary.trim(),
                            &dao::now(),
                        )?;
                        dao::insert_provenance(
                            store.conn(),
                            "summary",
                            &doc_id.to_string(),
                            Some(doc_id),
                            None,
                            None,
                            provider.name(),
                            crate::knowledge::KNOWLEDGE_VERSION,
                        )?;
                        dao::set_readiness(store.conn(), doc_id, ReadinessState::Complete)?;
                        self.inner
                            .bus
                            .publish("document.summary-ready", doc_id, &doc.filename);
                    }
                    Err(_) => {
                        // Summary failure must not fail ingestion: stay `ready`.
                        dao::set_readiness(store.conn(), doc_id, ReadinessState::Ready)?;
                        self.inner
                            .bus
                            .publish("document.status", doc_id, "summary-failed");
                    }
                }
            }
        }
        self.inner
            .bus
            .publish("knowledge.updated", doc_id, &doc.filename);
        Ok(())
    }

    // ------------------------------------------------------------------
    // querying
    // ------------------------------------------------------------------

    /// Full query path: plan → retrieve → assemble → explain.
    pub fn query(&self, req: QueryRequest) -> Result<QueryResponse> {
        let started = Instant::now();
        if req.text.trim().is_empty() {
            return Err(LkosError::InvalidQuery("empty query".into()));
        }
        let store = Store::open(&self.inner.path)?;
        let cfg = &self.inner.config;
        let plan = crate::query::plan(&req, cfg);

        // Known entities present in the query text.
        let lower_q = req.text.to_lowercase();
        let all_ents = dao::all_entities_min(store.read())?;
        let known_entities: Vec<(String, i64)> = all_ents
            .into_iter()
            .filter(|(name, _)| name.len() > 2 && lower_q.contains(&name.to_lowercase()))
            .take(20)
            .collect();

        // Summary fast path.
        if plan.use_summary {
            if let Some(doc_ids) = &req.filters.document_ids {
                if doc_ids.len() == 1 {
                    let doc = dao::get_document(store.read(), doc_ids[0])?;
                    if let Some(summary) = doc.summary {
                        return Ok(QueryResponse {
                            intent: plan.intent,
                            plan_explanation: format!(
                                "summary fast-path: pre-built summary returned with no retrieval ({}): {}",
                                plan.explanation, plan.explanation
                            ),
                            hits: Vec::new(),
                            context: summary.clone(),
                            provenance: Vec::new(),
                            related_entities: crate::storage::dao::entities_for_document(
                                store.read(),
                                doc.id,
                            )?,
                            elapsed_us: started.elapsed().as_micros(),
                        });
                    }
                }
            }
        }

        let mut hits = crate::retrieval::hybrid_search(
            store.read(),
            self.inner.embedder.as_ref(),
            &req.text,
            req.top_k,
            plan.mode,
            Some(&req.filters),
            cfg.rrf_k,
            plan.w_vector,
            plan.w_fts,
            cfg.mmr_lambda,
            cfg.max_dense_scan,
            &known_entities,
        )?;

        // Temporal post-filtering with graceful fallback.
        if !matches!(
            plan.temporal,
            crate::temporal::TemporalConstraint::None
        ) {
            let kept: Vec<SearchHit> = hits
                .iter()
                .filter(|h| {
                    crate::temporal::matches_constraint(&h.text, &plan.temporal)
                })
                .cloned()
                .collect();
            if !kept.is_empty() {
                hits = kept;
            }
        }

        let context = crate::query::assemble_context(&hits, req.context_budget, req.max_per_document);

        let mut provenance = Vec::new();
        if req.include_provenance {
            for h in &hits {
                let doc = dao::get_document(store.read(), h.document_id)?;
                let prov = dao::provenance_for(store.read(), "chunk", &h.chunk_id.to_string())?;
                let first = prov.first();
                provenance.push(HitProvenance {
                    artifact_type: "chunk".into(),
                    document: doc.filename,
                    content_hash: doc.content_hash,
                    extractor: first
                        .map(|p| p.extractor.clone())
                        .unwrap_or_else(|| "unknown".into()),
                    offsets: first.map(|p| p.offsets.unwrap_or((0, 0))).unwrap_or((0, 0)),
                });
            }
        }

        let mut related_entities = Vec::new();
        if req.include_graph || plan.use_entities {
            let mut seen = std::collections::HashSet::new();
            for h in hits.iter().take(6) {
                for eid in dao::entity_ids_for_chunk(store.read(), h.chunk_id)? {
                    if seen.insert(eid) {
                        if let Ok(rec) = dao::get_entity(store.read(), eid) {
                            related_entities.push(EntitySummary {
                                id: rec.id,
                                display_name: rec.display_name,
                                entity_type: rec.entity_type,
                                mention_count: rec.mention_count,
                            });
                        }
                    }
                }
            }
        }

        Ok(QueryResponse {
            intent: plan.intent,
            plan_explanation: plan.explanation,
            hits,
            context,
            provenance,
            related_entities,
            elapsed_us: started.elapsed().as_micros(),
        })
    }

    /// Grounded answering via the configured LLM (optional capability).
    pub fn ask(&self, question: &str) -> Result<String> {
        let llm = self
            .inner
            .llm
            .read()
            .expect("llm lock")
            .clone()
            .ok_or_else(|| {
                LkosError::Llm(
                    "no LLM provider configured; use `query()` for retrieval-only answers"
                        .into(),
                )
            })?;
        let req = QueryRequest::new(question).top_k(8);
        let resp = self.query(req)?;
        if resp.hits.is_empty() {
            return Ok("I cannot find this in your documents.".into());
        }
        let prompt = grounded_prompt(question, &resp.context);
        Ok(llm.generate(&prompt, 512, 0.2)?.trim().to_string())
    }

    /// Get the stored summary of a document (pre-built fast path).
    pub fn summary(&self, doc_id: i64) -> Result<Option<String>> {
        let store = Store::open(&self.inner.path)?;
        Ok(dao::get_document(store.read(), doc_id)?.summary)
    }

    // ------------------------------------------------------------------
    // library management
    // ------------------------------------------------------------------

    /// List all documents.
    pub fn documents(&self) -> Result<Vec<DocumentInfo>> {
        let store = Store::open(&self.inner.path)?;
        dao::list_documents(store.read())
    }

    /// Get one document.
    pub fn document(&self, id: i64) -> Result<DocumentInfo> {
        let store = Store::open(&self.inner.path)?;
        dao::get_document(store.read(), id)
    }

    /// Delete a document and all derived knowledge.
    pub fn delete_document(&self, id: i64) -> Result<()> {
        let mut store = Store::open(&self.inner.path)?;
        // Decrement entity mention counts and df counters first.
        let entity_ids = dao::distinct_entities_for_doc(store.read(), id)?;
        for eid in &entity_ids {
            dao::bump_entity_mentions(store.conn(), *eid, -1)?;
        }
        let doc = dao::get_document(store.read(), id)?;
        let _ = &doc;
        // Terms: rebuild from chunk knowledge payloads.
        let mut terms = Vec::new();
        for c in dao::chunks_for_document(store.read(), id)? {
            if let Ok(Some(kj)) = store.read().query_row(
                "SELECT knowledge_json FROM chunks WHERE id = ?1",
                rusqlite::params![c.id],
                |r| r.get::<_, Option<String>>(0),
            ) {
                if let Some(ko) = crate::knowledge::ko_from_json(&kj) {
                    for (kw, _) in ko.keywords {
                        terms.push(kw);
                    }
                }
            }
        }
        terms.sort();
        terms.dedup();
        dao::drop_term_df(store.conn(), &terms)?;
        dao::delete_mentions_for_doc(store.conn(), id)?;
        dao::delete_claims_for_doc(store.conn(), id)?;
        let deleted = dao::delete_document(store.conn(), id)?;
        if !deleted {
            return Err(LkosError::DocumentNotFound(id));
        }
        self.inner.bus.publish("document.deleted", id, "");
        Ok(())
    }

    /// Entities of a document.
    pub fn document_entities(&self, doc_id: i64) -> Result<Vec<EntitySummary>> {
        let store = Store::open(&self.inner.path)?;
        dao::entities_for_document(store.read(), doc_id)
    }

    /// Claims of a document.
    pub fn document_claims(&self, doc_id: i64) -> Result<Vec<ClaimRecord>> {
        let store = Store::open(&self.inner.path)?;
        dao::claims_for_document(store.read(), doc_id)
    }

    /// Claims about a subject (canonicalized).
    pub fn claims_about(&self, subject: &str) -> Result<Vec<ClaimRecord>> {
        let store = Store::open(&self.inner.path)?;
        dao::claims_for_subject(store.read(), &crate::entities::canonical_key(subject))
    }

    /// Recent detected conflicts.
    pub fn conflicts(&self, limit: usize) -> Result<Vec<ConflictRecord>> {
        let store = Store::open(&self.inner.path)?;
        dao::list_conflicts(store.read(), limit)
    }

    /// One-hop entity graph neighborhood.
    pub fn neighborhood(&self, entity_id: i64, limit: usize) -> Result<crate::graph::Neighborhood> {
        let store = Store::open(&self.inner.path)?;
        crate::graph::neighborhood(store.read(), entity_id, limit)
    }

    /// Top entities by mention count.
    pub fn list_entities(&self, limit: usize) -> Result<Vec<EntityRecord>> {
        let store = Store::open(&self.inner.path)?;
        dao::list_entities(store.read(), limit)
    }

    /// Resolve an entity id by name (display or canonical).
    pub fn entity_id_by_name(&self, name: &str) -> Result<Option<i64>> {
        let store = Store::open(&self.inner.path)?;
        dao::entity_id_by_name(store.read(), name)
    }

    /// Provenance trail of an artifact.
    pub fn provenance_of(&self, artifact_type: &str, artifact_id: &str) -> Result<Vec<ProvenanceRecord>> {
        let store = Store::open(&self.inner.path)?;
        dao::provenance_for(store.read(), artifact_type, artifact_id)
    }

    /// Aggregate statistics.
    pub fn stats(&self) -> Result<LibraryStats> {
        let store = Store::open(&self.inner.path)?;
        let c = store.read();
        let count = |sql: &str| -> Result<i64> {
            Ok(c.query_row(sql, [], |r| r.get(0))?)
        };
        Ok(LibraryStats {
            documents: count("SELECT COUNT(*) FROM documents")?,
            chunks: count("SELECT COUNT(*) FROM chunks")?,
            entities: count("SELECT COUNT(*) FROM entities")?,
            entity_mentions: count("SELECT COUNT(*) FROM entity_mentions")?,
            claims: count("SELECT COUNT(*) FROM claims")?,
            conflicts: count("SELECT COUNT(*) FROM claim_conflicts")?,
            relationships: count("SELECT COUNT(*) FROM relationships")?,
            provenance_records: count("SELECT COUNT(*) FROM provenance")?,
            db_size_bytes: store.file_size_bytes(),
        })
    }

    /// Online backup to a destination path.
    pub fn backup_to(&self, dest: impl AsRef<std::path::Path>) -> Result<()> {
        let store = Store::open(&self.inner.path)?;
        store.backup_to(dest)
    }

    /// Run a database integrity check.
    pub fn integrity_check(&self) -> Result<Vec<String>> {
        let store = Store::open(&self.inner.path)?;
        store.integrity_check()
    }

    fn start_worker(&self) {
        let engine = self.clone();
        let shutdown = self.inner.shutdown.clone();
        let handle = std::thread::spawn(move || loop {
            if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            let job = {
                match Store::open(&engine.inner.path) {
                    Ok(mut store) => match crate::jobs::claim_next(store.conn()) {
                        Ok(Some(job)) => Some(job),
                        Ok(None) => None,
                        Err(_) => None,
                    },
                    Err(_) => None,
                }
            };
            match job {
                Some(job) => {
                    let doc_id: Option<i64> = serde_json::from_str::<serde_json::Value>(&job.payload)
                        .ok()
                        .and_then(|v| v.get("document_id").and_then(|d| d.as_i64()));
                    let result = match (job.kind.as_str(), doc_id) {
                        ("process_document", Some(id)) => engine.process_document(id),
                        ("delete_document", Some(id)) => engine.delete_document(id),
                        _ => Err(LkosError::Other(format!("unknown job kind {}", job.kind))),
                    };
                    let mut store = match Store::open(&engine.inner.path) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    match result {
                        Ok(_) => {
                            let _ = crate::jobs::complete(store.conn(), &job);
                        }
                        Err(e) => {
                            let _ = crate::jobs::fail_and_maybe_retry(store.conn(), &job, &e.to_string());
                        }
                    }
                }
                None => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        });
        if let Ok(mut w) = self.inner.worker.lock() {
            *w = Some(handle);
        }
    }
}
