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

use crate::ann::{AnnCacheEntry, HnswIndex};
use crate::chunking;
use crate::config::Config;
use crate::embeddings::{EmbeddingProvider, HashingEmbedder, LsaEmbedder};
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
    embedder: Arc<dyn EmbeddingProvider>,
    /// Present when `embedding_provider = "lsa"` (dual-mode provider).
    lsa: Option<Arc<LsaEmbedder>>,
    bus: EventBus,
    llm: RwLock<Option<Arc<dyn LlmProvider>>>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    /// Directory to remove on close (in-memory scratch), if any.
    scratch_dir: Option<std::path::PathBuf>,
    /// Cached deterministic HNSW index over the current embedding model's
    /// chunks (v0.10; see `ann` module docs and ADR-010). Rebuilt lazily
    /// when the (model, COUNT, MAX(id)) fingerprint changes.
    ann: Mutex<Option<AnnCacheEntry>>,
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
        let lsa = if config.embedding_provider == "lsa" {
            Some(Arc::new(LsaEmbedder::new(config.embedding_dim)))
        } else {
            None
        };
        let embedder: Arc<dyn EmbeddingProvider> = match &lsa {
            Some(l) => l.clone(),
            None => Arc::new(HashingEmbedder::new(config.embedding_dim)),
        };
        {
            let store = Store::open(&path)?;
            // Record / verify embedding model identity.
            let meta_model = dao::meta_get(store.read(), "embedding_model")?;
            match meta_model {
                None => {
                    dao::meta_set(store.read(), "embedding_model", &embedder.name())?;
                    dao::meta_set(store.read(), "embedding_dim", &embedder.dim().to_string())?;
                }
                Some(m) => {
                    // A trained LSA model restores the semantic space; without
                    // it the provider falls back to hashing — mismatch against
                    // an LSA-built index is a hard error (reindex required).
                    if let Some(l) = &lsa {
                        if let Some(model) = crate::embeddings::lsa::load_model(store.read())? {
                            l.install_model(model);
                        }
                    }
                    let active = match (&lsa, lsa.as_ref().map(|l| l.is_trained())) {
                        (Some(_), Some(true)) => "lsa-pmi-svd-v1",
                        _ => "hashing-lex-v1",
                    };
                    let meta_dim: usize = dao::meta_get(store.read(), "embedding_dim")?
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(config.embedding_dim);
                    if m != active || meta_dim != embedder.dim() {
                        return Err(LkosError::EmbeddingMismatch {
                            index_model: m,
                            index_dim: meta_dim,
                            provider_model: embedder.name().to_string(),
                            provider_dim: embedder.dim(),
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
                embedder,
                lsa,
                bus: EventBus::new(),
                llm: RwLock::new(None),
                workers: Mutex::new(Vec::new()),
                shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                scratch_dir: None,
                ann: Mutex::new(None),
            }),
        };
        if !engine.inner.config.synchronous_ingestion {
            engine.start_workers();
        }
        Ok(engine)
    }

    /// Open with an in-memory-backed scratch database (tests/examples).
    /// The scratch directory is removed on [`Lkos::close`].
    pub fn open_in_memory(config: Config) -> Result<Lkos> {
        // Uniqueness contract: process id + a process-global monotonically
        // increasing counter. A wall-clock timestamp alone is NOT unique —
        // on Windows its effective granularity let two parallel tests obtain
        // the same scratch path and migrate the same fresh database
        // concurrently ("duplicate column name"). A counter cannot collide
        // within a process; the pid separates processes.
        use std::sync::atomic::{AtomicU64, Ordering};
        static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("lkos-mem-{}-{}", std::process::id(), seq));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("mem.lkos");
        let mut engine = Self::open(path, config)?;
        // Shared inner state: patch the scratch dir through Arc::get_mut.
        // (open_in_memory is only used before clones exist.)
        if let Some(inner) = Arc::get_mut(&mut engine.inner) {
            inner.scratch_dir = Some(dir);
        }
        Ok(engine)
    }

    /// Install an LLM provider (optional capability).
    pub fn set_llm(&self, provider: Arc<dyn LlmProvider>) {
        *self.inner.llm.write().expect("llm lock") = Some(provider);
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
        if let Ok(mut w) = self.inner.workers.lock() {
            for handle in w.drain(..) {
                let _ = handle.join();
            }
        }
        // Cancel any jobs still queued (workers are gone).
        if let Ok(mut store) = Store::open(&self.inner.path) {
            let _ = crate::jobs::cancel_pending(store.conn());
        }
        if let Some(dir) = &self.inner.scratch_dir {
            let _ = std::fs::remove_dir_all(dir);
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
            self.inner.bus.publish("document.queued", doc_id, filename);
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
            self.inner
                .bus
                .publish("document.queued", doc_id, &extracted.filename);
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
            &self.inner.embedder.name(),
            crate::knowledge::KNOWLEDGE_VERSION,
        )?;
        if created {
            self.inner
                .bus
                .publish("document.added", doc_id, &extracted.filename);
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
                &self.inner.embedder.name(),
                &crate::knowledge::ko_to_json(&ko),
            )?;
            dao::insert_provenance(
                store.conn(),
                "chunk",
                &cid.to_string(),
                Some(doc_id),
                Some(cid),
                Some((p.start_offset as i64, p.end_offset as i64)),
                &format!(
                    "{} -> {} -> {}",
                    ingestion::EXTRACTOR_VERSION,
                    chunking::CHUNKER_VERSION,
                    self.inner.embedder.name()
                ),
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
                let (claims_stored, _conflicts) =
                    crate::claims::persist_claims(store.conn(), doc_id, *cid, &ko.claims)?;
                if claims_stored > 0 {
                    self.inner.bus.publish(
                        "claim.extracted",
                        doc_id,
                        &format!("{claims_stored} claims in chunk {cid}"),
                    );
                }
                if !mentioned.is_empty() {
                    self.inner.bus.publish(
                        "entity.discovered",
                        doc_id,
                        &format!("{} entities in chunk {cid}", mentioned.len()),
                    );
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
        let llm = self.inner.llm.read().expect("llm lock").clone();
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
                        dao::set_summary(store.conn(), doc_id, summary.trim(), &dao::now())?;
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

    /// Resolve the dense-channel ANN option for a query.
    ///
    /// Returns `Some((index, ef_search))` when the configured `ann_mode`
    /// policy wants the HNSW path for the current corpus state, else `None`
    /// (brute force). The cache is validated against the
    /// `(model, COUNT, MAX(chunk id))` fingerprint — exact because chunk
    /// vectors are immutable within a model name (written at insert or at
    /// model-change migration only; see the `ann` module docs and ADR-010).
    fn resolve_ann_index(
        &self,
        conn: &rusqlite::Connection,
        unfiltered: bool,
        dense_used: bool,
    ) -> Result<Option<(Arc<HnswIndex>, usize)>> {
        let cfg = &self.inner.config;
        if !dense_used || !unfiltered {
            return Ok(None);
        }
        let mode = cfg.ann_mode.trim().to_lowercase();
        if mode == "brute" {
            return Ok(None);
        }
        let model = self.inner.embedder.name().into_owned();
        let (count, max_id): (i64, i64) = conn.query_row(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks \
             WHERE embedding_model = ?1 AND embedding IS NOT NULL",
            rusqlite::params![model],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        if count == 0 {
            return Ok(None);
        }
        // `auto`: use ANN at the measured crossover, or degrade-instead-of-
        // refuse past the brute-force cap. `hnsw`: always.
        let wants_ann = mode == "hnsw"
            || count >= cfg.ann_min_chunks as i64
            || count > cfg.max_dense_scan as i64;
        if !wants_ann {
            return Ok(None);
        }

        let mut cache = self.inner.ann.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = cache.as_ref() {
            if entry.model == model && entry.count == count && entry.max_id == max_id {
                return Ok(Some((entry.index.clone(), cfg.ann_ef_search)));
            }
        }
        // Build (or rebuild after invalidation). The fetch + build happen
        // under the cache lock: rebuild cost is amortized across queries and
        // documented in benchmarks/results/ann-crossover-*.txt (ADR-010).
        let corpus = dao::all_embeddings(conn, None, &model)?;
        // M=16, ef_construction=200: measured on the v0.10 crossover bench
        // (recall@10 >= 0.99 at ef_search=128 on 100k clustered chunks;
        // see benchmarks/results/ann-crossover-*.txt and ADR-010).
        let index = Arc::new(HnswIndex::build(corpus, 16, 200));
        *cache = Some(AnnCacheEntry {
            model,
            count,
            max_id,
            index: index.clone(),
        });
        Ok(Some((index, cfg.ann_ef_search)))
    }

    /// Full query path: plan → retrieve → assemble → explain.
    #[allow(clippy::too_many_lines)]
    pub fn query(&self, req: QueryRequest) -> Result<QueryResponse> {
        let started = Instant::now();
        if req.text.trim().is_empty() {
            return Err(LkosError::InvalidQuery("empty query".into()));
        }
        let store = Store::open(&self.inner.path)?;
        let cfg = &self.inner.config;
        let plan = crate::query::plan(&req, cfg);

        // Known entities present in the query text (SQL LIKE prefilter —
        // v0.1 loaded the whole entity table and substring-scanned in Rust).
        let mut known_entities: Vec<(String, i64)> = Vec::new();
        for token_raw in req.text.split_whitespace() {
            let token = token_raw.trim_matches(|c: char| !c.is_alphanumeric());
            if token.len() < 3 {
                continue;
            }
            for summary in dao::find_entities_by_name(store.read(), token)? {
                let name = summary.display_name;
                if name.len() > 2
                    && req.text.to_lowercase().contains(&name.to_lowercase())
                    && !known_entities.iter().any(|(_, eid)| *eid == summary.id)
                {
                    known_entities.push((name, summary.id));
                }
            }
            if known_entities.len() >= 20 {
                break;
            }
        }

        // Summary fast path.
        if plan.use_summary {
            if let Some(doc_ids) = &req.filters.document_ids {
                if doc_ids.len() == 1 {
                    let doc = dao::get_document(store.read(), doc_ids[0])?;
                    if let Some(summary) = doc.summary {
                        return Ok(QueryResponse {
                            intent: plan.intent,
                            plan_explanation: format!(
                                "summary fast-path: pre-built summary returned with no retrieval ({})",
                                plan.explanation
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

        let dense_used = matches!(
            plan.mode,
            RetrievalMode::Auto | RetrievalMode::Hybrid | RetrievalMode::VectorOnly
        );
        let ann =
            self.resolve_ann_index(store.read(), req.filters.is_effectively_empty(), dense_used)?;
        let ann_ref = ann.as_ref().map(|(idx, ef)| (idx, *ef));

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
            ann_ref,
            &known_entities,
            cfg.enable_reranking,
            cfg.rerank_top_n,
        )?;

        // Temporal handling: `Latest` re-ranks (boost recent); Year/Range
        // hard-filter with graceful fallback (v0.1 filtered for both, which
        // could drop the newest evidence for "latest" queries).
        match &plan.temporal {
            crate::temporal::TemporalConstraint::None => {}
            crate::temporal::TemporalConstraint::Latest => {
                if cfg.freshness_boost > 0.0 {
                    for h in &mut hits {
                        let s = crate::temporal::temporal_score(&h.text, &plan.temporal);
                        if s > 0.0 {
                            h.score += cfg.freshness_boost * s;
                        }
                    }
                    hits.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
                    });
                    for (i, h) in hits.iter_mut().enumerate() {
                        h.rank = i + 1;
                    }
                }
            }
            _ => {
                let kept: Vec<SearchHit> = hits
                    .iter()
                    .filter(|h| crate::temporal::matches_constraint(&h.text, &plan.temporal))
                    .cloned()
                    .collect();
                if !kept.is_empty() {
                    hits = kept;
                }
            }
        }

        let context =
            crate::query::assemble_context(&hits, req.context_budget, req.max_per_document);

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
                    "no LLM provider configured; use `query()` for retrieval-only answers".into(),
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

    /// Delete a document and all derived knowledge, including graph edges
    /// that lost their last evidence (v0.1 left stale co-occurrence edges).
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
        // Rebuild graph edges around every affected entity from surviving
        // evidence (mentions deleted above, so lost evidence is excluded).
        dao::recompute_relationships_for_entities(store.conn(), &entity_ids)?;
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
    pub fn provenance_of(
        &self,
        artifact_type: &str,
        artifact_id: &str,
    ) -> Result<Vec<ProvenanceRecord>> {
        let store = Store::open(&self.inner.path)?;
        dao::provenance_for(store.read(), artifact_type, artifact_id)
    }

    /// Aggregate statistics.
    pub fn stats(&self) -> Result<LibraryStats> {
        let store = Store::open(&self.inner.path)?;
        let c = store.read();
        let count = |sql: &str| -> Result<i64> { Ok(c.query_row(sql, [], |r| r.get(0))?) };
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

    // ------------------------------------------------------------------
    // semantic index (LSA) lifecycle
    // ------------------------------------------------------------------

    /// Train (or retrain) the corpus-trained semantic model from all stored
    /// chunk texts and install it. Returns `true` when a model was trained.
    ///
    /// No-op when the corpus is below [`Config::semantic_min_chunks`] or too
    /// small to factor. On success the persisted model is reloaded on every
    /// subsequent [`Lkos::open`], and stale chunks are re-embedded by
    /// [`Lkos::reembed_stale_chunks`] (call this afterwards, or enqueue a
    /// `train_semantic` job in background mode, which chains both).
    pub fn train_semantic_index(&self) -> Result<bool> {
        let Some(lsa) = &self.inner.lsa else {
            return Ok(false); // provider is hashing; nothing to train
        };
        let mut store = Store::open(&self.inner.path)?;
        let n_chunks: i64 = store
            .read()
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        if (n_chunks as usize) < self.inner.config.semantic_min_chunks {
            return Ok(false);
        }
        // Stream chunk texts in id order (deterministic training input).
        let mut stmt = store
            .read()
            .prepare("SELECT text FROM chunks ORDER BY id")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let texts: Vec<String> = rows.filter_map(|r| r.ok()).collect();
        drop(stmt);
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let Some(model) = crate::embeddings::lsa::train(
            &refs,
            self.inner.config.lsa_dim,
            self.inner.config.lsa_min_df,
            self.inner.config.lsa_max_vocab,
            4,
        ) else {
            return Ok(false);
        };
        crate::embeddings::lsa::save_model(store.conn(), &model)?;
        dao::meta_set(
            store.conn(),
            "embedding_model",
            crate::embeddings::lsa::LSA_MODEL_NAME,
        )?;
        dao::meta_set(store.conn(), "embedding_dim", &model.dim.to_string())?;
        lsa.install_model(model);
        self.inner.bus.publish(
            "semantic.trained",
            0,
            crate::embeddings::lsa::LSA_MODEL_NAME,
        );
        Ok(true)
    }

    /// Re-embed chunks whose embedding model differs from `target` (batched).
    /// Returns the number of chunks migrated. `job_id` enables progress
    /// reporting + cooperative cancellation when invoked from a worker.
    pub fn reembed_stale_chunks(&self, target: &str, job_id: Option<i64>) -> Result<usize> {
        let mut migrated = 0usize;
        const BATCH: usize = 256;
        loop {
            if let Some(jid) = job_id {
                let store = Store::open(&self.inner.path)?;
                if crate::jobs::is_cancelled(store.read(), jid)? {
                    return Err(LkosError::Cancelled);
                }
            }
            let stale = {
                let store = Store::open(&self.inner.path)?;
                dao::stale_embedding_chunks(store.read(), target, BATCH)?
            };
            if stale.is_empty() {
                break;
            }
            {
                let mut store = Store::open(&self.inner.path)?;
                let mut texts: Vec<String> = Vec::with_capacity(stale.len());
                for cid in &stale {
                    let t: String = store.read().query_row(
                        "SELECT text FROM chunks WHERE id = ?1",
                        rusqlite::params![cid],
                        |r| r.get(0),
                    )?;
                    texts.push(t);
                }
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let vectors = self.inner.embedder.embed_batch(&refs)?;
                for (cid, v) in stale.iter().zip(vectors) {
                    dao::update_chunk_embedding(
                        store.conn(),
                        *cid,
                        &self.inner.embedder.name(),
                        &dao::f32_to_bytes(&v),
                    )?;
                }
            }
            migrated += stale.len();
            if let Some(jid) = job_id {
                if let Ok(mut store) = Store::open(&self.inner.path) {
                    let total: i64 = store
                        .read()
                        .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
                        .unwrap_or(1);
                    let _ = crate::jobs::progress(
                        store.conn(),
                        jid,
                        ((migrated as f64 / total.max(1) as f64) * 100.0) as i64,
                    );
                }
            }
            if stale.len() < BATCH {
                break;
            }
        }
        Ok(migrated)
    }

    /// Check whether the semantic index needs (re)training given the current
    /// share of stale-model chunks. Returns the number of stale chunks.
    pub fn stale_embedding_count(&self) -> Result<usize> {
        let store = Store::open(&self.inner.path)?;
        Ok(
            dao::stale_embedding_chunks(store.read(), &self.inner.embedder.name(), usize::MAX)?
                .len(),
        )
    }

    // ------------------------------------------------------------------
    // entity resolution management
    // ------------------------------------------------------------------

    /// Register a user-supplied alias for an entity (e.g. "MSFT" →
    /// "Microsoft"). Future mentions of the alias resolve to the entity.
    pub fn add_entity_alias(&self, entity_id: i64, alias: &str) -> Result<()> {
        let mut store = Store::open(&self.inner.path)?;
        dao::add_alias(store.conn(), entity_id, alias)?;
        // Alias surfaces also resolve via the alias table at extraction time.
        Ok(())
    }

    /// Merge two entities (e.g. a false split). `survivor` keeps its identity;
    /// `merged`'s aliases, mentions, counts and edges are transferred and an
    /// audit row is written. Graph edges around both entities are recomputed.
    pub fn merge_entities(&self, survivor: i64, merged: i64, reason: &str) -> Result<()> {
        let mut store = Store::open(&self.inner.path)?;
        dao::merge_entity_rows(store.conn(), survivor, merged, reason)?;
        dao::recompute_relationships_for_entities(store.conn(), &[survivor])?;
        self.inner.bus.publish(
            "entity.merged",
            survivor,
            &format!("absorbed {merged}: {reason}"),
        );
        Ok(())
    }

    fn start_workers(&self) {
        let n = self.inner.config.worker_threads.max(1);
        for _ in 0..n {
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
                        let payload: serde_json::Value =
                            serde_json::from_str(&job.payload).unwrap_or_default();
                        let doc_id = payload.get("document_id").and_then(|d| d.as_i64());
                        let model = payload.get("model").and_then(|m| m.as_str()).unwrap_or("");
                        let result = match (job.kind.as_str(), doc_id) {
                            ("process_document", Some(id)) => engine.process_document(id),
                            ("delete_document", Some(id)) => engine.delete_document(id),
                            ("reembed_stale", _) => {
                                engine.reembed_stale_chunks(model, Some(job.id)).map(|_| ())
                            }
                            ("train_semantic", _) => {
                                engine.train_semantic_index().and_then(|trained| {
                                    if trained {
                                        engine
                                            .reembed_stale_chunks("lsa-pmi-svd-v1", Some(job.id))
                                            .map(|_| ())
                                    } else {
                                        Ok(())
                                    }
                                })
                            }
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
                                let _ = crate::jobs::fail_and_maybe_retry(
                                    store.conn(),
                                    &job,
                                    &e.to_string(),
                                    engine.inner.config.job_backoff_base_secs,
                                    engine.inner.config.job_backoff_max_secs,
                                );
                            }
                        }
                    }
                    None => {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                }
            });
            if let Ok(mut w) = self.inner.workers.lock() {
                w.push(handle);
            }
        }
    }
}
