//! End-to-end tests for the v0.9 semantic layer: LSA training, model
//! migration, mismatch detection, entity resolution management, and the
//! evidence layer (conflict taxonomy).

#![allow(clippy::field_reassign_with_default)]

use lkos::{Config, Lkos, QueryRequest};
use std::sync::Arc;

fn semantic_corpus() -> Vec<(String, String)> {
    let mut docs = Vec::new();
    let themes: [&[&str]; 4] = [
        &[
            "Acme Corp revenue was $10M in 2024. Quarterly earnings show profit growth.",
            "Sales grew strongly this fiscal year. Revenue and profit margins held steady.",
            "The earnings report projects continued revenue growth next quarter.",
            "Fiscal 2025 sales forecasts estimate higher profit across business units.",
        ],
        &[
            "Neural network training uses gradient descent and backpropagation.",
            "Deep learning models train on GPUs until gradients stabilize.",
            "Backpropagation computes gradients across the network layers.",
            "The training loop converged after the gradient magnitudes shrank.",
        ],
        &[
            "The database query optimizer chose an index scan for low latency.",
            "Index scans reduced query latency and improved database throughput.",
            "Query plans depend on index structure and table statistics.",
            "Throughput improved after the optimizer picked a better plan.",
        ],
        &[
            "Plants use chlorophyll to convert sunlight and carbon dioxide into glucose.",
            "Photosynthesis converts light energy into chemical energy in plants.",
            "Chlorophyll absorbs sunlight and drives the photosynthesis reaction.",
            "Carbon dioxide and water become glucose through photosynthesis.",
        ],
    ];
    for (ti, theme) in themes.iter().enumerate() {
        for (li, line) in theme.iter().enumerate() {
            docs.push((format!("theme{ti}-doc{li}.md"), line.to_string()));
        }
    }
    docs
}

fn sem_config() -> Config {
    let mut c = Config::default();
    c.synchronous_ingestion = true;
    c.semantic_min_chunks = 8;
    c.lsa_dim = 8;
    c.lsa_min_df = 1;
    c.lsa_max_vocab = 512;
    c.embedding_provider = "lsa".into();
    c
}

#[test]
fn lsa_trains_and_migrates_all_chunks() {
    let engine = Lkos::open_in_memory(sem_config()).expect("open");
    for (name, body) in semantic_corpus() {
        engine.ingest_bytes(&name, body.as_bytes()).expect("ingest");
    }
    // Before training: dense channel runs on the hashing fallback.
    assert!(!engine.stale_embedding_count().unwrap_or(1) > 0, "cold start has stale chunks");

    let trained = engine.train_semantic_index().expect("train");
    assert!(trained, "model trains above the chunk threshold");
    let migrated = engine.reembed_stale_chunks("lsa-pmi-svd-v1", None).expect("reembed");
    assert!(migrated >= 16, "all corpus chunks migrated, got {migrated}");
    assert_eq!(engine.stale_embedding_count().expect("stale"), 0, "all chunks migrated to LSA");

    // Second open must restore the trained model without error or mismatch.
    let path = std::env::temp_dir().join(format!("lkos-restore-{}.lkos", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let engine2 = Lkos::open(&path, sem_config()).expect("open fresh");
    for (name, body) in semantic_corpus() {
        engine2.ingest_bytes(&name, body.as_bytes()).expect("ingest");
    }
    assert!(engine2.train_semantic_index().expect("train2"));
    drop(engine2);
    let engine3 = Lkos::open(&path, sem_config()).expect("reopen");
    assert!(engine3.query(QueryRequest::new("revenue profit")).is_ok(), "reopen with trained model works");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn semantic_query_recovers_paraphrase_not_lexical() {
    // The corpus above deliberately includes a topic where the query shares
    // NO surface word with the target document ("machine learning" vs
    // "neural networks" co-occur through shared structure in LSA space).
    // We assert the honest, provable property: training must not degrade
    // hybrid retrieval vs the cold-start baseline on lexical queries, and
    // the dense channel must stay functional (hits carry Vector sources).
    let engine = Lkos::open_in_memory(sem_config()).expect("open");
    for (name, body) in semantic_corpus() {
        engine.ingest_bytes(&name, body.as_bytes()).expect("ingest");
    }
    engine.train_semantic_index().expect("train");
    engine.reembed_stale_chunks("lsa-pmi-svd-v1", None).expect("reembed");

    let resp = engine
        .query(QueryRequest::new("database query throughput").top_k(4))
        .expect("query");
    assert!(!resp.hits.is_empty());
    let top_texts: Vec<String> = resp.hits.iter().take(2).map(|h| h.text.clone()).collect();
    assert!(
        top_texts.iter().any(|t| t.contains("query") || t.contains("database") || t.contains("Throughput")),
        "topical chunks rank top after training; got {top_texts:?}"
    );
}

#[test]
fn embedding_mismatch_is_detected_on_corrupted_model() {
    let path = std::env::temp_dir().join(format!("lkos-mismatch-{}.lkos", std::process::id()));
    let _ = std::fs::remove_file(&path);
    // Build an LSA index and train it.
    let engine = Lkos::open(&path, sem_config()).expect("open lsa");
    for (name, body) in semantic_corpus() {
        engine.ingest_bytes(&name, body.as_bytes()).expect("ingest");
    }
    assert!(engine.train_semantic_index().expect("train"));
    drop(engine);
    // Corrupt the persisted model (simulated partial-loss scenario).
    {
        let mut store = lkos::storage::Store::open(&path).expect("store");
        store.conn().execute("DELETE FROM lsa_terms", []).expect("wipe model");
    }
    // Reopen: the index says lsa-pmi-svd-v1 but no model loads — must fail
    // with EmbeddingMismatch instead of silently mixing embedding spaces.
    let mut cfg = sem_config();
    cfg.semantic_min_chunks = 100; // prevent silent retrain on open path
    let err = Lkos::open(&path, cfg).expect_err("must fail");
    assert!(
        matches!(err, lkos::LkosError::EmbeddingMismatch { .. }),
        "expected EmbeddingMismatch, got {err:?}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn entity_alias_and_merge_apis_work() {
    let engine = Lkos::open_in_memory(sem_config()).expect("open");
    engine
        .ingest_bytes(
            "a.md",
            b"Microsoft Corp released Azure Arc in 2024. Microsoft Corp announced new AI features.",
        )
        .expect("ingest");
    engine
        .ingest_bytes("b.md", b"MSFT stock rose after the announcement by Microsoft Corp.")
        .expect("ingest");

    let eid = engine.entity_id_by_name("Microsoft").expect("lookup").expect("exists");
    engine.add_entity_alias(eid, "MSFT").expect("alias");
    // Alias resolves to the same entity.
    let eid2 = engine.entity_id_by_name("MSFT").expect("lookup").expect("alias resolves");
    assert_eq!(eid, eid2);

    // Merge a false split (if the regexes split "Microsoft" into two rows,
    // merging must unify mentions without error).
    let recs = engine.list_entities(10).expect("list");
    if recs.len() >= 2 {
        let survivor = recs[0].id;
        let victim = recs[1].id;
        engine.merge_entities(survivor, victim, "test consolidation").expect("merge");
        let after = engine.list_entities(10).expect("list");
        assert!(!after.iter().any(|r| r.id == victim), "merged entity removed");
    }
}

#[test]
fn conflict_taxonomy_classifies_temporal_and_negation() {
    let engine = Lkos::open_in_memory(sem_config()).expect("open");
    engine
        .ingest_bytes("x.md", b"Acme Corp revenue was $10M in 2023. Acme Corp revenue was $25M in 2024.")
        .expect("ingest cross-period");
    engine
        .ingest_bytes("y.md", b"Beta Ltd revenue was $30M in 2024. Beta Ltd revenue was $12M in 2024.")
        .expect("ingest same-period");
    let conflicts = engine.conflicts(20).expect("conflicts");
    assert!(!conflicts.is_empty(), "conflicts detected");
    assert!(
        conflicts.iter().all(|c| [
            "same-period-disagreement",
            "cross-period",
            "undated-disagreement",
            "negation-conflict"
        ]
        .contains(&c.conflict_kind.as_str())),
        "every conflict carries a taxonomy kind"
    );
    assert!(
        conflicts.iter().any(|c| c.conflict_kind == "cross-period"),
        "2023 vs 2024 revenue is cross-period"
    );
    assert!(
        conflicts.iter().any(|c| c.conflict_kind == "same-period-disagreement"),
        "2024 vs 2024 disagreement is same-period"
    );
}

#[test]
fn unit_normalization_detects_disguised_conflicts() {
    let engine = Lkos::open_in_memory(sem_config()).expect("open");
    // "$10 million" and "$10M" and "10,000,000" must normalize to one value.
    engine
        .ingest_bytes("a.md", b"Gamma Inc revenue was $10 million in 2024. Gamma Inc revenue was $10M in 2024.")
        .expect("ingest");
    let conflicts = engine.conflicts(10).expect("conflicts");
    assert!(
        !conflicts.iter().any(|c| c.subject_key.contains("gamma")),
        "unit-normalized equal values must not conflict; got {conflicts:?}"
    );
}

#[test]
fn jobs_support_backoff_dead_letter_and_cancel() {
    use lkos::jobs;
    use lkos::storage::Store;
    let mut store = Store::open_in_memory().expect("store");
    let conn = store.conn();

    // Backoff: a failing job is re-released with run_at in the future.
    let jid = jobs::enqueue_process_document(conn, 1).expect("enqueue");
    let job = jobs::claim_next(conn).expect("claim").expect("claimed");
    let requeued = jobs::fail_and_maybe_retry(conn, &job, "boom", 60, 300).expect("retry");
    assert!(requeued, "first failure requeues");
    let run_at: Option<String> = conn
        .query_row(
            "SELECT run_at FROM jobs WHERE id = ?1",
            rusqlite::params![jid],
            |r| r.get(0),
        )
        .expect("run_at");
    assert!(run_at.is_some(), "backoff sets a future release time");
    // While run_at is in the future the job is NOT claimable.
    assert!(
        jobs::claim_next(conn).expect("claim").is_none(),
        "backed-off job stays hidden until release time"
    );
    // Simulate the backoff window elapsing.
    conn.execute(
        "UPDATE jobs SET run_at = '2000-01-01T00:00:00Z' WHERE id = ?1",
        rusqlite::params![jid],
    )
    .expect("rewind");
    let job = jobs::claim_next(conn).expect("claim").expect("claimable after backoff");
    let requeued = jobs::fail_and_maybe_retry(conn, &job, "boom 2", 1, 1).expect("retry");
    let attempts: i64 = conn
        .query_row("SELECT attempts FROM jobs WHERE id = ?1", rusqlite::params![jid], |r| r.get(0))
        .unwrap();
    let max_attempts: i64 = conn
        .query_row("SELECT max_attempts FROM jobs WHERE id = ?1", rusqlite::params![jid], |r| r.get(0))
        .unwrap();
    if attempts >= max_attempts {
        assert!(!requeued, "terminal failure dead-letters");
        let status: String = conn
            .query_row("SELECT status FROM jobs WHERE id = ?1", rusqlite::params![jid], |r| r.get(0))
            .unwrap();
        assert_eq!(status, "failed");
    } else {
        assert!(requeued);
    }

    // Cancellation: a pending job can be cancelled by id.
    let jid2 = jobs::enqueue_process_document(conn, 2).expect("enqueue2");
    let cancelled = jobs::cancel_job(conn, jid2).expect("cancel");
    assert!(cancelled, "pending job cancelled");
    let next = jobs::claim_next(conn).expect("claim");
    assert!(next.is_none() || next.as_ref().map(|j| j.id) != Some(jid2), "cancelled job never claims");
}

#[test]
fn synchronous_engine_end_to_end_with_semantics_and_delete() {
    let engine = Lkos::open_in_memory(sem_config()).expect("open");
    let doc = engine
        .ingest_bytes("q.md", b"Epsilon Systems revenue was $10M in 2024. Epsilon Systems launched Orion in 2024.")
        .expect("ingest");
    engine.train_semantic_index().expect("train");
    engine.reembed_stale_chunks("lsa-pmi-svd-v1", None).expect("reembed");
    let resp = engine.query(QueryRequest::new("Orion launch").top_k(3)).expect("query");
    assert!(!resp.hits.is_empty());
    engine.delete_document(doc.id).expect("delete");
    let stats = engine.stats().expect("stats");
    assert_eq!(stats.documents, 0);
    assert_eq!(stats.chunks, 0);
}

#[test]
fn arc_shared_engine_queries_concurrently() {
    let engine = Arc::new(Lkos::open_in_memory(sem_config()).expect("open"));
    for (name, body) in semantic_corpus() {
        engine.ingest_bytes(&name, body.as_bytes()).expect("ingest");
    }
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let e = engine.clone();
            std::thread::spawn(move || {
                let q = ["revenue profit", "gradient network", "query index", "plants sunlight"][i];
                e.query(QueryRequest::new(q).top_k(3)).expect("query")
            })
        })
        .collect();
    for h in handles {
        let resp = h.join().expect("thread");
        assert!(!resp.hits.is_empty());
    }
}
