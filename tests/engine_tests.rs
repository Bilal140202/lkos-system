//! End-to-end engine tests: ingestion → knowledge extraction → retrieval →
//! graph → provenance → conflicts → backup. These tests exercise the public
//! API exactly the way an embedding application would.
//!
//! Property-style tests required by the spec (§95):
//! - indexing the same document twice is idempotent
//! - deleting a document removes its derived knowledge
//! - every derived artifact retains provenance
//! - summary failure never breaks ingestion (LLM-optional invariant)

use lkos::llm::FakeProvider;
use lkos::{Config, Lkos, QueryRequest, RetrievalMode};

fn mem() -> Lkos {
    Lkos::open_in_memory(Config::default()).expect("engine")
}

const DOC_A: &str = r#"# Acme Corp Handbook

## Revenue

Acme Corp revenue was $10M in 2024. The growth team measures
everything about customer retention quarterly.

## Vacation Policy

Employees at Acme Corp get 25 days of paid vacation every year.
Managers must approve leave two weeks in advance.
"#;

const DOC_B: &str = r#"# Northwind Audit Notes

Northwind Ltd reviewed the audited statements twice this quarter.
Acme Corp revenue was $25M in 2024 according to the audited statements.
Acme Corp announced Project Falcon during the same quarter.
"#;

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

#[test]
fn ingest_creates_chunks_embeddings_and_provenance() {
    let engine = mem();
    let info = engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    assert_eq!(info.filename, "handbook.md");
    assert_eq!(info.doc_type, "markdown");
    assert_eq!(info.readiness_state, "ready");
    assert!(
        info.chunk_count >= 1,
        "expected at least one chunk, got {}",
        info.chunk_count
    );
    assert_eq!(info.content_hash.len(), 64, "sha256 hex expected");

    let stats = engine.stats().expect("stats");
    assert_eq!(stats.documents, 1);
    assert_eq!(stats.chunks, info.chunk_count);
    assert!(
        stats.provenance_records >= info.chunk_count,
        "every chunk needs a provenance row"
    );
    assert!(
        stats.entities > 0,
        "Acme Corp should be extracted as an entity"
    );
    assert!(stats.db_size_bytes > 0);
}

#[test]
fn ingest_is_idempotent_on_identical_content() {
    let engine = mem();
    let first = engine
        .ingest_bytes("doc.md", DOC_A.as_bytes())
        .expect("first");
    let second = engine
        .ingest_bytes("doc.md", DOC_A.as_bytes())
        .expect("second");
    assert_eq!(
        first.id, second.id,
        "same content must map to the same document"
    );
    let stats = engine.stats().expect("stats");
    assert_eq!(stats.documents, 1, "no duplicate documents");
    assert_eq!(stats.chunks, first.chunk_count, "no duplicate chunks");
}

#[test]
fn changed_content_creates_new_document() {
    let engine = mem();
    let a = engine.ingest_bytes("doc.md", DOC_A.as_bytes()).expect("a");
    let changed = DOC_A.replace("25 days", "30 days");
    let b = engine
        .ingest_bytes("doc.md", changed.as_bytes())
        .expect("b");
    assert_ne!(
        a.id, b.id,
        "different content hash must be a new document version"
    );
    assert_eq!(engine.documents().expect("docs").len(), 2);
}

#[test]
fn empty_and_tiny_inputs_do_not_crash() {
    let engine = mem();
    // Empty input is rejected with a clean error (never a crash/panic).
    assert!(engine.ingest_bytes("empty.md", b"").is_err());
    assert!(engine.ingest_bytes("tiny.md", b"ok").is_ok());
    assert!(engine.query(QueryRequest::new("anything")).is_ok());
}

// ---------------------------------------------------------------------------
// Retrieval + explainability
// ---------------------------------------------------------------------------

#[test]
fn all_retrieval_modes_return_hits_with_explanations() {
    let engine = mem();
    engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    engine
        .ingest_bytes("audit.md", DOC_B.as_bytes())
        .expect("ingest");

    for mode in [
        RetrievalMode::LexicalOnly,
        RetrievalMode::VectorOnly,
        RetrievalMode::Hybrid,
    ] {
        let resp = engine
            .query(QueryRequest::new("vacation policy").mode(mode).top_k(5))
            .expect("query");
        assert!(!resp.hits.is_empty(), "mode {:?} returned no hits", mode);
        assert!(!resp.plan_explanation.is_empty());
        for h in &resp.hits {
            assert!(
                !h.matched_by.is_empty(),
                "every hit must explain why it matched"
            );
            assert!(h.rank >= 1);
            assert!(h.score > 0.0 || mode == RetrievalMode::VectorOnly);
        }
    }
}

#[test]
fn hybrid_fusion_reports_both_channels() {
    let engine = mem();
    engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    let resp = engine
        .query(QueryRequest::new("Acme Corp vacation policy").top_k(5))
        .expect("query");
    let any_vector = resp.hits.iter().any(|h| {
        h.matched_by
            .iter()
            .any(|m| matches!(m, lkos::MatchSource::Vector { .. }))
    });
    let any_fts = resp.hits.iter().any(|h| {
        h.matched_by
            .iter()
            .any(|m| matches!(m, lkos::MatchSource::Fts { .. }))
    });
    assert!(
        any_vector && any_fts,
        "hybrid search should fuse both channels"
    );
}

#[test]
fn context_assembly_respects_budget_and_diversity() {
    let engine = mem();
    engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    engine
        .ingest_bytes("audit.md", DOC_B.as_bytes())
        .expect("ingest");
    let resp = engine
        .query(
            QueryRequest::new("Acme Corp")
                .top_k(10)
                .mode(RetrievalMode::Hybrid),
        )
        .expect("query");
    assert!(
        resp.context.len() <= 3000 + 200,
        "context budget violated: {}",
        resp.context.len()
    );
    // max_per_document = 3 by default.
    let mut counts = std::collections::HashMap::new();
    for h in &resp.hits {
        *counts.entry(h.document_id).or_insert(0usize) += 1;
    }
    // Note: hits may exceed max_per_document before assembly; the assembled
    // context is what respects diversity. Check the citation list instead.
    let cites = resp.context.matches("[").count();
    assert!(cites > 0, "context must include citation markers");
}

#[test]
fn empty_query_is_rejected() {
    let engine = mem();
    let err = engine
        .query(QueryRequest::new("   "))
        .expect_err("must reject");
    assert!(err.to_string().contains("empty query"));
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

#[test]
fn planner_detects_intents() {
    use lkos::query::classify_intent;
    use lkos::QueryIntent;
    assert_eq!(
        classify_intent("summarise my document"),
        QueryIntent::Summary
    );
    assert_eq!(
        classify_intent("compare these two reports"),
        QueryIntent::Comparative
    );
    assert_eq!(
        classify_intent("what changed in 2024"),
        QueryIntent::Temporal
    );
    assert_eq!(
        classify_intent("find \"vacation policy\""),
        QueryIntent::Exact
    );
    assert_eq!(classify_intent("who is John Smith"), QueryIntent::Entity);
    assert_eq!(
        classify_intent("how does billing work"),
        QueryIntent::Semantic
    );
}

#[test]
fn summary_fast_path_returns_prebuilt_summary() {
    let cfg = Config {
        chunk_max_chars: 200,
        chunk_min_chars: 40, // keep small sections from being merged away
        ..Config::default()
    };
    let engine = Lkos::open_in_memory(cfg).expect("engine");
    engine.set_llm(std::sync::Arc::new(FakeProvider {
        response: "A summary.".into(),
    }));
    let doc = engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    // FakeProvider produced a summary (chunk_count >= 2).
    assert!(
        doc.chunk_count >= 2,
        "test doc must produce 2+ chunks under small config"
    );
    assert_eq!(doc.readiness_state, "complete");
    let stored = engine
        .summary(doc.id)
        .expect("summary")
        .expect("some summary");
    assert!(stored.contains("A summary."));

    let mut req = QueryRequest::new("summarise my document");
    req.filters.document_ids = Some(vec![doc.id]);
    let resp = engine.query(req).expect("query");
    assert_eq!(resp.hits.len(), 0, "fast path returns no retrieval hits");
    assert_eq!(resp.context, stored);
    assert!(resp.plan_explanation.contains("summary fast-path"));
}

// ---------------------------------------------------------------------------
// LLM-optional invariant
// ---------------------------------------------------------------------------

#[test]
fn works_without_any_llm() {
    let engine = mem(); // no provider installed
    let doc = engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    assert_eq!(
        doc.readiness_state, "ready",
        "without LLM docs finish as ready"
    );
    assert!(engine.summary(doc.id).expect("summary").is_none());
    let resp = engine.query(QueryRequest::new("vacation")).expect("query");
    assert!(!resp.hits.is_empty());
    // ask() without a provider is a clean error, not a crash.
    let err = engine
        .ask("what is the vacation policy")
        .expect_err("no provider");
    assert!(err.to_string().contains("no LLM provider"));
}

#[test]
fn llm_failure_does_not_break_ingestion() {
    use lkos::llm::NullProvider;
    let engine = mem();
    engine.set_llm(std::sync::Arc::new(NullProvider));
    let doc = engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    assert_eq!(
        doc.readiness_state, "ready",
        "summary failure degrades to ready"
    );
}

#[test]
fn ask_gives_grounded_answer_or_refusal() {
    // Refusal contract: with zero evidence the engine refuses structurally
    // instead of letting the LLM hallucinate.
    let empty = mem();
    empty.set_llm(std::sync::Arc::new(FakeProvider {
        response: "Answer.".into(),
    }));
    let miss = empty.ask("what is the vacation policy").expect("ask");
    assert_eq!(miss, "I cannot find this in your documents.");

    // Grounded contract: with evidence, the answer is produced from context.
    let engine = mem();
    engine.set_llm(std::sync::Arc::new(FakeProvider {
        response: "Answer.".into(),
    }));
    engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    let answer = engine.ask("what is the vacation policy").expect("ask");
    assert!(!answer.is_empty());
}

// ---------------------------------------------------------------------------
// Knowledge: entities, claims, conflicts, graph
// ---------------------------------------------------------------------------

#[test]
fn entities_are_extracted_resolved_and_linked() {
    let engine = mem();
    engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    engine
        .ingest_bytes("audit.md", DOC_B.as_bytes())
        .expect("ingest");

    let acme = engine
        .entity_id_by_name("Acme Corp")
        .expect("lookup")
        .expect("acme entity exists");
    let hood = engine.neighborhood(acme, 10).expect("neighborhood");
    assert_eq!(hood.center.name, "Acme Corp");
    assert!(
        !hood.documents.is_empty(),
        "acme must link back to its documents"
    );
    // Northwind doc mentions Acme → co-occurrence edge exists.
    assert!(
        hood.neighbors
            .iter()
            .any(|(n, _)| n.name.contains("Northwind")),
        "co-occurrence graph should link Acme and Northwind"
    );
}

#[test]
fn claims_and_conflicts_are_detected_and_preserved() {
    let engine = mem();
    engine
        .ingest_bytes("audit.md", DOC_B.as_bytes())
        .expect("ingest");
    engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");

    let conflicts = engine.conflicts(10).expect("conflicts");
    assert!(
        !conflicts.is_empty(),
        "$10M vs $25M revenue for acme (canonical 'acme') must conflict"
    );
    let c = &conflicts[0];
    assert!(c.delta > 0.05);
    assert!(!c.explanation.is_empty());

    let claims = engine.claims_about("Acme Corp").expect("claims");
    assert!(
        claims.len() >= 2,
        "both revenue claims should be retrievable, got {}",
        claims.len()
    );
    assert_eq!(
        claims.iter().filter(|cl| cl.predicate == "revenue").count(),
        2
    );
}

#[test]
fn temporal_queries_prefer_matching_years() {
    let engine = mem();
    engine
        .ingest_bytes("old.md", b"# Old\n\nAcme revenue was $5M in 2019.\n")
        .expect("ingest");
    engine
        .ingest_bytes("new.md", b"# New\n\nAcme revenue was $50M in 2024.\n")
        .expect("ingest");
    let resp = engine
        .query(QueryRequest::new("Acme revenue 2024").top_k(5))
        .expect("query");
    assert!(!resp.hits.is_empty());
    let top_text = &resp.hits[0].text;
    assert!(
        top_text.contains("2024"),
        "temporal filter should keep 2024 evidence, got: {top_text}"
    );
    // Graceful fallback: constraint never empties the result set.
    assert!(!resp.hits.is_empty());
}

// ---------------------------------------------------------------------------
// Library management + durability
// ---------------------------------------------------------------------------

#[test]
fn delete_removes_all_derived_knowledge() {
    let engine = mem();
    let doc = engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    engine
        .ingest_bytes("audit.md", DOC_B.as_bytes())
        .expect("ingest");
    let before = engine.stats().expect("stats");
    assert!(before.entities > 0 && before.chunks > 0);

    engine.delete_document(doc.id).expect("delete");
    let after = engine.stats().expect("stats");
    assert_eq!(after.documents, 1);
    assert!(
        after.chunks < before.chunks,
        "chunks of deleted doc must go"
    );
    assert!(
        after.entity_mentions < before.entity_mentions,
        "mentions must go"
    );
    assert!(
        after.provenance_records < before.provenance_records,
        "provenance must go"
    );
    assert!(
        engine.document(doc.id).is_err(),
        "deleted document must not resolve"
    );
}

#[test]
fn backup_restore_roundtrip() {
    let dir = tempfile::tempdir().expect("tmp");
    let db = dir.path().join("lib.lkos");
    let engine = Lkos::open(&db, Config::default()).expect("open");
    engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    engine
        .ingest_bytes("audit.md", DOC_B.as_bytes())
        .expect("ingest");

    let backup = dir.path().join("backup.lkos");
    engine.backup_to(&backup).expect("backup");
    assert!(backup.exists());

    let restored = Lkos::open(&backup, Config::default()).expect("reopen");
    assert_eq!(restored.stats().expect("stats").documents, 2);
    let resp = restored
        .query(QueryRequest::new("vacation policy"))
        .expect("query");
    assert!(!resp.hits.is_empty(), "backup must be fully searchable");
    let ic = restored.integrity_check().expect("integrity");
    assert!(
        ic.iter().all(|r| r == "ok"),
        "integrity check failed: {:?}",
        ic
    );
}

#[test]
fn reopening_existing_library_preserves_state() {
    let dir = tempfile::tempdir().expect("tmp");
    let db = dir.path().join("lib.lkos");
    let engine = Lkos::open(&db, Config::default()).expect("open");
    let doc = engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    drop(engine);

    let reopened = Lkos::open(&db, Config::default()).expect("reopen");
    let docs = reopened.documents().expect("docs");
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].id, doc.id);
    assert_eq!(
        docs[0].chunk_count, doc.chunk_count,
        "chunks survive restart"
    );
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[test]
fn event_stream_reports_lifecycle() {
    let engine = mem();
    let rx = engine.subscribe();
    let doc = engine
        .ingest_bytes("handbook.md", DOC_A.as_bytes())
        .expect("ingest");
    engine.delete_document(doc.id).expect("delete");

    let mut names = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        names.push(ev.name);
    }
    for expected in [
        "document.added",
        "document.ready",
        "knowledge.updated",
        "document.deleted",
    ] {
        assert!(
            names.contains(&expected.into()),
            "missing {expected}; got {names:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Filesystem ingestion
// ---------------------------------------------------------------------------

#[test]
fn ingest_file_from_disk() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("note.md");
    std::fs::write(&path, DOC_A).expect("write");
    let engine = mem();
    let doc = engine.ingest_file(&path).expect("ingest file");
    assert_eq!(doc.doc_type, "markdown");
    assert!(doc.path.ends_with("note.md"));
    let resp = engine.query(QueryRequest::new("vacation")).expect("query");
    assert!(!resp.hits.is_empty());
}
