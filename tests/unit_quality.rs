//! Unit tests for the deterministic core (chunking, embeddings, entity
//! resolution, temporal logic, knowledge extraction) + a fixed golden
//! retrieval benchmark on a miniature corpus.
//!
//! These tests pin *behavior*, not implementation details, so refactors of
//! internals stay free as long as observable guarantees hold.

use lkos::chunking::{chunk_document, CHUNKER_VERSION};
use lkos::embeddings::{cosine, tokenize, EmbeddingProvider, HashingEmbedder};
use lkos::entities::canonical_key;
use lkos::ingestion;
use lkos::knowledge::{
    extract_claims, extract_entities, extract_keywords, ko_from_json, ko_to_json,
    split_sentences, T_ORG, T_PERSON,
};
use lkos::query::{assemble_context, plan};
use lkos::retrieval::fts_escape;
use lkos::temporal::{detect_temporal, matches_constraint, TemporalConstraint};
use lkos::types::{DocType, SearchHit};
use lkos::{Config, Lkos, QueryRequest};

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

#[test]
fn markdown_chunking_respects_structure() {
    let cfg = Config::default();
    let md = format!(
        "# Title\n\n{}paragraph one.\n\n## Section A\n\n{}paragraph two.\n",
        "word ".repeat(120),
        "value ".repeat(120)
    );
    let chunks = chunk_document(&md, DocType::Markdown, &cfg);
    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(|c| c.text.chars().count() <= cfg.chunk_max_chars + 50));
    // Heading chunks carry their section title forward.
    let with_sections = chunks.iter().filter(|c| c.section_title.is_some()).count();
    assert!(with_sections > 0, "markdown headings must propagate section titles");
    // Offsets must be ordered and non-decreasing.
    let offs: Vec<(usize, usize)> = chunks.iter().map(|c| (c.start_offset, c.end_offset)).collect();
    for w in offs.windows(2) {
        assert!(w[0].0 <= w[1].0, "chunk offsets must be ordered");
    }
    assert!(!CHUNKER_VERSION.is_empty());
}

#[test]
fn code_chunking_keeps_units_together() {
    let cfg = Config::default();
    let code = r#"
fn alpha() -> u32 { 1 }

/// docs for beta
fn beta(x: u32) -> u32 {
    let mut acc = 0u32;
    for i in 0..x {
        acc = acc.wrapping_add(i);
    }
    acc
}

struct Gamma { field: u32 }
"#;
    let chunks = chunk_document(code, DocType::Code, &cfg);
    assert!(chunks.iter().any(|c| c.text.contains("fn beta")), "function bodies must not be split mid-unit");
    assert!(chunks.iter().any(|c| c.text.contains("struct Gamma")));
}

#[test]
fn chunk_hash_is_stable() {
    let a = lkos::chunking::chunk_hash("hello world");
    let b = lkos::chunking::chunk_hash("hello world");
    let c = lkos::chunking::chunk_hash("hello worlD");
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

#[test]
fn hashing_embedder_is_deterministic_and_semantically_useful() {
    let emb = HashingEmbedder::new(256);
    let q = emb.embed_batch(&["vacation policy days off"]).expect("embed");
    let docs = emb
        .embed_batch(&[
            "employees receive paid vacation leave",
            "database migration tooling",
        ])
        .expect("embed");
    let sim_on_topic = cosine(&q[0], &docs[0]);
    let sim_off_topic = cosine(&q[0], &docs[1]);
    assert!(sim_on_topic > sim_off_topic, "lexical hashing must separate topics");
    // Determinism.
    let again = emb.embed_batch(&["vacation policy days off"]).expect("embed");
    assert_eq!(q[0], again[0]);
    // Unit norm (cosine with itself ≈ 1).
    assert!((cosine(&q[0], &q[0]) - 1.0).abs() < 1e-4);
}

#[test]
fn tokenizer_lowercases_and_drops_punctuation() {
    let toks = tokenize("Hello, World! It's RRF-based.");
    assert!(toks.iter().all(|t| t.chars().all(|c| c.is_alphanumeric())), "punctuation splits tokens: {toks:?}");
    assert!(toks.contains(&"hello".to_string()));
    assert!(toks.contains(&"rrf".to_string()), "hyphens split: {toks:?}");
    assert!(toks.contains(&"based".to_string()));
}

// ---------------------------------------------------------------------------
// Entity resolution
// ---------------------------------------------------------------------------

#[test]
fn canonical_key_strips_legal_suffixes_and_case() {
    assert_eq!(canonical_key("OpenAI Inc."), canonical_key("openai"));
    assert_eq!(canonical_key("Acme Corp."), canonical_key("ACME"));
    assert_eq!(canonical_key("Northwind Ltd"), "northwind");
    assert_eq!(canonical_key("Apple"), canonical_key("Apple Inc"), "legal suffix must not change identity");
    assert_eq!(canonical_key("  Open  AI  "), "open ai");
}

#[test]
fn entity_extraction_finds_typed_candidates() {
    let text = "Jane Smith visited Acme Corp on 2024-03-01 and paid $10M, 25% of the budget.";
    let cands = extract_entities(text);
    assert!(cands.iter().any(|c| c.surface == "Jane Smith" && c.entity_type == T_PERSON));
    assert!(cands.iter().any(|c| c.entity_type == T_ORG && c.surface.contains("Acme")));
    assert!(cands.iter().any(|c| c.entity_type == "money"));
    assert!(cands.iter().any(|c| c.entity_type == "percent"));
    assert!(cands.iter().any(|c| c.entity_type == "date"));
}

// ---------------------------------------------------------------------------
// Temporal logic
// ---------------------------------------------------------------------------

#[test]
fn temporal_detection_and_matching() {
    assert_eq!(detect_temporal("what changed in 2024"), TemporalConstraint::Year(2024));
    assert_eq!(
        detect_temporal("revenue between 2022 and 2024"),
        TemporalConstraint::Range(2022, 2024)
    );
    assert_eq!(detect_temporal("latest revenue"), TemporalConstraint::Latest);
    assert_eq!(detect_temporal("how does billing work"), TemporalConstraint::None);

    assert!(matches_constraint("profit in 2024 was high", &TemporalConstraint::Year(2024)));
    assert!(!matches_constraint("profit in 2019 was low", &TemporalConstraint::Year(2024)));
    assert!(matches_constraint("years 2022 and 2023", &TemporalConstraint::Range(2022, 2024)));
}

// ---------------------------------------------------------------------------
// Knowledge object serialization + claims
// ---------------------------------------------------------------------------

#[test]
fn knowledge_object_json_roundtrip() {
    let text = "Jane Smith joined Acme Corp. Acme revenue was $10M in 2024.";
    let ko = lkos::knowledge::extract_knowledge(text, &|_| None, 1);
    let json = ko_to_json(&ko);
    let back = ko_from_json(&json).expect("deserialize");
    assert_eq!(back.entities.len(), ko.entities.len());
    assert_eq!(back.keywords.len(), ko.keywords.len());
    assert_eq!(back.claims.len(), ko.claims.len());
}

#[test]
fn claim_extraction_finds_svo_patterns() {
    let claims = extract_claims("Acme Corp revenue was $10M in 2024. Bob Lee joined Initech LLC.");
    assert!(
        claims.iter().any(|c| c.predicate == "revenue" && c.object.contains("10")),
        "numeric metric claim expected"
    );
    assert!(claims.iter().any(|c| c.predicate == "joined"), "release/action claim expected");
}

#[test]
fn sentence_splitter_handles_abbreviations_reasonably() {
    let sentences = split_sentences("First sentence here. Second one follows! Third? Fragment");
    assert_eq!(sentences.len(), 4);
}

#[test]
fn keywords_exclude_stopwords() {
    let kws = extract_keywords("the database and the database index and the query planner", 5);
    assert!(!kws.is_empty());
    assert!(kws.iter().all(|(w, _)| !["the", "and"].contains(&w.as_str())));
}

// ---------------------------------------------------------------------------
// Query utilities
// ---------------------------------------------------------------------------

#[test]
fn fts_escape_quotes_user_tokens() {
    let out = fts_escape("vacation \"policy\" AND (NOR)");
    // Every user token must appear quoted (injection-safe); the user's bare
    // AND must be demoted to a quoted token, not an operator.
    assert!(out.contains("\"and\"*"), "user AND must be quoted: {out}");
    assert!(out.contains("\"vacation\"*"));
    assert!(out.contains("\"nor\"*"));
    assert!(!out.contains('('), "parentheses must not survive: {out}");
    assert_eq!(fts_escape("   "), "", "empty query escapes to empty string");
}

#[test]
fn assemble_context_orders_citations_and_respects_per_doc_cap() {
    let mk = |id: i64, doc: &str| SearchHit {
        chunk_id: id,
        document_id: if doc == "a.md" { 1 } else { 2 }, // a.md shares one doc id
        document: doc.into(),
        section: Some("sec".into()),
        text: format!("chunk text {id}"),
        score: 1.0,
        rank: 0,
        matched_by: vec![],
        authority_score: 1.0,
    };
    let hits = vec![mk(1, "a.md"), mk(2, "a.md"), mk(3, "b.md")];
    let ctx = assemble_context(&hits, 10_000, 1);
    assert!(ctx.contains("[1] a.md"));
    assert!(ctx.contains("[3] b.md"));
    assert!(!ctx.contains("[2] a.md"), "per-document cap must drop the second a.md chunk");
    assert!(ctx.contains("CITATIONS:"));
}

#[test]
fn planner_weights_shift_by_intent() {
    let cfg = Config::default();
    let exact = plan(&QueryRequest::new("find \"MAX_CONTEXT\""), &cfg);
    assert_eq!(exact.w_fts, 0.75, "exact queries lean lexical");
    let semantic = plan(&QueryRequest::new("how does billing work"), &cfg);
    assert_eq!(semantic.w_vector, cfg.weight_vector);
    assert!(!semantic.explanation.is_empty());
}

// ---------------------------------------------------------------------------
// Ingestion normalization
// ---------------------------------------------------------------------------

#[test]
fn normalization_preserves_structure_and_cleanwhitespace() {
    let raw = "# Heading\r\n\nSome *text* here.\n\n\n\n\n\nEnd.";
    let text = ingestion::normalize(raw);
    assert!(text.contains("# Heading"), "markdown structure must survive normalization: {text}");
    assert!(text.contains("Some *text* here."));
    assert!(!text.contains('\r'), "CRLF must be normalized to LF");
    assert!(!text.contains("\n\n\n\n"), "blank runs must collapse");
}

#[test]
fn content_hash_matches_sha256_length() {
    let h = ingestion::content_hash(b"hello");
    assert_eq!(h.len(), 64);
    assert_eq!(h, ingestion::content_hash(b"hello"));
    assert_ne!(h, ingestion::content_hash(b"hellO"));
}

#[test]
fn extract_detects_doc_types_by_extension() {
    assert_eq!(ingestion::extract("a.md", b"# hi").expect("md").doc_type, DocType::Markdown);
    assert_eq!(ingestion::extract("b.rs", b"fn f(){}").expect("rs").doc_type, DocType::Code);
    assert_eq!(ingestion::extract("c.json", b"{}").expect("json").doc_type, DocType::Data);
    assert_eq!(ingestion::extract("d.txt", b"hello").expect("txt").doc_type, DocType::Text);
}

// ---------------------------------------------------------------------------
// Golden retrieval benchmark (miniature, deterministic)
// ---------------------------------------------------------------------------

/// Fixed corpus + fixed probes with known ground truth. If a ranking change
/// breaks one of these expectations, retrieval regressed (or improved — then
/// update the goldens deliberately and record why in the PR).
#[test]
fn golden_retrieval_miniature_corpus() {
    let engine = Lkos::open_in_memory(Config::default()).expect("engine");
    let corpus: &[(&str, &str)] = &[
        ("vacation.md", "# Vacation Policy\n\nEmployees receive 25 paid vacation days per year. Leave requires manager approval two weeks ahead.\n"),
        ("billing.md", "# Billing Guide\n\nInvoices are issued monthly. Payment methods include card and bank transfer. Overdue invoices pause service.\n"),
        ("auth.md", "# Authentication\n\nThe system uses OAuth2 with rotating refresh tokens. Sessions expire after 24 hours of inactivity.\n"),
        ("backup.md", "# Backup Runbook\n\nNightly snapshots run at 02:00 UTC. Retention keeps 30 daily and 12 monthly copies. Restores are tested quarterly.\n"),
    ];
    for (name, body) in corpus {
        engine.ingest_bytes(name, body.as_bytes()).expect("ingest");
    }

    let probes: &[(&str, &str)] = &[
        ("how many vacation days do employees get", "vacation.md"),
        ("invoice payment methods", "billing.md"),
        ("session token expiration", "auth.md"),
        ("snapshot retention restore", "backup.md"),
        ("manager approval for leave", "vacation.md"),
        ("oauth refresh tokens", "auth.md"),
    ];
    let mut hits_at_1 = 0usize;
    for (query, expected) in probes {
        let resp = engine.query(QueryRequest::new(*query).top_k(3)).expect("query");
        let top = resp.hits.first().map(|h| h.document.as_str()).unwrap_or("NONE");
        if top == *expected {
            hits_at_1 += 1;
        } else {
            panic!(
                "golden miss for {query:?}: expected {expected:?} at rank 1, got {top:?} (plan: {})",
                resp.plan_explanation
            );
        }
    }
    assert_eq!(hits_at_1, probes.len(), "all golden probes must hit at rank 1");
}
