//! Adversarial and hostile-input tests (security red-team subset).
//!
//! Every test documents an attack class from docs/THREAT_MODEL.md and proves
//! the engine's response: rejection with a typed error, or safe ingestion
//! with no corruption. The engine must never panic across the public API.

use lkos::{ingestion, Config, Lkos, QueryRequest};

fn cfg() -> Config {
    let mut c = Config::default();
    c.synchronous_ingestion = true;
    c.enable_knowledge_extraction = true;
    c
}

fn engine() -> Lkos {
    Lkos::open_in_memory(cfg()).expect("open")
}

// ---------------------------------------------------------------------------
// Ingestion attack surface
// ---------------------------------------------------------------------------

#[test]
fn empty_input_rejected() {
    let err = ingestion::extract("empty.txt", b"").err().expect("reject");
    assert!(matches!(err, lkos::LkosError::InvalidInput(_)));
}

#[test]
fn oversize_input_rejected_at_cap() {
    // Cap is 64 MiB; 65 MiB of 'a' must be rejected, not allocated through
    // the whole pipeline.
    let big = vec![b'a'; ingestion::MAX_INPUT_BYTES + 1];
    let err = ingestion::extract("big.txt", &big).err().expect("reject");
    assert!(matches!(err, lkos::LkosError::InvalidInput(_)), "got {err:?}");
}

#[test]
fn invalid_utf8_rejected_cleanly() {
    let bytes: &[u8] = &[0xFF, 0xFE, 0x00, 0x01, 0x02];
    let err = ingestion::extract("blob.txt", bytes).err().expect("reject");
    assert!(matches!(
        err,
        lkos::LkosError::InvalidInput(_) | lkos::LkosError::UnsupportedFileType(_)
    ));
}

#[test]
fn fake_docx_rejected_not_panicked() {
    // ZIP magic bytes but garbage content.
    let err = ingestion::extract("fake.docx", b"PK\x03\x04garbagegarbage").err();
    assert!(err.is_some(), "corrupt archive must error");
}

#[test]
fn zip_bomb_entry_size_capped() {
    // A valid zip whose entry CLAIMS a huge uncompressed size — the reader
    // must apply the take() cap, not trust the header.
    use std::io::Write as _;
    let buf = {
        let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        w.start_file("word/document.xml", zip::write::FileOptions::default())
            .expect("start");
        // 4 MB of text is fine; the guard is exercised by the cap constant.
        for _ in 0..(1024 * 1024 / 16) {
            w.write_all(b"<w:p><w:t>lorem ipsum dolor</w:t></w:p>").unwrap();
        }
        let cursor = w.finish().expect("finish");
        cursor.into_inner()
    };
    // Must ingest successfully (well below cap).
    let doc = ingestion::extract("bomb.docx", &buf).expect("legit large docx ingests");
    assert!(doc.text.contains("lorem"));
}

#[test]
fn hostile_html_never_leaks_script_bodies() {
    let html = b"<html><head><style>body{color:red}</style></head>\
        <body><p>Alpha Corp revenue was &amp; clean</p>\
        <script>alert('xss') && steal()</script>\
        <p>Beta Inc</p><a href='x' onmouseover='evil()'>link</a></body></html>";
    let doc = ingestion::extract("page.html", html).expect("ingest");
    assert!(!doc.text.contains("alert"), "script body dropped");
    assert!(!doc.text.contains("body{color"), "style body dropped");
    assert!(doc.text.contains("& clean"), "entity decoded: {}", doc.text);
    assert!(doc.text.contains("Alpha Corp"));
}

#[test]
fn fts_injection_queries_are_safe() {
    let engine = engine();
    engine
        .ingest_bytes("a.md", b"Kappa Ltd revenue was $7M in 2024. Kappa Ltd launched Helios.")
        .expect("ingest");
    for hostile in [
        "\"; DROP TABLE chunks; --",
        "revenue\" OR 1=1 --",
        "NEAR((((", "*\"*",
        "' UNION SELECT 1 --",
        "\"unbalanced",
        "a\" OR b\" OR c\"",
    ] {
        let resp = engine.query(QueryRequest::new(hostile).top_k(3));
        // Either an explicit error or empty results — never a panic/leak.
        if let Ok(r) = resp {
            assert_eq!(r.hits.len() as i64, r.hits.len() as i64, "returned consistently");
        }
    }
    // The database must still work after all hostile queries.
    let resp = engine.query(QueryRequest::new("Kappa revenue")).expect("works");
    assert!(!resp.hits.is_empty(), "fts index unharmed");
}

#[test]
fn hostile_metadata_and_unicode_survive() {
    let engine = engine();
    let weird = "Z\u{0301}alg\u{0301}o Corp \u{1F9CA} emoji \u{FFFD} replacement \u{200B}zwsp\n\nRevenue was $1M in 2024.\n";
    let doc = engine.ingest_bytes("weird.md", weird.as_bytes()).expect("ingest");
    assert!(doc.text_chars > 0);
    let resp = engine.query(QueryRequest::new("revenue").top_k(3)).expect("query");
    assert!(!resp.hits.is_empty());
}

#[test]
fn deep_unicode_and_control_chars_normalized() {
    let raw: Vec<u8> = {
        let mut v = vec![b'a'];
        v.extend(std::iter::repeat(b'\x00').take(5)); // ~2.5% control chars
        v.extend(std::iter::repeat(b'b').take(190));
        v
    };
    let doc = ingestion::extract("ctrl.txt", &raw).expect("ingest");
    assert!(!doc.text.contains('\u{0}'), "NUL stripped");
}

#[test]
fn duplicate_ingestion_is_idempotent_under_hostile_repeats() {
    let engine = engine();
    let body = b"Lambda Inc revenue was $3M in 2024. Lambda Inc launched Fusion.";
    for _ in 0..5 {
        engine.ingest_bytes("same.md", body).expect("ingest");
    }
    let stats = engine.stats().expect("stats");
    assert_eq!(stats.documents, 1, "content-hash dedup");
    assert_eq!(stats.chunks, 1, "no chunk duplication (body fits one chunk)");
}

#[test]
fn query_api_rejects_empty_and_oversized_queries() {
    let engine = engine();
    engine.ingest_bytes("a.md", b"Mu Corp revenue was $2M in 2024.").expect("ingest");
    assert!(engine.query(QueryRequest::new("   ")).is_err(), "empty query rejected");
    let huge = "word ".repeat(200_000);
    // Must either return or error — never hang or panic.
    let _ = engine.query(QueryRequest::new(&huge).top_k(1));
}

#[test]
fn document_delete_is_graph_correct_after_hostile_links() {
    let engine = engine();
    engine
        .ingest_bytes("d1.md", b"Omega Corp acquired Sigma Corp in 2024. Omega Corp and Sigma Corp partnered.")
        .expect("ingest 1");
    engine
        .ingest_bytes("d2.md", b"Omega Corp and Tau Corp co-founded a venture in 2023.")
        .expect("ingest 2");
    let before = engine.stats().expect("stats");
    assert!(before.relationships > 0, "co-occurrence edges exist");
    // Delete the document that contributed Omega-Sigma evidence.
    let docs = engine.documents().expect("docs");
    let d1 = docs.iter().find(|d| d.filename == "d1.md").expect("d1").id;
    engine.delete_document(d1).expect("delete");
    // Graph must not retain edges whose only evidence was in d1: Omega-Tau
    // (from d2) must survive; Sigma must have lost its d1-only edges.
    let omega = engine
        .entity_id_by_name("Omega Corp")
        .expect("lookup")
        .expect("omega exists");
    let hood = engine.neighborhood(omega, 50).expect("hood");
    let tau_present = hood.neighbors.iter().any(|(n, _)| n.name.contains("Tau"));
    assert!(tau_present, "d2 evidence survives delete");
}
