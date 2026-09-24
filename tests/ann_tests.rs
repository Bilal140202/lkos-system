//! ANN (HNSW) integration tests: engine-level dense-channel behaviour.
//!
//! Covers GitHub issue #2 acceptance work:
//! - `hnsw` mode is *exact* when `ef >= corpus size` (matches brute force)
//! - cache invalidation on delete/insert (no stale chunk ids, no
//!   "vanished mid-query" errors)
//! - `auto` degrades past `max_dense_scan` instead of refusing (the v0.9
//!   cap refused; v0.10 switches to the ANN index)
//! - `brute` mode still enforces the cap (documented contract preserved)
//! - filtered queries bypass the ANN path (index is unfiltered)

use lkos::{Config, Filters, Lkos, QueryRequest, RetrievalMode};

fn engine(cfg: Config) -> Lkos {
    Lkos::open_in_memory(cfg).expect("engine")
}

fn hnsw_cfg() -> Config {
    Config {
        // Keep the corpus below the LSA training floor so the embedder is
        // the deterministic hashing provider — same vectors for both engines
        // (also set in every other cfg built here).
        semantic_min_chunks: 1_000,
        ann_mode: "hnsw".into(),
        ann_min_chunks: 1,
        ann_ef_search: 512, // >= corpus below -> exact
        ..Config::default()
    }
}

const DOCS: [(&str, &str); 5] = [
    (
        "alpha.md",
        r#"# Alpha Report

The quarterly quantum telescope calibration produced stable photon counts.
Engineers validated the quantum telescope alignment across all sensors.

## Budget

The telescope budget consumed $4M of the quantum program in 2024.
"#,
    ),
    (
        "beta.md",
        r#"# Beta Digest

Marine biology surveys recorded coral bleaching near the reef shelf.
The biology team published coral recovery statistics after the storm.

## Follow-up

Coral nurseries expanded biology monitoring to three additional reefs.
"#,
    ),
    (
        "gamma.md",
        r#"# Gamma Ledger

Vintage wine auctions cleared at record prices for burgundy lots.
The wine cellar inventory was digitized with vintage bottle provenance.

## Storage

Humidity controls protect vintage wine bottles in the cellar vault.
"#,
    ),
    (
        "delta.md",
        r#"# Delta Notebook

Robotics prototypes passed the warehouse navigation trial in spring.
The robotics team shaved pick latency using vision guided grippers.
"#,
    ),
    (
        "epsilon.md",
        r#"# Epsilon Brief

Alpine glacier sensors reported thinning snowpack across north faces.
Glacier meltwater feeding the alpine hydropower reservoirs increased.
"#,
    ),
];

fn ingest_all(e: &Lkos) -> Vec<i64> {
    DOCS.iter()
        .map(|(name, body)| e.ingest_bytes(name, body.as_bytes()).expect("ingest").id)
        .collect()
}

fn hits_of(e: &Lkos, q: &str, mode: RetrievalMode) -> Vec<(i64, usize)> {
    let mut req = QueryRequest::new(q);
    req.top_k = 5;
    req.mode = mode;
    e.query(req)
        .expect("query")
        .hits
        .into_iter()
        .map(|h| (h.chunk_id, h.rank))
        .collect()
}

// ---------------------------------------------------------------------------
// Exactness: with ef >= N the ANN path must equal the brute-force path.
// ---------------------------------------------------------------------------

#[test]
fn hnsw_mode_matches_brute_force_exactly_when_ef_covers_corpus() {
    let queries = [
        "quantum telescope calibration budget",
        "coral bleaching reef recovery",
        "vintage wine cellar provenance",
        "warehouse robotics navigation grippers",
        "alpine glacier meltwater reservoir",
    ];
    for mode in [RetrievalMode::Hybrid, RetrievalMode::VectorOnly] {
        let brute = engine(Config {
            ann_mode: "brute".into(),
            semantic_min_chunks: 1_000,
            ..Config::default()
        });
        ingest_all(&brute);
        let ann = engine(hnsw_cfg());
        ingest_all(&ann);

        for q in queries {
            assert_eq!(
                hits_of(&brute, q, mode),
                hits_of(&ann, q, mode),
                "hnsw path must be identical to brute force at ef >= N (query: {q})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Cache invalidation: delete/insert must be visible on the next query.
// ---------------------------------------------------------------------------

#[test]
fn ann_cache_invalidates_on_delete_and_insert() {
    let e = engine(hnsw_cfg());
    let ids = ingest_all(&e);
    let alpha_id = ids[0];

    let before = hits_of(
        &e,
        "quantum telescope calibration",
        RetrievalMode::VectorOnly,
    );
    assert!(
        !before.is_empty(),
        "expected dense hits for the alpha query before deletion"
    );

    // Deleting a document must invalidate the index: the next query may not
    // return its chunks (and must not hit "chunk vanished mid-query").
    e.delete_document(alpha_id).expect("delete");
    let mut req = QueryRequest::new("quantum telescope calibration");
    req.top_k = 10;
    req.mode = RetrievalMode::VectorOnly;
    for h in &e.query(req.clone()).expect("query").hits {
        assert_ne!(
            h.document_id, alpha_id,
            "stale ANN cache served a deleted document's chunk"
        );
    }

    // A newly ingested document must also appear (cache rebuild covers it).
    e.ingest_bytes(
        "zeta.md",
        b"# Zeta Memo\n\nThe quantum telescope calibration resumed after repairs.\n",
    )
    .expect("ingest");
    let mut req = QueryRequest::new("quantum telescope calibration");
    req.top_k = 10;
    req.mode = RetrievalMode::VectorOnly;
    let reopened: Vec<i64> = e
        .query(req)
        .expect("query")
        .hits
        .into_iter()
        .map(|h| h.document_id)
        .collect();
    // Find zeta's id: it is the newest document (max id).
    let docs = e.documents().expect("docs");
    let zeta_id = docs.iter().map(|d| d.id).max().expect("docs non-empty");
    assert!(
        reopened.contains(&zeta_id),
        "new document missing from dense hits after cache rebuild: {reopened:?}"
    );
}

// ---------------------------------------------------------------------------
// auto: degrade past the cap instead of refusing.
// ---------------------------------------------------------------------------

#[test]
fn auto_mode_degrades_to_ann_past_dense_cap() {
    // v0.9 behaviour: corpus > max_dense_scan -> hard error. v0.10: the
    // auto policy switches to the ANN index and the query succeeds.
    let cfg = Config {
        max_dense_scan: 2, // force the cap below the corpus size
        semantic_min_chunks: 1_000,
        ..Config::default()
    };
    let e = engine(cfg);
    ingest_all(&e);

    let auto_hits = hits_of(&e, "coral bleaching reef", RetrievalMode::VectorOnly);
    assert!(!auto_hits.is_empty(), "auto mode must degrade, not refuse");

    // Reference: an engine without the cap (brute, large cap) agrees.
    let reference = engine(Config {
        max_dense_scan: 1_000_000,
        ann_mode: "brute".into(),
        semantic_min_chunks: 1_000,
        ..Config::default()
    });
    ingest_all(&reference);
    assert_eq!(
        auto_hits,
        hits_of(
            &reference,
            "coral bleaching reef",
            RetrievalMode::VectorOnly
        ),
        "auto-degraded ANN results must match brute force at ef >= N"
    );
}

#[test]
fn brute_mode_still_enforces_dense_cap() {
    let e = engine(Config {
        max_dense_scan: 2,
        ann_mode: "brute".into(),
        semantic_min_chunks: 1_000,
        ..Config::default()
    });
    ingest_all(&e);

    let mut req = QueryRequest::new("coral bleaching reef");
    req.mode = RetrievalMode::VectorOnly;
    let err = e
        .query(req)
        .expect_err("brute mode must refuse past the cap");
    let msg = match err {
        lkos::LkosError::Other(m) => m,
        other => panic!("unexpected error variant: {other:?}"),
    };
    assert!(msg.contains("cap"), "cap error expected, got: {msg}");
}

// ---------------------------------------------------------------------------
// Filters: the unfiltered ANN index must not serve filtered queries.
// ---------------------------------------------------------------------------

#[test]
fn filtered_queries_bypass_ann_and_match_brute_force() {
    let brute = engine(Config {
        ann_mode: "brute".into(),
        semantic_min_chunks: 1_000,
        ..Config::default()
    });
    let ann = engine(hnsw_cfg());
    let ids_brute = ingest_all(&brute);
    let ids_ann = ingest_all(&ann);

    let f = Filters {
        document_ids: Some(vec![ids_brute[1]]),
        ..Filters::default()
    };
    let f_ann = Filters {
        document_ids: Some(vec![ids_ann[1]]),
        ..Filters::default()
    };

    let q = |e: &Lkos, filters: Filters| -> Vec<(i64, usize)> {
        let mut req = QueryRequest::new("coral bleaching reef recovery");
        req.top_k = 5;
        req.mode = RetrievalMode::Hybrid;
        req.filters = filters;
        e.query(req)
            .expect("query")
            .hits
            .into_iter()
            .map(|h| (h.chunk_id, h.rank))
            .collect()
    };

    let brute_hits = q(&brute, f);
    let ann_hits = q(&ann, f_ann);
    assert_eq!(brute_hits, ann_hits, "filtered queries must brute force");
    assert!(
        !brute_hits.is_empty(),
        "sanity: filtered corpus should match its own query"
    );
}
