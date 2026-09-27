//! Paraphrase-transfer probe (issue #8, findings B/C): queries that share
//! ZERO surface tokens with their target chunk.
//!
//! Why this is a real paraphrase test: `tokenize` (lowercase alphanumeric
//! runs) is the same tokenizer that feeds LSA training and the hashing
//! fallback. When `tokenize(query) ∩ tokenize(target) = ∅`, the lexical
//! channel cannot rank the target for the query, and the hashing fallback
//! only sees char-trigram noise. Ranking the target through the DENSE
//! channel is therefore possible only through the trained LSA space — the
//! corpus' sibling chunks co-occur query vocabulary with target vocabulary,
//! so the corpus-trained embedding space must bridge the gap. That is
//! distributional paraphrase transfer, asserted mechanically.
//!
//! Corpus design (4 themes × 5 docs, one chunk per doc):
//! - `D*` target docs use theme vocabulary T only;
//! - `B*` bridge docs co-occur query vocabulary Q with T (the LSA bridge);
//! - `F*` filler docs enrich the theme direction.
//!
//! Each pair (query, target) is checked at runtime for zero token overlap,
//! so corpus drift cannot silently weaken the probe. Content words are
//! theme-exclusive and no token appears in more than a handful of chunks
//! (ubiquitous function words create global TF-IDF correlation that leaks
//! across themes through subspace truncation).
//!
//! The companion canary test runs the SAME pairs on the hashing fallback
//! (no trained model) and asserts the probe discriminates: if a future
//! change degrades the semantic model to lexical-grade behavior, the floors
//! in the semantic test fail (semantic-regression is detectable).

#![allow(clippy::field_reassign_with_default)]

use lkos::{embeddings::tokenize, Config, Lkos, QueryRequest, RetrievalMode};

/// (query, target document, theme prefix) — the target document must NOT
/// contain any token of the query (asserted at runtime below).
const PAIRS: &[(&str, &str, &str)] = &[
    ("guitar chords", "music-strings.md", "music"),
    ("singer vocals", "music-opera.md", "music"),
    ("fertilizer vegetables", "garden-soil.md", "garden"),
    ("blooming flowers", "garden-roses.md", "garden"),
    ("graphics card", "compute-tensor.md", "compute"),
    ("supercomputer datacenter", "compute-cluster.md", "compute"),
    ("underwater ecosystems", "ocean-coral.md", "ocean"),
    ("moon gravity", "ocean-tides.md", "ocean"),
];

fn probe_corpus() -> Vec<(String, String)> {
    // Exclusive-bridge design: for every pair, the bridge chunk B co-occurs
    // the query vocabulary Q ONLY with target tokens that appear in no other
    // document of the corpus. The target is therefore the unique latent
    // neighbor of the query behind its literal bridge — intra-theme fillers
    // cannot shadow it without genuinely learned co-occurrence structure.
    let docs: &[(&str, &str)] = &[
        // ---- music ----
        // D1 (target of "guitar chords"): violin/concerto/bow exclusive here
        (
            "music-strings.md",
            "Violin sings above orchestra. Concerto demands bow that whispers. Violin soars; concerto gleams under bow.",
        ),
        // B1: co-occurs guitar/chords ONLY with violin/concerto
        (
            "music-folk.md",
            "Guitar chords weave beneath violin. Concerto swells when guitar chords soften. Violin, guitar trade chords; concerto closes.",
        ),
        // D2 (target of "singer vocals"): aria/pit/voice exclusive here
        (
            "music-opera.md",
            "Opera stages voice at its most extreme. Aria suspends time above pit. Aria returns; pit falls silent.",
        ),
        // B2: co-occurs singer/vocals ONLY with aria/pit/voice
        (
            "music-vocal.md",
            "Singer trains vocals; aria awaits. Singer gives voice to pit. Vocals swell; aria carries singer.",
        ),
        // F1: theme enrichment, shares no bridge-relevant token
        (
            "music-hall.md",
            "Orchestra tunes in hall. Conductor lifts baton.",
        ),
        // ---- garden ----
        // D3 (target of "fertilizer vegetables"): compost/feeds exclusive here
        (
            "garden-soil.md",
            "Compost feeds soil. Earthworms aerate bed; tomatoes ripen. Rich compost keeps bed alive.",
        ),
        // B3: co-occurs fertilizer/vegetables ONLY with compost/feeds
        (
            "garden-kitchen.md",
            "Fertilizer choices change vegetables. Compost feeds vegetables; synthetic fertilizer cannot feed like compost.",
        ),
        // D4 (target of "blooming flowers"): roses/mulch exclusive here
        (
            "garden-roses.md",
            "Roses want sun, mulch. Prune canes before buds break. Mulched roses bloom harder.",
        ),
        // B4: co-occurs blooming/flowers ONLY with roses/mulch
        (
            "garden-nursery.md",
            "Blooming flowers crown roses. Nursery sells blooming flowers beside mulch. Flowers, roses fill nursery.",
        ),
        // F2: theme enrichment
        (
            "garden-shed.md",
            "Shed holds twine, pots. Seeds wait; soil warms.",
        ),
        // ---- compute ----
        // D5 (target of "graphics card"): crunches/tensor/matrices exclusive here
        (
            "compute-tensor.md",
            "Accelerator crunches matrices. Tensor cores sprint through training loops. Tensor unit crunches numbers nightly.",
        ),
        // B5: co-occurs graphics/card ONLY with crunches/tensor/matrices
        (
            "compute-render.md",
            "Graphics card renders frames. Same graphics card crunches tensor matrices at speed. Graphics card crunches; tensor matrices stream.",
        ),
        // D6 (target of "supercomputer datacenter"): racks/nodes exclusive here
        (
            "compute-cluster.md",
            "Cluster schedules jobs across racks. Nodes hum until batch completes. Cluster pauses; maintenance window opens.",
        ),
        // B6: co-occurs datacenter/supercomputer ONLY with racks/nodes/cluster/jobs/batch
        (
            "compute-site.md",
            "Datacenter houses supercomputer. Racks hold nodes for supercomputer. Cluster jobs run batch inside datacenter racks; nodes hum, supercomputer completes.",
        ),
        // F3: theme enrichment
        (
            "compute-lab.md",
            "Lab measures cluster performance. Results stream through interconnect.",
        ),
        // ---- ocean ----
        // D7 (target of "underwater ecosystems"): coral/reefs/shelter exclusive here
        (
            "ocean-coral.md",
            "Coral reefs shelter lagoon. Parrotfish graze; reef glows at dusk. Coral gardens hide cleaner shrimp.",
        ),
        // B7: co-occurs underwater/ecosystems ONLY with coral/reefs/shelter
        (
            "ocean-science.md",
            "Underwater ecosystems depend on coral. Reefs shelter ecosystems underwater. Scientists map coral reefs yearly; ecosystems need reefs.",
        ),
        // D8 (target of "moon gravity"): tides exclusive here
        (
            "ocean-tides.md",
            "Tides sweep estuary twice daily. Currents carry plankton seaward. Tides turn; lagoon breathes.",
        ),
        // B8: co-occurs moon/gravity ONLY with tides
        (
            "ocean-moon.md",
            "Moon gravity drags tides. Moon sets tides in motion twice monthly. Gravity writes rhythm; tides obey.",
        ),
        // F4: theme enrichment
        (
            "ocean-deep.md",
            "Deep ocean stays dark. Lanternfish climb toward surface, flashing.",
        ),
    ];
    docs.iter()
        .map(|(n, b)| (n.to_string(), b.to_string()))
        .collect()
}

fn probe_config() -> Config {
    let mut c = Config::default();
    c.synchronous_ingestion = true;
    c.semantic_min_chunks = 8; // 20-chunk corpus trains comfortably
    c.lsa_dim = 8;
    c.lsa_min_df = 1; // query terms may occur in 1-2 bridge chunks only
    c.lsa_max_vocab = 512;
    c.embedding_provider = "lsa".into();
    c
}

fn ingest_all(engine: &Lkos) {
    for (name, body) in probe_corpus() {
        engine.ingest_bytes(&name, body.as_bytes()).expect("ingest");
    }
}

/// Engine with the trained LSA model installed and every chunk migrated.
fn semantic_engine() -> Lkos {
    let engine = Lkos::open_in_memory(probe_config()).expect("open");
    ingest_all(&engine);
    let trained = engine.train_semantic_index().expect("train");
    assert!(trained, "20-chunk probe corpus must train LSA");
    let migrated = engine
        .reembed_stale_chunks("lsa-pmi-svd-v1", None)
        .expect("reembed");
    assert!(migrated > 0, "chunks migrated to LSA");
    assert_eq!(
        engine.stale_embedding_count().expect("stale count"),
        0,
        "dense channel fully on the trained model"
    );
    engine
}

/// Engine without a trained model: dense channel runs the hashing fallback.
fn fallback_engine() -> Lkos {
    let mut cfg = probe_config();
    cfg.semantic_min_chunks = usize::MAX; // never train
    let engine = Lkos::open_in_memory(cfg).expect("open");
    ingest_all(&engine);
    engine
}

/// Dense-channel rank (1-based) of the target document for a query.
/// `top_k` covers the whole corpus so a miss is an explicit `None`.
fn dense_target_rank(engine: &Lkos, query: &str, target_doc: &str) -> Option<usize> {
    let resp = engine
        .query(
            QueryRequest::new(query)
                .top_k(24)
                .mode(RetrievalMode::VectorOnly),
        )
        .expect("dense query");
    assert!(!resp.hits.is_empty(), "dense channel returned no hits");
    resp.hits
        .iter()
        .find(|h| h.document == target_doc)
        .map(|h| h.rank)
}

/// Mechanical zero-overlap check: the probe is only meaningful while every
/// pair shares no tokenizer token between query and target text.
#[test]
fn probe_pairs_have_zero_token_overlap_by_construction() {
    let corpus = probe_corpus();
    for (query, target_doc, _) in PAIRS {
        let (_, text) = corpus
            .iter()
            .find(|(n, _)| n == target_doc)
            .unwrap_or_else(|| panic!("corpus doc {target_doc} missing"));
        let q_tokens: std::collections::HashSet<_> = tokenize(query).into_iter().collect();
        let d_tokens: std::collections::HashSet<_> = tokenize(text).into_iter().collect();
        let shared: Vec<_> = q_tokens.intersection(&d_tokens).collect();
        assert!(
            shared.is_empty(),
            "pair ('{query}' -> {target_doc}) shares tokens {shared:?}; \
             the probe is invalid until the corpus is fixed"
        );
    }
}

#[test]
fn paraphrase_probe_dense_channel_recovers_zero_overlap_targets() {
    let engine = semantic_engine();
    let mut reciprocal_ranks = Vec::new();
    let mut rank_violations: Vec<String> = Vec::new();
    let mut rows = String::new();
    for (query, target_doc, theme) in PAIRS {
        let rank = dense_target_rank(&engine, query, target_doc).unwrap_or_else(|| {
            panic!("target {target_doc} absent from dense ranking for '{query}'")
        });
        // Every document ranked ABOVE the target must be from the same theme:
        // the probe transfers into the right neighborhood, not into noise.
        let resp = engine
            .query(
                QueryRequest::new(*query)
                    .top_k(24)
                    .mode(RetrievalMode::VectorOnly),
            )
            .expect("dense query for ordering check");
        let target_pos = resp
            .hits
            .iter()
            .position(|h| h.document == *target_doc)
            .expect("target present");
        let above: Vec<&str> = resp.hits[..target_pos]
            .iter()
            .map(|h| h.document.as_str())
            .collect();
        for doc in &above {
            let doc_theme = doc.split('-').next().unwrap_or("");
            assert_eq!(
                doc_theme, *theme,
                "cross-theme doc {doc} ranked above target {target_doc} for '{query}'; above={above:?}"
            );
        }
        let rank_ok = rank <= 3;
        if !rank_ok {
            rank_violations.push(format!("{target_doc} ranked {rank} for '{query}'"));
        }
        let top3: Vec<&str> = resp
            .hits
            .iter()
            .take(3)
            .map(|h| h.document.as_str())
            .collect();
        rows.push_str(&format!(
            "  '{query}' -> {target_doc} rank {rank} | top3 = {top3:?}\n"
        ));
        reciprocal_ranks.push(1.0 / rank as f64);
    }
    let mean_rr = reciprocal_ranks.iter().sum::<f64>() / reciprocal_ranks.len() as f64;
    println!("paraphrase probe (LSA dense channel):\n{rows}mean reciprocal rank = {mean_rr:.3}");
    assert!(
        rank_violations.is_empty(),
        "targets must rank top-3; violations: {:?}\n{rows}",
        rank_violations
    );
    assert!(
        mean_rr >= 0.40,
        "paraphrase-transfer MRR floor: got {mean_rr:.3}"
    );
}

/// Regression canary (issue #8 acceptance): the probe must FAIL without
/// semantics. Same pairs, hashing fallback (purely lexical dense channel):
/// targets share zero tokens with queries, so they surface only by
/// trigram-collision noise. If a future change degrades the trained model
/// to fallback-grade behavior, the semantic floors above fail — this test
/// proves that discrimination is real, not assumed.
#[test]
fn paraphrase_probe_discriminates_semantics_from_hashing_fallback() {
    let semantic = semantic_engine();
    let fallback = fallback_engine();
    let mut sem_rr = Vec::new();
    let mut fb_rr = Vec::new();
    let mut fb_top3_misses = 0usize;
    let mut rows = String::new();
    for (query, target_doc, _) in PAIRS {
        let r_sem = dense_target_rank(&semantic, query, target_doc)
            .unwrap_or_else(|| panic!("semantic: target {target_doc} missing for '{query}'"));
        let r_fb = dense_target_rank(&fallback, query, target_doc);
        sem_rr.push(1.0 / r_sem as f64);
        match r_fb {
            Some(rank) => {
                fb_rr.push(1.0 / rank as f64);
                if rank > 3 {
                    fb_top3_misses += 1;
                }
                rows.push_str(&format!(
                    "  '{query}': lsa rank {r_sem} | fallback rank {rank}\n"
                ));
            }
            None => {
                fb_rr.push(0.0);
                fb_top3_misses += 1;
                rows.push_str(&format!(
                    "  '{query}': lsa rank {r_sem} | fallback rank >24 (miss)\n"
                ));
            }
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let sem_mrr = mean(&sem_rr);
    let fb_mrr = mean(&fb_rr);
    println!(
        "discrimination canary:\n{rows}lsa MRR = {sem_mrr:.3} | fallback MRR = {fb_mrr:.3} | fallback top-3 misses = {fb_top3_misses}/8"
    );
    assert!(
        sem_mrr >= fb_mrr + 0.20,
        "trained LSA must beat the hashing fallback by a clear margin: {sem_mrr:.3} vs {fb_mrr:.3}"
    );
    assert!(
        fb_top3_misses >= 3,
        "fallback must miss the top-3 floor on several pairs for the probe to discriminate; got {fb_top3_misses}"
    );
}
