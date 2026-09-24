//! LKOS benchmark harness.
//!
//! Measures, on this machine, with no network and no external models:
//!
//! 1. **Ingestion throughput** — documents/second and chunks/second for a
//!    synthetic corpus of configurable size.
//! 2. **Query latency** — p50 / p95 / p99 over a mixed-intent query set,
//!    separated per retrieval mode (lexical / vector / hybrid).
//! 3. **Retrieval quality** — Recall@K and MRR against a synthetic corpus
//!    where the generating document of each probe query is known ground
//!    truth (a standard self-supervised retrieval check; it validates the
//!    fusion stack end-to-end, it does not replace a human-labeled set).
//!
//! Usage:
//! ```text
//! lkos-bench [--docs N] [--queries N] [--k 10] [--json]
//! ```
//!
//! Exit code is 0 on success. Results are printed as a table (or JSON with
//! `--json`) so they can be archived in `benchmarks/results/`.

use std::time::Instant;

use lkos::{Config, Lkos, QueryRequest, RetrievalMode};

#[derive(Clone, Copy)]
struct Args {
    docs: usize,
    queries: usize,
    k: usize,
    json: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        docs: 200,
        queries: 100,
        k: 10,
        json: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--docs" => args.docs = it.next().and_then(|v| v.parse().ok()).unwrap_or(args.docs),
            "--queries" => {
                args.queries = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.queries)
            }
            "--k" => args.k = it.next().and_then(|v| v.parse().ok()).unwrap_or(args.k),
            "--json" => args.json = true,
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: lkos-bench [--docs N] [--queries N] [--k 10] [--json]");
                std::process::exit(2);
            }
        }
    }
    args
}

// ---------------------------------------------------------------------------
// Synthetic corpus generation
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random generator (xorshift64*). We avoid external
/// RNG crates so the benchmark is reproducible from the dependency graph.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const TOPICS: &[&str] = &[
    "authentication",
    "database migrations",
    "payment processing",
    "vacation policy",
    "invoice handling",
    "vector search",
    "entity resolution",
    "background jobs",
    "backup procedures",
    "rate limiting",
    "audit logging",
    "data retention",
];

const VERBS: &[&str] = &[
    "uses",
    "requires",
    "supports",
    "documents",
    "measures",
    "configures",
    "monitors",
    "reviews",
];

const ORGS: &[&str] = &[
    "Acme Corp",
    "Northwind Ltd",
    "Globex Inc",
    "Initech LLC",
    "Umbrella Group",
];

const YEARS: &[u32] = &[2021, 2022, 2023, 2024, 2025, 2026];

/// Generate one synthetic markdown document. The document's identity is a
/// (topic, org, year) triple; probe queries later cite the triple so the
/// ground-truth document is known. v0.9: paragraphs include SVO claim
/// sentences ("{org} revenue was ${n}M in {year}") so the claim extractor
/// and conflict engine are exercised too (v0.1's corpus yielded 0 claims).
fn make_document(rng: &mut Rng, idx: usize) -> (String, String, String) {
    let topic = TOPICS[rng.below(TOPICS.len())];
    let org = ORGS[rng.below(ORGS.len())];
    let year = YEARS[rng.below(YEARS.len())];
    let mut body = String::with_capacity(8_192);
    body.push_str(&format!(
        "# {org} {topic} handbook {year}\n\n\
         This document describes how {org} handles {topic} as of {year}.\n\n"
    ));
    for p in 0..12 {
        let verb = VERBS[rng.below(VERBS.len())];
        let amount = 10_000 + rng.below(90_000);
        let revenue = 5 + rng.below(60);
        let launched = format!(
            "Project {}",
            ["Aurora", "Borealis", "Cascade", "Delta", "Echo", "Fusion"][rng.below(6)]
        );
        body.push_str(&format!(
            "## Section {p}: {topic} procedure {p}\n\n\
             {org} {verb} structured review for {topic} procedure {p} in {year}. \
             {org} revenue was ${revenue}M in {year} per the {topic} filing. \
             {org} launched {launched} in {year}. \
             The approved budget for this procedure is ${amount} per quarter. \
             Owners must record every decision in the audit log and annotate \
             the retention class before the quarterly {topic} review closes.\n\n"
        ));
    }
    let filename = {
        let topic_slug = topic.replace(' ', "-");
        let org_slug = org.split(' ').next().unwrap_or("org").to_lowercase();
        format!("bench-{idx:04}-{topic_slug}-{org_slug}-{year}.md")
    };
    (filename, body, format!("{org} {topic} {year}"))
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn fmt_us(us: f64) -> String {
    if us >= 1_000.0 {
        format!("{:.2} ms", us / 1_000.0)
    } else {
        format!("{:.0} us", us)
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_ingestion(engine: &Lkos, args: Args) -> (f64, usize, usize) {
    let mut rng = Rng::new(0x1057 ^ 42);
    let mut total_chunks = 0usize;
    let mut total_chars = 0usize;
    let t0 = Instant::now();
    for i in 0..args.docs {
        let (filename, body, _key) = make_document(&mut rng, i);
        let info = engine
            .ingest_bytes(&filename, body.as_bytes())
            .expect("ingest failed");
        total_chunks += info.chunk_count as usize;
        total_chars += info.text_chars;
    }
    let secs = t0.elapsed().as_secs_f64();
    (secs, total_chunks, total_chars)
}

fn bench_latency(engine: &Lkos, args: Args, mode: RetrievalMode) -> Vec<f64> {
    let mut rng = Rng::new(0xBE_C0 ^ (mode as u64 + 7));
    let mut lat_us = Vec::with_capacity(args.queries);
    // Warm-up (compiles FTS query plans, populates SQLite page cache).
    for _ in 0..5 {
        let _ = engine.query(QueryRequest::new("authentication policy").mode(mode));
    }
    for _ in 0..args.queries {
        let topic = TOPICS[rng.below(TOPICS.len())];
        let org = ORGS[rng.below(ORGS.len())];
        let q = format!("{org} {topic}");
        let t0 = Instant::now();
        let resp = engine
            .query(QueryRequest::new(q).mode(mode).top_k(args.k))
            .expect("query failed");
        lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
        let _ = resp;
    }
    lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lat_us
}

/// Recall@K + MRR against the synthetic ground truth.
///
/// Ground truth = the filename(s) generated from the same (topic, org, year)
/// triple as the probe query. The corpus list is regenerated deterministically
/// from the same seed used by `bench_ingestion`, so every probe has a known
/// expected document (or several, on triple collisions).
fn bench_quality(engine: &Lkos, args: Args) -> (f64, f64) {
    // Rebuild the exact corpus manifest (same seed, same generation order).
    let mut rng = Rng::new(0x1057 ^ 42);
    let mut by_triple: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for i in 0..args.docs {
        let (filename, _body, key) = make_document(&mut rng, i);
        by_triple.entry(key).or_default().push(filename);
    }

    let mut rng_q = Rng::new(0x0A11);
    let keys: Vec<&String> = by_triple.keys().collect();
    if keys.is_empty() {
        return (0.0, 0.0);
    }
    let mut recall_sum = 0.0f64;
    let mut mrr_sum = 0.0f64;
    let mut counted = 0usize;
    for _ in 0..args.queries {
        let key = keys[rng_q.below(keys.len())].clone();
        let expected = &by_triple[&key];
        let resp = engine
            .query(QueryRequest::new(key.as_str()).top_k(args.k))
            .expect("query failed");
        if let Some(rank) = resp
            .hits
            .iter()
            .position(|h| expected.contains(&h.document))
        {
            recall_sum += 1.0;
            mrr_sum += 1.0 / (rank + 1) as f64;
        }
        counted += 1;
    }
    if counted == 0 {
        return (0.0, 0.0);
    }
    (recall_sum / counted as f64, mrr_sum / counted as f64)
}

#[allow(clippy::field_reassign_with_default)]
fn main() {
    // `ann` subcommand: dense-channel crossover benchmark (v0.10, issue #2).
    if std::env::args().nth(1).as_deref() == Some("ann") {
        ann_crossover();
        return;
    }
    let args = parse_args();
    println!("LKOS benchmark");
    println!(
        "  docs: {}  queries: {}  k: {}",
        args.docs, args.queries, args.k
    );
    println!();

    let mut cfg = Config::default();
    cfg.semantic_min_chunks = 24;
    cfg.lsa_dim = 64;
    let engine = Lkos::open_in_memory(cfg).expect("engine");
    engine.set_llm(std::sync::Arc::new(lkos::llm::NullProvider));

    // -- ingestion ---------------------------------------------------------
    let (secs, chunks, chars) = bench_ingestion(&engine, args);
    let docs_per_s = args.docs as f64 / secs;
    let chunks_per_s = chunks as f64 / secs;

    // -- semantic training (LSA) -------------------------------------------
    let train_started = Instant::now();
    let trained = engine.train_semantic_index().expect("train");
    let train_secs = train_started.elapsed().as_secs_f64();
    let reembed_started = Instant::now();
    let migrated = engine
        .reembed_stale_chunks("lsa-pmi-svd-v1", None)
        .expect("reembed");
    let reembed_secs = reembed_started.elapsed().as_secs_f64();

    // -- latency per mode (post-training: dense channel = LSA) --------------
    let lex = bench_latency(&engine, args, RetrievalMode::LexicalOnly);
    let vec_ = bench_latency(&engine, args, RetrievalMode::VectorOnly);
    let hyb = bench_latency(&engine, args, RetrievalMode::Hybrid);

    // -- quality (hybrid) ----------------------------------------------------
    let (recall, mrr) = bench_quality(&engine, args);

    let stats = engine.stats().expect("stats");

    if args.json {
        println!(
            "{{\"docs\":{},\"chunks\":{},\"ingest_s\":{:.3},\"docs_per_s\":{:.1},\"chunks_per_s\":{:.1},\
             \"lex_p50_us\":{:.0},\"lex_p95_us\":{:.0},\"lex_p99_us\":{:.0},\
             \"vec_p50_us\":{:.0},\"vec_p95_us\":{:.0},\"vec_p99_us\":{:.0},\
             \"hyb_p50_us\":{:.0},\"hyb_p95_us\":{:.0},\"hyb_p99_us\":{:.0},\
             \"recall_at_k\":{:.3},\"mrr\":{:.3},\"k\":{},\
             \"entities\":{},\"claims\":{},\"relationships\":{},\"provenance\":{}}}",
            args.docs,
            chunks,
            secs,
            docs_per_s,
            chunks_per_s,
            percentile(&lex, 50.0),
            percentile(&lex, 95.0),
            percentile(&lex, 99.0),
            percentile(&vec_, 50.0),
            percentile(&vec_, 95.0),
            percentile(&vec_, 99.0),
            percentile(&hyb, 50.0),
            percentile(&hyb, 95.0),
            percentile(&hyb, 99.0),
            recall,
            mrr,
            args.k,
            stats.entities,
            stats.claims,
            stats.relationships,
            stats.provenance_records,
        );
        return;
    }

    println!("== Ingestion ==");
    println!(
        "  corpus           : {} documents, {} chunks, {} chars",
        args.docs, chunks, chars
    );
    println!("  wall time        : {:.2} s", secs);
    println!(
        "  throughput       : {:.1} docs/s, {:.0} chunks/s",
        docs_per_s, chunks_per_s
    );
    println!();
    println!("== Semantic index (LSA) ==");
    if trained {
        println!(
            "  trained on      : {} chunks in {:.2} s",
            chunks, train_secs
        );
        println!(
            "  re-embedded     : {} chunks in {:.2} s ({:.0} chunks/s)",
            migrated,
            reembed_secs,
            migrated as f64 / reembed_secs.max(1e-9)
        );
    } else {
        println!("  training skipped (corpus below threshold)");
    }
    println!();
    println!("== Query latency ({} queries per mode) ==", args.queries);
    for (name, xs) in [
        ("lexical (BM25)", &lex),
        ("vector (LSA cosine)", &vec_),
        ("hybrid (RRF+rerank)", &hyb),
    ] {
        println!(
            "  {:<20} p50 {:>10}   p95 {:>10}   p99 {:>10}",
            name,
            fmt_us(percentile(xs, 50.0)),
            fmt_us(percentile(xs, 95.0)),
            fmt_us(percentile(xs, 99.0)),
        );
    }
    println!();
    println!("== Retrieval quality (hybrid, self-supervised) ==");
    println!("  Recall@{}        : {:.3}", args.k, recall);
    println!("  MRR              : {:.3}", mrr);
    println!();
    println!("== Library stats ==");
    println!(
        "  entities {} | mentions {} | claims {} | conflicts {} | relationships {} | provenance {}",
        stats.entities,
        stats.entity_mentions,
        stats.claims,
        stats.conflicts,
        stats.relationships,
        stats.provenance_records
    );
    println!();
    println!("Note: quality numbers are self-supervised (ground truth = generating document).");
    println!("They validate the fusion stack end-to-end; they are not a human benchmark.");
}

// ---------------------------------------------------------------------------
// ANN crossover benchmark (v0.10, GitHub issue #2)
//
// Dense-channel comparison: brute-force cosine scan vs deterministic HNSW.
// Corpus: clustered unit-space vectors (topical clusters), dim = 128 (the
// default LSA latent size). Queries jitter actual cluster centers — the
// regime the dense channel actually serves. Ground truth = exact scan.
// Everything is deterministic (fixed seeds, xorshift64*).
// ---------------------------------------------------------------------------

fn rng_f32(rng: &mut Rng) -> f32 {
    ((rng.next() >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
}

fn ann_crossover() {
    const DIM: usize = 128; // default LSA latent dimensionality
    const CLUSTERS: usize = 40; // topical clusters
    const JITTER: f32 = 0.05;
    const QUERIES: usize = 30;
    const M: usize = 16;
    const EF_CONSTRUCTION: usize = 200; // matches the engine default (ADR-010)
    const EFS: [usize; 3] = [32, 64, 128];
    const SIZES: [usize; 5] = [1_000, 5_000, 20_000, 50_000, 100_000];

    println!("LKOS ANN crossover benchmark (dense channel: brute vs HNSW)");
    println!(
        "  dim={DIM} clusters={CLUSTERS} jitter={JITTER} queries={QUERIES} \
         M={M} efC={EF_CONSTRUCTION} ef_search={EFS:?}"
    );
    println!("  seeded, deterministic; ground truth = exact cosine scan");
    println!();

    let mut rng = Rng::new(0x0A77_C0DE);
    let centers: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|_| (0..DIM).map(|_| rng_f32(&mut rng)).collect())
        .collect();

    for &n in &SIZES {
        let per = n / CLUSTERS;
        // Clustered corpus + jittered-center queries (the realistic regime).
        let mut pairs: Vec<(i64, Vec<f32>)> = Vec::with_capacity(n);
        let mut id = 0i64;
        for center in &centers {
            for _ in 0..per {
                let v: Vec<f32> = center
                    .iter()
                    .map(|&x| x + JITTER * rng_f32(&mut rng))
                    .collect();
                pairs.push((id, v));
                id += 1;
            }
        }
        let queries: Vec<Vec<f32>> = (0..QUERIES)
            .map(|i| {
                centers[i % CLUSTERS]
                    .iter()
                    .map(|&x| x + 0.1 * rng_f32(&mut rng))
                    .collect()
            })
            .collect();

        // --- brute force: latency + ground truth --------------------------
        let mut brute_us: Vec<f64> = Vec::with_capacity(QUERIES);
        let mut truth: Vec<Vec<(i64, f32)>> = Vec::with_capacity(QUERIES);
        for q in &queries {
            let t0 = Instant::now();
            let mut s: Vec<(i64, f32)> = pairs
                .iter()
                .map(|(cid, v)| (*cid, lkos::embeddings::cosine(q, v)))
                .filter(|(_, x)| *x > 0.0)
                .collect();
            s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
            s.truncate(10);
            brute_us.push(t0.elapsed().as_micros() as f64);
            truth.push(s);
        }
        brute_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = pairs.len();

        // --- HNSW build ----------------------------------------------------
        let build_t0 = Instant::now();
        let index = lkos::ann::HnswIndex::build(pairs.clone(), M, EF_CONSTRUCTION);
        let build_secs = build_t0.elapsed().as_secs_f64();
        let stats = index.stats();

        // --- HNSW search per ef -------------------------------------------
        print!(
            "N = {n:>7} chunks\n  brute p50 {:>9}  p95 {:>9}   (exact)\n  hnsw build {:.2} s  edges {}  mem ~{:.1} MiB\n",
            fmt_us(percentile(&brute_us, 50.0)),
            fmt_us(percentile(&brute_us, 95.0)),
            build_secs,
            stats.undirected_edges,
            stats.memory_estimate_bytes as f64 / (1024.0 * 1024.0),
        );

        let mut ef_rows: Vec<String> = Vec::new();
        for &ef in &EFS {
            let mut ann_us: Vec<f64> = Vec::with_capacity(QUERIES);
            let mut recall_sum = 0.0f64;
            for (qi, q) in queries.iter().enumerate() {
                let t0 = Instant::now();
                let got = index.search(q, 10, ef);
                ann_us.push(t0.elapsed().as_micros() as f64);
                let truth_ids: std::collections::HashSet<i64> =
                    truth[qi].iter().map(|(cid, _)| *cid).collect();
                let got_ids: std::collections::HashSet<i64> =
                    got.iter().map(|(cid, _)| *cid).collect();
                recall_sum += truth_ids.intersection(&got_ids).count() as f64 / 10.0;
            }
            ann_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ef_rows.push(format!(
                "  ef={ef:>3}: p50 {:>9}  p95 {:>9}  recall@10 {:.3}",
                fmt_us(percentile(&ann_us, 50.0)),
                fmt_us(percentile(&ann_us, 95.0)),
                recall_sum / QUERIES as f64,
            ));
        }
        for row in &ef_rows {
            println!("{row}");
        }

        // --- deletion cost: drop 10%, rebuild ------------------------------
        let survivors: Vec<(i64, Vec<f32>)> = pairs
            .iter()
            .filter(|(cid, _)| cid % 10 != 7)
            .cloned()
            .collect();
        let rebuild_t0 = Instant::now();
        let _rebuilt = lkos::ann::HnswIndex::build(survivors, M, EF_CONSTRUCTION);
        println!(
            "  delete 10% + rebuild {:.2} s   (invalidation strategy: full rebuild; ADR-010)",
            rebuild_t0.elapsed().as_secs_f64()
        );

        // Amortization: how many queries until the build pays for itself?
        let brute_p50 = percentile(&brute_us, 50.0);
        let ann_p50 = percentile(
            &{
                // re-derive from ef=64 row: reuse last measured latency list
                // (kept simple: rerun one batch at ef=64)
                let mut us = Vec::with_capacity(QUERIES);
                for q in &queries {
                    let t0 = Instant::now();
                    let _ = index.search(q, 10, 64);
                    us.push(t0.elapsed().as_micros() as f64);
                }
                us.sort_by(|a, b| a.partial_cmp(b).unwrap());
                us
            },
            50.0,
        );
        let saved_us = brute_p50 - ann_p50;
        if saved_us > 0.0 {
            let q_break_even = (build_secs * 1_000_000.0 / saved_us).ceil() as u64;
            println!("  break-even: ~{q_break_even} dense queries after build (at ef=64)");
        } else {
            println!("  break-even: ANN slower than brute at this N for ef=64");
        }
        println!();
    }
    println!("Decision inputs for ADR-010: see docs/adr/ADR.md and docs/RETRIEVAL.md.");
}
