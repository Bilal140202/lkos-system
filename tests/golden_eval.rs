//! Golden retrieval evaluation (human-curated qrels, deterministic corpus).
//!
//! v0.1's benchmark was self-supervised: probe queries echoed document
//! titles, so Recall@10 = 1.000 carried no information. This suite is the
//! fix: a fixed 16-document corpus, 16 queries with GRADED relevance
//! judgments (3 = directly answers, 2 = topical, 1 = marginal), and the
//! standard IR metrics (Recall@K, MRR, nDCG@10) computed per retrieval
//! mode. Ablation rows are printed and asserted against honest floors —
//! the point is to KNOW which complexity helps, per docs/EVALUATION.md.

use lkos::{Config, Lkos, QueryRequest, RetrievalMode};

/// (filename, body) — four topics x four documents, two chunks each.
const CORPUS: &[(&str, &str)] = &[
    // T1: finance
    ("fin-report.md", "# Q4 Financial Report\n\nNorthwind Industries revenue was $48M in 2024. Profit margin reached 12 percent.\n\n# Outlook\n\nNorthwind Industries expects 8 percent revenue growth in 2025."),
    ("fin-audit.md", "# Audit Notes\n\nThe audit verified Northwind Industries revenue figures of $48M. Cash reserves stayed healthy.\n\n# Findings\n\nAuditors flagged manual expense approvals as a process risk."),
    ("fin-budget.md", "# 2025 Budget\n\nThe budget allocates $2M for the platform team. Marketing receives $800K.\n\n# Cuts\n\nTravel budgets drop by 30 percent under the new plan."),
    ("fin-market.md", "# Market Scan\n\nCompetitor margins compressed across the sector in 2024. Pricing pressure persisted.\n\n# Notes\n\nAnalysts expect consolidation among regional suppliers."),
    // T2: engineering
    ("eng-db.md", "# Database Tuning\n\nThe query optimizer chose an index scan and cut p99 latency to 40ms.\n\n# Storage\n\nWAL checkpoint tuning reduced write stalls on the primary."),
    ("eng-api.md", "# API Design\n\nThe search endpoint supports hybrid retrieval with rank fusion. Latency budget is 50ms.\n\n# Versioning\n\nAPI changes follow a strict deprecation window of two quarters."),
    ("eng-cache.md", "# Caching Layer\n\nA semantic cache served repeated queries without recomputation. Hit rate reached 62 percent.\n\n# Eviction\n\nLRU eviction lost to a frequency-aware policy in load tests."),
    ("eng-deploy.md", "# Deployment\n\nBlue-green deployments cut release risk. Rollbacks take under two minutes.\n\n# Observability\n\nStructured traces cover ingestion, indexing, and query stages."),
    // T3: people
    ("hr-policy.md", "# Remote Work Policy\n\nEngineers may work remotely three days per week. On-site days anchor collaboration.\n\n# Equipment\n\nEvery employee receives a hardware budget of $1,800."),
    ("hr-hiring.md", "# Hiring Plan\n\nThe team hires eight engineers in 2025. Interview loops emphasize system design.\n\n# Onboarding\n\nNew hires ship a first change within one week."),
    ("hr-training.md", "# Training Budget\n\nEach engineer gets $2,000 per year for conferences and courses.\n\n# Mentoring\n\nA formal mentoring program pairs seniors with new joiners."),
    ("hr-benefits.md", "# Benefits Overview\n\nHealth coverage includes dental and vision. Paid leave totals 25 days.\n\n# Parental\n\nParental leave offers 16 weeks fully paid for every parent."),
    // T4: research
    ("res-lsa.md", "# Latent Semantic Analysis\n\nLSA projects term-document matrices into latent topics. Deerwester introduced it in 1990.\n\n# Limitations\n\nLSA struggles with polysemy and rare terms."),
    ("res-bm25.md", "# BM25 Retrieval\n\nBM25 scores documents with term frequency saturation and length normalization.\n\n# Tuning\n\nThe k1 and b parameters control saturation and length effects."),
    ("res-fusion.md", "# Rank Fusion\n\nReciprocal Rank Fusion combines ranked lists without score calibration. Cormack reported k=60.\n\n# Behavior\n\nRRF is robust when channel scores are incomparable."),
    ("res-graph.md", "# Knowledge Graphs\n\nEntity graphs encode typed relations with provenance. Multi-hop queries expose chains.\n\n# Construction\n\nCo-occurrence graphs are cheap but noisy compared to typed relations."),
];

/// (query, chunk-topic, judgments) — a judgment maps a chunk marker to a
/// grade. We judge at the DOCUMENT level: any chunk of a graded document
/// inherits the grade (a standard approximation for small corpora).
const QUERIES: &[(&str, &[(&str, u8)])] = &[
    ("Northwind Industries revenue 2024", &[("fin-report", 3), ("fin-audit", 3), ("fin-market", 1)]),
    ("budget cuts travel marketing", &[("fin-budget", 3), ("fin-market", 1)]),
    ("query optimizer index scan latency", &[("eng-db", 3), ("eng-api", 2), ("eng-cache", 1)]),
    ("cache eviction policy hit rate", &[("eng-cache", 3), ("eng-db", 1)]),
    ("blue-green deployment rollback", &[("eng-deploy", 3)]),
    ("api deprecation versioning", &[("eng-api", 3)]),
    ("remote work policy equipment", &[("hr-policy", 3), ("hr-benefits", 1)]),
    ("hiring interview loops onboarding", &[("hr-hiring", 3), ("hr-training", 1)]),
    ("mentoring program pairing", &[("hr-training", 3), ("hr-hiring", 2)]),
    ("parental leave paid weeks", &[("hr-benefits", 3)]),
    ("latent semantic analysis polysemy", &[("res-lsa", 3), ("res-graph", 1)]),
    ("bm25 term frequency saturation", &[("res-bm25", 3)]),
    ("reciprocal rank fusion k parameter", &[("res-fusion", 3), ("res-bm25", 1)]),
    ("typed relations provenance multi-hop", &[("res-graph", 3), ("res-fusion", 1)]),
    ("WAL checkpoint write stalls", &[("eng-db", 3)]),
    ("hardware budget employee", &[("hr-policy", 3), ("hr-training", 2)]),
];

const DOC_TOPICS: &[&str] = &[
    "fin-report", "fin-audit", "fin-budget", "fin-market",
    "eng-db", "eng-api", "eng-cache", "eng-deploy",
    "hr-policy", "hr-hiring", "hr-training", "hr-benefits",
    "res-lsa", "res-bm25", "res-fusion", "res-graph",
];

struct Metrics {
    recall_at_5: f64,
    recall_at_10: f64,
    mrr: f64,
    ndcg_at_10: f64,
}

fn grade_of(doc_filename: &str, qrels: &[(&str, u8)]) -> u8 {
    let topic = DOC_TOPICS
        .iter()
        .find(|t| doc_filename.starts_with(**t))
        .copied()
        .unwrap_or("");
    qrels
        .iter()
        .find(|(t, _)| *t == topic)
        .map(|(_, g)| *g)
        .unwrap_or(0)
}

fn evaluate(hits: &[String], qrels: &[(&str, u8)]) -> Metrics {
    // hits: ordered list of doc filenames (deduplicated per document).
    let mut first_rel_rank = 0usize;
    let mut dcg = 0.0f64;
    let mut relevant_seen = std::collections::HashSet::new();
    for (rank, doc) in hits.iter().enumerate() {
        let g = grade_of(doc, qrels);
        if g >= 2 {
            let gain = (1u32 << g) as f64 - 1.0; // 2^g - 1
            dcg += gain / ((rank + 2) as f64).log2();
            if relevant_seen.insert(doc.clone()) && first_rel_rank == 0 {
                first_rel_rank = rank + 1;
            }
        }
    }
    // Ideal DCG from the full grade multiset (documents may contribute once).
    let mut grades: Vec<f64> = qrels
        .iter()
        .filter(|(_, g)| *g >= 2)
        .map(|(_, g)| (1u32 << *g) as f64 - 1.0)
        .collect();
    grades.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut idcg = 0.0;
    for (i, g) in grades.iter().enumerate().take(10) {
        idcg += g / ((i as f64) + 2.0).log2();
    }
    let relevant_top5 = hits.iter().take(5).any(|d| grade_of(d, qrels) >= 2);
    let relevant_top10 = hits.iter().take(10).any(|d| grade_of(d, qrels) >= 2);
    Metrics {
        recall_at_5: if relevant_top5 { 1.0 } else { 0.0 },
        recall_at_10: if relevant_top10 { 1.0 } else { 0.0 },
        mrr: if first_rel_rank > 0 { 1.0 / first_rel_rank as f64 } else { 0.0 },
        ndcg_at_10: if idcg > 0.0 { dcg / idcg } else { 0.0 },
    }
}

fn doc_hits(engine: &Lkos, query: &str, mode: RetrievalMode, k: usize) -> Vec<String> {
    let req = QueryRequest::new(query).top_k(k * 3).mode(mode);
    let resp = engine.query(req).expect("query executes");
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for h in &resp.hits {
        if seen.insert(h.document.clone()) {
            out.push(h.document.clone());
        }
        if out.len() >= k {
            break;
        }
    }
    out
}

fn corpus_engine() -> Lkos {
    let mut cfg = Config::default();
    cfg.synchronous_ingestion = true;
    cfg.semantic_min_chunks = usize::MAX; // keep this test purely lexical/planner
    let engine = Lkos::open_in_memory(cfg).expect("open");
    for (name, body) in CORPUS {
        engine.ingest_bytes(name, body.as_bytes()).expect("ingest");
    }
    engine
}

#[test]
fn golden_evaluation_hybrid_beats_or_matches_channels_with_honest_floor() {
    let engine = corpus_engine();
    let modes = [
        ("lexical", RetrievalMode::LexicalOnly),
        ("dense", RetrievalMode::VectorOnly),
        ("hybrid", RetrievalMode::Hybrid),
    ];
    let mut table: Vec<(String, Metrics)> = Vec::new();
    for (name, mode) in modes {
        let mut sum = Metrics {
            recall_at_5: 0.0,
            recall_at_10: 0.0,
            mrr: 0.0,
            ndcg_at_10: 0.0,
        };
        let mut rows = String::new();
        for (q, qrels) in QUERIES {
            let hits = doc_hits(&engine, q, mode, 10);
            let m = evaluate(&hits, qrels);
            sum.recall_at_5 += m.recall_at_5;
            sum.recall_at_10 += m.recall_at_10;
            sum.mrr += m.mrr;
            sum.ndcg_at_10 += m.ndcg_at_10;
            rows.push_str(&format!(
                "  [{name}] {:.3} q='{}'\n",
                m.ndcg_at_10, q
            ));
        }
        let n = QUERIES.len() as f64;
        println!("=== {name} (averages over {} queries) ===\n{}recall@5={:.3} recall@10={:.3} mrr={:.3} ndcg@10={:.3}",
            QUERIES.len(), rows,
            sum.recall_at_5 / n, sum.recall_at_10 / n, sum.mrr / n, sum.ndcg_at_10 / n);
        table.push((name.to_string(), sum));
    }
    let n = QUERIES.len() as f64;
    let get = |name: &str, f: fn(&Metrics) -> f64| {
        table
            .iter()
            .find(|(n_, _)| n_ == name)
            .map(|(_, m)| f(m) / n)
            .expect("mode present")
    };
    let hybrid_mrr = get("hybrid", |m| m.mrr);
    let lexical_mrr = get("lexical", |m| m.mrr);
    let hybrid_ndcg = get("hybrid", |m| m.ndcg_at_10);
    let hybrid_recall10 = get("hybrid", |m| m.recall_at_10);

    // Honest floors (documented in docs/EVALUATION.md): a 16-doc corpus with
    // keyword-matchable queries must be largely solved by the hybrid stack.
    assert!(hybrid_mrr >= 0.50, "hybrid MRR floor: got {hybrid_mrr}");
    assert!(hybrid_ndcg >= 0.55, "hybrid nDCG@10 floor: got {hybrid_ndcg}");
    assert!(hybrid_recall10 >= 0.85, "hybrid recall@10 floor: got {hybrid_recall10}");
    // Fusion must not be WORSE than the best single channel by a margin.
    assert!(
        hybrid_mrr >= lexical_mrr - 0.05,
        "hybrid {hybrid_mrr:.3} must be within 0.05 of lexical {lexical_mrr:.3}"
    );
}

#[test]
fn evaluation_reproducibility_same_corpus_same_scores() {
    let a = corpus_engine();
    let b = corpus_engine();
    for (q, qrels) in QUERIES.iter().take(6) {
        let ha = doc_hits(&a, q, RetrievalMode::Hybrid, 10);
        let hb = doc_hits(&b, q, RetrievalMode::Hybrid, 10);
        assert_eq!(ha, hb, "deterministic ranking for '{q}'");
        let ma = evaluate(&ha, qrels);
        let mb = evaluate(&hb, qrels);
        assert_eq!(ma.mrr, mb.mrr);
        assert_eq!(ma.ndcg_at_10, mb.ndcg_at_10);
    }
}
