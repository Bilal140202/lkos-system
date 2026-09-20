//! Query engine: intent classification, planning, execution, context assembly.
//!
//! Deterministic rule-based planner (ADR-008): no LLM in the query path.
//! The plan chooses channels, weights, entity lookups, and summary fast-paths
//! based on observable query characteristics. Every plan is explained.

use crate::config::Config;
use crate::types::*;

/// Classify the intent of a query (deterministic heuristics).
pub fn classify_intent(query: &str) -> QueryIntent {
    let lower = query.to_lowercase();
    let trimmed = lower.trim();

    // Summary intents.
    for kw in [
        "summarize", "summarise", "summary of", "overview of", "tl;dr", "tldr",
        "what is this document about", "main points of",
    ] {
        if trimmed.starts_with(kw) || trimmed.contains(kw) {
            return QueryIntent::Summary;
        }
    }

    // Comparative intents.
    if ["compare", "difference between", "versus", " vs ", "contradict"]
        .iter()
        .any(|k| trimmed.contains(k))
    {
        return QueryIntent::Comparative;
    }

    // Temporal intents.
    let temporal_words = ["when", "latest", "most recent", "before", "after", "since", "until"];
    let has_year = query.split_whitespace().any(|w| {
        w.trim_matches(|c: char| !c.is_ascii_digit())
            .parse::<i32>()
            .map(|y| (1900..=2100).contains(&y))
            .unwrap_or(false)
    });
    if has_year || temporal_words.iter().any(|k| trimmed.starts_with(k) || trimmed.contains(k)) {
        return QueryIntent::Temporal;
    }

    // Exact intents: quoted phrase, ALLCAPS token, identifier-ish token, file name.
    if query.contains('"') {
        return QueryIntent::Exact;
    }
    for tok in query.split_whitespace() {
        let core = tok.trim_matches(|c: char| !c.is_alphanumeric());
        if core.len() > 2
            && core.chars().all(|c| c.is_ascii_uppercase() || c == '_')
            && core.chars().any(|c| c.is_ascii_alphabetic())
        {
            return QueryIntent::Exact;
        }
        let looks_identifier =
            core.contains('_') || (core.chars().any(|c| c.is_ascii_uppercase()) && core.chars().any(|c| c.is_ascii_lowercase()) && core.len() > 6);
        if looks_identifier {
            return QueryIntent::Exact;
        }
        if core.contains('.') && !core.contains(' ') {
            return QueryIntent::Exact;
        }
    }

    // Entity intents.
    if trimmed.starts_with("who is")
        || trimmed.starts_with("who was")
        || trimmed.starts_with("what is ")
            && entity_case_heuristic(query)
        || trimmed.contains("everything about")
        || trimmed.contains("documents about")
    {
        return QueryIntent::Entity;
    }

    QueryIntent::Semantic
}

fn entity_case_heuristic(q: &str) -> bool {
    // "What is OpenAI" (capitalized non-first word) → entity-ish.
    q.split_whitespace()
        .skip(2)
        .any(|w| w.chars().next().is_some_and(|c| c.is_uppercase()))
}

/// The execution plan produced by the planner.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Detected/forced intent.
    pub intent: QueryIntent,
    /// Retrieval mode actually used.
    pub mode: RetrievalMode,
    /// Dense channel weight.
    pub w_vector: f32,
    /// Lexical channel weight.
    pub w_fts: f32,
    /// Use the pre-built summary fast-path?
    pub use_summary: bool,
    /// Look up entity index?
    pub use_entities: bool,
    /// Temporal constraint detected.
    pub temporal: crate::temporal::TemporalConstraint,
    /// Human-readable explanation.
    pub explanation: String,
}

/// Build a plan from a request + config.
pub fn plan(req: &QueryRequest, cfg: &Config) -> QueryPlan {
    let intent = match req.mode {
        RetrievalMode::Auto => classify_intent(&req.text),
        _ => QueryIntent::Semantic, // forced modes are semantic-neutral
    };
    let mode = match req.mode {
        RetrievalMode::Auto => match intent {
            QueryIntent::Entity => RetrievalMode::Hybrid,
            QueryIntent::Exact => RetrievalMode::LexicalOnly,
            QueryIntent::Summary => RetrievalMode::Hybrid,
            _ => RetrievalMode::Hybrid,
        },
        m => m,
    };
    // Weight tuning per intent (justified in RETRIEVAL.md benchmarks).
    let (wv, wf) = match intent {
        QueryIntent::Exact => (0.25, 0.75),
        QueryIntent::Temporal => (0.35, 0.65),
        _ => (cfg.weight_vector, cfg.weight_fts),
    };
    let use_summary = matches!(intent, QueryIntent::Summary);
    let use_entities = matches!(intent, QueryIntent::Entity | QueryIntent::Comparative);
    let temporal = crate::temporal::detect_temporal(&req.text);
    let explanation = format!(
        "intent={} mode={} weights(dense={wv:.2}, lexical={wf:.2}){}{}{}",
        intent.as_str(),
        mode_label(mode),
        if use_summary { ", summary fast-path ON" } else { "" },
        if use_entities { ", entity lookup ON" } else { "" },
        match temporal {
            crate::temporal::TemporalConstraint::None => String::new(),
            _ => format!(", temporal constraint {:?}", temporal),
        },
    );
    QueryPlan {
        intent,
        mode,
        w_vector: wv,
        w_fts: wf,
        use_summary,
        use_entities,
        temporal,
        explanation,
    }
}

fn mode_label(m: RetrievalMode) -> &'static str {
    match m {
        RetrievalMode::Auto => "auto",
        RetrievalMode::LexicalOnly => "lexical",
        RetrievalMode::VectorOnly => "dense",
        RetrievalMode::Hybrid => "hybrid",
        RetrievalMode::EntityLookup => "entity",
    }
}

/// Assemble a context block from hits under a character budget with
/// per-document diversity and [n] citation markers.
pub fn assemble_context(hits: &[SearchHit], budget: usize, max_per_doc: usize) -> String {
    let mut out = String::new();
    let mut per_doc: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    let mut used = 0usize;
    let mut cites = Vec::new();
    for (i, h) in hits.iter().enumerate() {
        let d = per_doc.entry(h.document_id).or_insert(0);
        if *d >= max_per_doc {
            continue;
        }
        let snippet = h.text.chars().take(700).collect::<String>();
        let block = format!("[{}] {} :: {}\n{}\n\n", i + 1, h.document, h.section.as_deref().unwrap_or("-"), snippet);
        if used + block.len() > budget {
            break;
        }
        used += block.len();
        *d += 1;
        cites.push(format!("[{}] {} :: {}", i + 1, h.document, h.section.as_deref().unwrap_or("-")));
        out.push_str(&block);
    }
    if !cites.is_empty() {
        out.push_str("CITATIONS: ");
        out.push_str(&cites.join(" | "));
        out.push('\n');
    }
    out
}
