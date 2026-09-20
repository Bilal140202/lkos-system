//! Deterministic knowledge extraction — the "neuro-symbolic" symbolic half.
//!
//! Everything in this module is regex/heuristic-based, zero-cost, and fully
//! deterministic (no LLM involved). LLM enrichment is *optional* and layered
//! on top by the background pipeline. Extracted knowledge is always labeled
//! with its extractor identity and heuristic confidence (no fake precision).

use crate::ingestion::hash_text;
use serde::{Deserialize, Serialize};

/// Extractor identity written to provenance.
pub const KNOWLEDGE_VERSION: &str = "knowledge-v1.0.0";
pub(crate) const ENTITY_EXTRACTOR: &str = "entity-heuristics-v1";
pub(crate) const CLAIM_EXTRACTOR: &str = "claim-heuristics-v1";

/// Entity type label: person.
pub const T_PERSON: &str = "person";
/// Entity type label: organization.
pub const T_ORG: &str = "organization";
/// Entity type label: location.
pub const T_LOCATION: &str = "location";
/// Entity type label: date.
pub const T_DATE: &str = "date";
/// Entity type label: monetary amount.
pub const T_MONEY: &str = "money";
/// Entity type label: percentage.
pub const T_PERCENT: &str = "percent";
/// Entity type label: email address.
pub const T_EMAIL: &str = "email";
/// Entity type label: URL.
pub const T_URL: &str = "url";
/// Entity type label: generic concept.
pub const T_CONCEPT: &str = "concept";

/// One candidate entity mention found in text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCandidate {
    /// Surface form.
    pub surface: String,
    /// Entity type label.
    pub entity_type: String,
    /// Character offsets in the scanned text.
    pub start: usize,
    /// End offset (exclusive).
    pub end: usize,
    /// Heuristic confidence.
    pub confidence: f32,
}

/// A Knowledge Object: the enrichment payload attached to every chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeObject {
    /// Entity candidates (typed).
    pub entities: Vec<EntityCandidate>,
    /// Extracted keywords with heuristic weights.
    pub keywords: Vec<(String, f32)>,
    /// ISO dates found in text.
    pub dates: Vec<String>,
    /// Extracted claims.
    pub claims: Vec<ExtractedClaim>,
}

/// A claim candidate extracted by sentence-pattern analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedClaim {
    /// Subject phrase.
    pub subject: String,
    /// Predicate phrase.
    pub predicate: String,
    /// Object phrase.
    pub object: String,
    /// Original sentence.
    pub sentence: String,
    /// Heuristic confidence.
    pub confidence: f32,
    /// Optional validity start (temporal annotation).
    pub valid_from: Option<String>,
    /// Optional validity end.
    pub valid_until: Option<String>,
    /// Sentence start offset inside the chunk text (provenance).
    pub start_offset: usize,
    /// Sentence end offset inside the chunk text (provenance).
    pub end_offset: usize,
    /// True when the sentence negates the predicate ("Acme is not profitable").
    pub negated: bool,
}

use std::sync::OnceLock;

static RE_ORG: OnceLock<regex::Regex> = OnceLock::new();
static RE_PERSON: OnceLock<regex::Regex> = OnceLock::new();
static RE_MONEY: OnceLock<regex::Regex> = OnceLock::new();
static RE_PERCENT: OnceLock<regex::Regex> = OnceLock::new();
static RE_DATE: OnceLock<regex::Regex> = OnceLock::new();
static RE_EMAIL: OnceLock<regex::Regex> = OnceLock::new();
static RE_URL: OnceLock<regex::Regex> = OnceLock::new();

fn re_org() -> &'static regex::Regex {
    RE_ORG.get_or_init(|| {
        regex::Regex::new(
            r"\b([A-Z][A-Za-z0-9&\-]*(?:\s+[A-Z][A-Za-z0-9&\-]*){0,3}?)\s+(Inc|Corp|Corporation|Ltd|Limited|LLC|LLP|GmbH|AG|PLC|Group|Holdings|University|Institute|Laboratory|Laboratories|Foundation|Association|Systems|Labs)\b",
        )
        .expect("org regex")
    })
}

/// Legal-suffix tokens that terminate an organization surface name.
const ORG_SUFFIX_TOKENS: &[&str] = &[
    "Inc", "Corp", "Corporation", "Ltd", "Limited", "LLC", "LLP", "GmbH", "AG", "PLC", "Group",
    "Holdings", "University", "Institute", "Laboratory", "Laboratories", "Foundation",
    "Association", "Systems", "Labs",
];

/// Truncate an org surface at the first legal-suffix token, keeping the
/// minimal legal name. "Omega Corp partnered with Tau Corp" → "Omega Corp".
fn truncate_at_suffix_boundary(surface: &str) -> String {
    for (i, word) in surface.split_whitespace().enumerate() {
        if i > 0 && ORG_SUFFIX_TOKENS.contains(&word) {
            return surface
                .split_whitespace()
                .take(i + 1)
                .collect::<Vec<_>>()
                .join(" ");
        }
    }
    surface.to_string()
}

fn re_person() -> &'static regex::Regex {
    RE_PERSON.get_or_init(|| {
        regex::Regex::new(r"\b([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})\b").expect("person regex")
    })
}

fn re_money() -> &'static regex::Regex {
    RE_MONEY.get_or_init(|| {
        regex::Regex::new(r"[$€£]\s?\d[\d.,]*\s?[MBK]?").expect("money regex")
    })
}

fn re_percent() -> &'static regex::Regex {
    RE_PERCENT.get_or_init(|| regex::Regex::new(r"\b\d{1,3}(?:\.\d+)?%").expect("percent regex"))
}

fn re_date() -> &'static regex::Regex {
    RE_DATE.get_or_init(|| {
        regex::Regex::new(
            r"\b(?:\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|\b(?:19|20)\d{2}\b)\b",
        )
        .expect("date regex")
    })
}

fn re_email() -> &'static regex::Regex {
    RE_EMAIL.get_or_init(|| {
        regex::Regex::new(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
            .expect("email regex")
    })
}

fn re_url() -> &'static regex::Regex {
    RE_URL.get_or_init(|| {
        regex::Regex::new(r#"\bhttps?://[^\s)>\]"']+"#).expect("url regex")
    })
}

/// Words that disqualify a capitalized pair from being a person name.
const NAME_STOPWORDS: &[&str] = &[
    "the", "this", "that", "these", "those", "there", "then", "than", "thus", "when", "where",
    "while", "with", "without", "within", "into", "onto", "over", "under", "after", "before",
    "during", "since", "about", "above", "below", "between", "because", "although", "however",
    "therefore", "furthermore", "moreover", "meanwhile", "chapter", "section", "figure", "table",
    "note", "summary", "abstract", "introduction", "conclusion", "results", "methodology",
    "background", "overview", "appendix", "references", "document", "file", "image", "code",
    "pub", "fn", "let", "var", "const", "new", "return", "import", "export", "class", "struct",
    "enum", "impl", "type", "def", "self", "none", "true", "false", "error", "warning", "todo",
];

/// Common multi-word capitals that are not names (e.g. "The Quick" in prose).
fn looks_like_person(full: &str) -> bool {
    let lower = full.to_lowercase();
    for word in lower.split_whitespace() {
        if NAME_STOPWORDS.contains(&word) {
            return false;
        }
    }
    // Avoid month names ("March 2024" style handled by date regex first).
    let months = [
        "january", "february", "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december",
    ];
    for m in months {
        if lower.starts_with(m) {
            return false;
        }
    }
    true
}

/// Extract all entity candidates from a text span (in textual order).
pub fn extract_entities(text: &str) -> Vec<EntityCandidate> {
    let mut out: Vec<EntityCandidate> = Vec::new();

    for m in re_email().find_iter(text) {
        out.push(EntityCandidate {
            surface: m.as_str().to_string(),
            entity_type: T_EMAIL.to_string(),
            start: m.start(),
            end: m.end(),
            confidence: 0.95,
        });
    }
    for m in re_url().find_iter(text) {
        out.push(EntityCandidate {
            surface: m.as_str().to_string(),
            entity_type: T_URL.to_string(),
            start: m.start(),
            end: m.end(),
            confidence: 0.95,
        });
    }
    for m in re_money().find_iter(text) {
        out.push(EntityCandidate {
            surface: m.as_str().to_string(),
            entity_type: T_MONEY.to_string(),
            start: m.start(),
            end: m.end(),
            confidence: 0.9,
        });
    }
    for m in re_percent().find_iter(text) {
        out.push(EntityCandidate {
            surface: m.as_str().to_string(),
            entity_type: T_PERCENT.to_string(),
            start: m.start(),
            end: m.end(),
            confidence: 0.9,
        });
    }
    for m in re_date().find_iter(text) {
        out.push(EntityCandidate {
            surface: m.as_str().to_string(),
            entity_type: T_DATE.to_string(),
            start: m.start(),
            end: m.end(),
            confidence: 0.85,
        });
    }
    for m in re_org().find_iter(text) {
        // The lazy head is still fed by word sequences like "Omega Corp
        // partnered with Tau Corp" — truncate at the FIRST suffix boundary
        // so the surface is the minimal legal name ("Omega Corp").
        let name = truncate_at_suffix_boundary(m.as_str().trim());
        let name_end = m.start() + name.len();
        if name.len() > 2 {
            out.push(EntityCandidate {
                surface: name,
                entity_type: T_ORG.to_string(),
                start: m.start(),
                end: name_end,
                confidence: 0.8,
            });
        }
    }
    for m in re_person().find_iter(text) {
        let full = m.as_str();
        if looks_like_person(full) {
            out.push(EntityCandidate {
                surface: full.to_string(),
                entity_type: T_PERSON.to_string(),
                start: m.start(),
                end: m.end(),
                confidence: 0.55, // heuristic: capitalized pairs are noisy
            });
        }
    }

    // De-duplicate overlaps: prefer higher confidence for identical spans.
    out.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    let mut filtered: Vec<EntityCandidate> = Vec::new();
    let mut last_end = 0usize;
    for c in out {
        if c.start >= last_end {
            last_end = c.end;
            filtered.push(c);
        }
    }
    filtered
}

/// Stopwords for keyword extraction (compact list, deterministic).
pub const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or",
    "our", "she", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
    "you", "your", "not", "no", "can", "do", "does", "did", "done", "also", "may", "might",
    "must", "shall", "should", "would", "could", "about", "after", "all", "any", "because",
    "been", "before", "between", "both", "during", "each", "more", "most", "other", "over",
    "such", "some", "only", "same", "such", "very", "use", "used", "using", "via", "while",
];

/// Extract keyword candidates with TF weights (IDF applied later by the engine).
pub fn extract_keywords(text: &str, max: usize) -> Vec<(String, f32)> {
    let words = crate::embeddings::tokenize(&text.to_lowercase());
    let mut tf: Vec<(String, f32)> = Vec::new();
    for w in &words {
        if w.len() < 3 || STOPWORDS.contains(&w.as_str()) || w.chars().all(|c| c.is_ascii_digit())
        {
            continue;
        }
        match tf.iter_mut().find(|(s, _)| s == w) {
            Some((_, c)) => *c += 1.0,
            None => tf.push((w.clone(), 1.0)),
        }
    }
    for (_w, c) in tf.iter_mut() {
        *c = 1.0 + (*c).ln();
    }
    tf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    tf.truncate(max);
    tf
}

/// Build a full KnowledgeObject for a chunk of text.
pub fn extract_knowledge(text: &str, df_lookup: &dyn Fn(&str) -> Option<i64>, total_docs: i64) -> KnowledgeObject {
    let entities = extract_entities(text);
    let mut keywords = extract_keywords(text, 12);
    // Apply IDF where available: score = tf * ln(1 + N / (df)).
    for (term, score) in keywords.iter_mut() {
        let df = df_lookup(term).unwrap_or(0);
        if df > 0 && total_docs > 1 {
            *score *= ((total_docs as f32) / (df as f32)).ln().max(0.15);
        }
    }
    keywords.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let dates = entities
        .iter()
        .filter(|e| e.entity_type == T_DATE)
        .map(|e| e.surface.clone())
        .collect();
    let claims = extract_claims(text);
    KnowledgeObject {
        entities,
        keywords,
        dates,
        claims,
    }
}

/// Serialize a KnowledgeObject for storage.
pub fn ko_to_json(ko: &KnowledgeObject) -> String {
    serde_json::to_string(ko).unwrap_or_else(|_| "{}".into())
}

/// Deserialize a stored KnowledgeObject.
pub fn ko_from_json(s: &str) -> Option<KnowledgeObject> {
    serde_json::from_str(s).ok()
}

/// Chunk-level content hash helper (re-exported for pipeline use).
pub fn chunk_content_hash(text: &str) -> String {
    hash_text(text)
}

// ---------------------------------------------------------------------------
// claim pattern extraction (deterministic SVO heuristics)
// ---------------------------------------------------------------------------

/// Split text into sentences (deterministic: [.!?] followed by whitespace+capital/EOF).
pub fn split_sentences(text: &str) -> Vec<(usize, String)> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    for &(pos, ch) in &chars {
        if ch == '.' || ch == '!' || ch == '?' {
            let rest: String = text[pos + ch.len_utf8()..].chars().take(2).collect();
            let ends = rest.trim_start();
            if rest.trim().is_empty()
                || ends
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_uppercase() || c.is_ascii_digit())
            {
                let s = text[start..pos + ch.len_utf8()].trim();
                if !s.is_empty() {
                    sentences.push((start, s.to_string()));
                }
                start = pos + ch.len_utf8();
            }
        }
    }
    if start < text.len() {
        let s = text[start..].trim();
        if !s.is_empty() {
            sentences.push((start, s.to_string()));
        }
    }
    sentences
}

/// Extract claim candidates from text using SVO patterns, with sentence
/// offsets (provenance) and negation detection (contradiction taxonomy).
pub fn extract_claims(text: &str) -> Vec<ExtractedClaim> {
    let mut out = Vec::new();
    for (off, sentence) in split_sentences(text) {
        if sentence.len() < 12 || sentence.len() > 400 {
            continue;
        }
        let negated = crate::claims::sentence_is_negative(&sentence);
        let end = off + sentence.len();
        for mut claim in match_sentence(&sentence) {
            claim.start_offset = off;
            claim.end_offset = end;
            claim.negated = negated;
            out.push(claim);
        }
    }
    out
}

static RE_NUM_CLAIM: OnceLock<regex::Regex> = OnceLock::new();
static RE_RELEASE: OnceLock<regex::Regex> = OnceLock::new();
static RE_VALID_FROM: OnceLock<regex::Regex> = OnceLock::new();
static RE_VALID_UNTIL: OnceLock<regex::Regex> = OnceLock::new();

fn match_sentence(sentence: &str) -> Vec<ExtractedClaim> {
    let mut out = Vec::new();
    let negated = crate::claims::sentence_is_negative(sentence);

    // Numeric metric claims: "<Subject> revenue/profit/users ... $10M / 45% ..."
    // NOTE: `[ \t]+` (not `\s+`) so a subject can never swallow a preceding
    // heading line ("## Revenue\n\nAcme Corp revenue ..." must yield
    // subject "Acme Corp", not "Revenue Acme Corp").
    let re_num = RE_NUM_CLAIM.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b([A-Z][\w&.\-]*(?:[ \t]+[A-Za-z][\w&.\-]*){0,3})[ \t]+(revenue|profit|users|sales|growth|market share|arr|mrr|funding)[ \t]+(?:of|was|is|were|reached|hit|=|grew to)[ \t]+([$€£]?[ \t]?\d[\d.,]*[ \t]?(?:million|billion|thousand|[MBKm%])?)",
        )
        .expect("numeric claim regex")
    });
    if let Some(m) = re_num.captures(sentence) {
        let subject = m.get(1).map_or("", |g| g.as_str()).trim().to_string();
        let predicate = m.get(2).map_or("", |g| g.as_str()).to_lowercase();
        let object = m.get(3).map_or("", |g| g.as_str()).trim().to_string();
        let (vf, vu) = temporal_bounds(sentence);
        out.push(ExtractedClaim {
            subject,
            predicate,
            object,
            sentence: sentence.to_string(),
            confidence: if negated { 0.45 } else { 0.6 },
            valid_from: vf,
            valid_until: vu,
            start_offset: 0,
            end_offset: 0,
            negated,
        });
        return out;
    }

    // Release/action claims: "<Org/Person> released|launched|acquired|announced <Object>"
    let re_rel = RE_RELEASE.get_or_init(|| {
        regex::Regex::new(
            r"\b([A-Z][\w&.\-]*(?:[ \t]+[A-Za-z][\w&.\-]*){0,3})[ \t]+(released|launched|acquired|announced|founded|joined|deployed|published|introduced)[ \t]+([A-Z0-9][^.!?]{2,120})",
        )
        .expect("release claim regex")
    });
    if let Some(m) = re_rel.captures(sentence) {
        let subject = m.get(1).map_or("", |g| g.as_str()).trim().to_string();
        let predicate = m.get(2).map_or("", |g| g.as_str()).to_lowercase();
        let object = m
            .get(3)
            .map_or("", |g| g.as_str())
            .trim()
            .trim_end_matches('.')
            .to_string();
        if !subject.is_empty() && !object.is_empty() {
            let (vf, vu) = temporal_bounds(sentence);
            out.push(ExtractedClaim {
                subject,
                predicate,
                object,
                sentence: sentence.to_string(),
                confidence: if negated { 0.35 } else { 0.5 },
                valid_from: vf,
                valid_until: vu,
                start_offset: 0,
                end_offset: 0,
                negated,
            });
            return out;
        }
    }

    // Copula claims: "<X> is/are/was/were <Y>"
    let re_cop = regex::Regex::new(
        r"\b([A-Z][\w&.\-]*(?:[ \t]+[A-Za-z][\w&.\-]*){0,2})[ \t]+(is|are|was|were)[ \t]+((?:a|an|the)?[ \t]?[A-Za-z0-9][^.!?]{2,120})",
    )
    .expect("copula regex");
    if let Some(m) = re_cop.captures(sentence) {
        let subject = m.get(1).map_or("", |g| g.as_str()).trim().to_string();
        let object = m
            .get(3)
            .map_or("", |g| g.as_str())
            .trim()
            .trim_end_matches('.')
            .to_string();
        if subject.len() > 2 && object.len() > 3 {
            let (vf, vu) = temporal_bounds(sentence);
            out.push(ExtractedClaim {
                subject,
                predicate: "is".into(),
                object,
                sentence: sentence.to_string(),
                confidence: if negated { 0.3 } else { 0.4 },
                valid_from: vf,
                valid_until: vu,
                start_offset: 0,
                end_offset: 0,
                negated,
            });
        }
    }
    out
}

fn temporal_bounds(sentence: &str) -> (Option<String>, Option<String>) {
    let re_from = RE_VALID_FROM.get_or_init(|| {
        regex::Regex::new(r"(?i)\b(?:since|from|as of|starting|in)\s+((?:19|20)\d{2}(?:-\d{2}-\d{2})?)").expect("vf regex")
    });
    let re_until = RE_VALID_UNTIL.get_or_init(|| {
        regex::Regex::new(r"(?i)\b(?:until|through|till|ending|by)\s+((?:19|20)\d{2}(?:-\d{2}-\d{2})?)").expect("vu regex")
    });
    let vf = re_from
        .captures(sentence)
        .and_then(|c| c.get(1))
        .map(|g| g.as_str().to_string());
    let vu = re_until
        .captures(sentence)
        .and_then(|c| c.get(1))
        .map(|g| g.as_str().to_string());
    (vf, vu)
}
