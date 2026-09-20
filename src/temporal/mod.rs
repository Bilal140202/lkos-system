//! Temporal knowledge utilities.
//!
//! Extracted dates/validity bounds live on claims (see `knowledge`). This
//! module provides query-side temporal reasoning: detect years/ranges in a
//! query and rank chunks by temporal relevance.

use chrono::Datelike;

/// A temporal constraint parsed from a query.
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalConstraint {
    /// as-of year (results about or before this year preferred).
    Year(i32),
    /// A year range (inclusive).
    Range(i32, i32),
    /// "latest/most recent" intent.
    Latest,
    /// No temporal constraint detected.
    None,
}

/// Parse temporal constraints from a query string (deterministic).
pub fn detect_temporal(query: &str) -> TemporalConstraint {
    let lower = query.to_lowercase();
    if lower.contains("latest") || lower.contains("most recent") || lower.contains("current") {
        return TemporalConstraint::Latest;
    }
    let words: Vec<&str> = lower.split_whitespace().collect();
    let mut years: Vec<i32> = Vec::new();
    for w in &words {
        let cleaned = w.trim_matches(|c: char| !c.is_alphanumeric());
        if let Ok(y) = cleaned.parse::<i32>() {
            if (1900..=2100).contains(&y) {
                years.push(y);
            }
        }
    }
    match years.len() {
        0 => TemporalConstraint::None,
        1 => TemporalConstraint::Year(years[0]),
        _ => {
            let lo = *years.iter().min().expect("non-empty");
            let hi = *years.iter().max().expect("non-empty");
            TemporalConstraint::Range(lo, hi)
        }
    }
}

/// Does a chunk text satisfy the temporal constraint?
pub fn matches_constraint(text: &str, c: &TemporalConstraint) -> bool {
    match c {
        TemporalConstraint::None => true,
        _ => temporal_score(text, c) > 0.0,
    }
}

/// Temporal relevance score in [0, 1] for a chunk against a constraint.
pub fn temporal_score(text: &str, c: &TemporalConstraint) -> f32 {
    match c {
        TemporalConstraint::None => 0.5,
        TemporalConstraint::Latest => {
            // Prefer chunks mentioning recent years.
            let this_year = current_year();
            let mut best = 0.0f32;
            for y in extract_years(text) {
                if y >= this_year - 2 {
                    best = best.max(1.0);
                } else if y >= this_year - 6 {
                    best = best.max(0.6);
                }
            }
            best
        }
        TemporalConstraint::Year(y) => {
            if extract_years(text).contains(y) {
                1.0
            } else {
                0.0
            }
        }
        TemporalConstraint::Range(lo, hi) => {
            let hit = extract_years(text).iter().any(|y| *y >= *lo && *y <= *hi);
            if hit {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn extract_years(text: &str) -> Vec<i32> {
    let mut years = Vec::new();
    let mut buf = String::new();
    for ch in text.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() {
            buf.push(ch);
        } else {
            if buf.len() == 4 {
                if let Ok(y) = buf.parse::<i32>() {
                    if (1900..=2100).contains(&y) {
                        years.push(y);
                    }
                }
            }
            buf.clear();
        }
    }
    years
}

fn current_year() -> i32 {
    chrono::Utc::now().year()
}
