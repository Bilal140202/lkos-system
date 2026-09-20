//! Temporal knowledge utilities (v0.2: structural date reasoning).
//!
//! v0.1 matched 4-digit year *strings* in chunk text. This version parses a
//! normalized date set per chunk — ISO dates (2024-06-01), month-name dates
//! ("June 2024", "Jun 15, 2024"), quarters ("Q3 2024"), and bare years —
//! and answers temporal constraints over that set (bitemporal validity on
//! claims remains on the roadmap; see docs/TEMPORAL_KNOWLEDGE.md).
//!
//! Query-side changes vs v0.1:
//! - `Latest` boosts recent chunks instead of hard-filtering (a query for
//!   the "latest" revenue should still return last year's report if it is
//!   the newest evidence).
//! - `Year`/`Range` still filter, but fall back to re-rank when the filter
//!   would empty the result set.

use chrono::Datelike;
use std::sync::OnceLock;

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
            // Prefer chunks whose freshest parsed date is recent.
            let this_year = current_year();
            let freshest = extract_dates(text)
                .map(|d| d.year as i32)
                .max()
                .unwrap_or(0);
            let age = this_year - freshest;
            if freshest == 0 {
                0.0
            } else if age <= 2 {
                1.0
            } else if age <= 6 {
                0.6
            } else {
                0.3
            }
        }
        TemporalConstraint::Year(y) => {
            if extract_dates(text).any(|d| d.year as i32 == *y) {
                1.0
            } else {
                0.0
            }
        }
        TemporalConstraint::Range(lo, hi) => {
            let hit = extract_dates(text).any(|d| {
                let y = d.year as i32;
                *lo <= y && y <= *hi
            });
            if hit {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// A normalized date extracted from text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalDate {
    /// Year (1-9999).
    pub year: u16,
    /// Month 1-12 (1 when unknown).
    pub month: u8,
    /// Day 1-31 (1 when unknown).
    pub day: u8,
    /// True when parsed from an ISO `YYYY-MM-DD` (highest precision).
    pub precise: bool,
}

static RE_DATE: OnceLock<regex::Regex> = OnceLock::new();

/// Extract normalized dates from text: ISO dates, "June 2024"/"Jun 15, 2024",
/// "Q3 2024", and bare years. Deduplicated, in order of appearance.
pub fn extract_dates(text: &str) -> impl Iterator<Item = NormalDate> {
    let re = RE_DATE.get_or_init(|| {
        regex::Regex::new(
            r"(?x)
            (?P<iso>(?:19|20)\d{2}-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12]\d|3[01]))?)
          | (?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s(?:[0-3]?\d,\s)?(?P<mony>(?:19|20)\d{2})
          | Q(?P<q>[1-4])\s(?P<qy>(?:19|20)\d{2})
          | \b(?P<year>(?:19|20)\d{2})\b
            ",
        )
        .expect("date regex")
    });
    let mut out: Vec<NormalDate> = Vec::new();
    for cap in re.captures_iter(text) {
        let date = if let Some(iso) = cap.name("iso") {
            let parts: Vec<u16> = iso
                .as_str()
                .split('-')
                .filter_map(|p| p.parse().ok())
                .collect();
            NormalDate {
                year: *parts.first().unwrap_or(&0),
                month: *parts.get(1).unwrap_or(&1) as u8,
                day: *parts.get(2).unwrap_or(&1) as u8,
                precise: parts.len() >= 3,
            }
        } else if let Some(mony) = cap.name("mony") {
            let mon = cap
                .name("mon")
                .and_then(|m| month_from_name(m.as_str()))
                .unwrap_or(1);
            NormalDate {
                year: mony.as_str().parse().unwrap_or(0),
                month: mon,
                day: 1,
                precise: false,
            }
        } else if let Some(qy) = cap.name("qy") {
            let q: u8 = cap
                .name("q")
                .and_then(|q| q.as_str().parse().ok())
                .unwrap_or(1);
            NormalDate {
                year: qy.as_str().parse().unwrap_or(0),
                month: q * 3 - 2,
                day: 1,
                precise: false,
            }
        } else if let Some(y) = cap.name("year") {
            NormalDate {
                year: y.as_str().parse().unwrap_or(0),
                month: 1,
                day: 1,
                precise: false,
            }
        } else {
            continue;
        };
        if date.year >= 1900 && date.year <= 2100 && !out.contains(&date) {
            out.push(date);
        }
    }
    out.into_iter()
}

fn month_from_name(name: &str) -> Option<u8> {
    let n = name.trim_end_matches('.').to_lowercase();
    let n = if n == "sept" { "sep".to_string() } else { n };
    Some(match n.as_str() {
        "jan" => 1,
        "feb" => 2,
        "mar" => 3,
        "apr" => 4,
        "may" => 5,
        "jun" => 6,
        "jul" => 7,
        "aug" => 8,
        "sep" => 9,
        "oct" => 10,
        "nov" => 11,
        "dec" => 12,
        _ => return None,
    })
}

fn current_year() -> i32 {
    chrono::Utc::now().year()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_iso_month_quarter_and_year() {
        let dates: Vec<NormalDate> =
            extract_dates("Q3 2024 report dated 2024-06-01, filed June 2024, in 2024").collect();
        assert!(dates.iter().any(|d| d.year == 2024 && d.month == 7)); // Q3 start
        assert!(dates
            .iter()
            .any(|d| d.year == 2024 && d.month == 6 && d.day == 1 && d.precise));
        assert!(dates.iter().any(|d| d.year == 2024 && d.month == 6)); // "June 2024"
    }

    #[test]
    fn quarters_map_correctly() {
        let dates: Vec<NormalDate> = extract_dates("Q1 2023 revenue").collect();
        assert_eq!(dates.first().map(|d| d.month), Some(1));
        let dates: Vec<NormalDate> = extract_dates("Q4 2023 revenue").collect();
        assert_eq!(dates.first().map(|d| d.month), Some(10));
    }

    #[test]
    fn years_are_deduplicated() {
        let n = extract_dates("2023 and 2023 and 2023-05-01").count();
        assert_eq!(n, 2); // bare 2023 + ISO date
    }
}
