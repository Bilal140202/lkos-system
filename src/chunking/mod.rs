//! Structural chunking.
//!
//! Design (see `docs/INGESTION.md` and ADR-003):
//! 1. Split normalized text into blocks (markdown headings, blank-line
//!    paragraphs, code fences, code units for code documents).
//! 2. Aggregate blocks into chunks up to `max_chars`; a block larger than the
//!    budget is split on sentence boundaries.
//! 3. Detect and forward-propagate section titles (deterministic heuristics
//!    from the original LKOS architecture, kept and validated by tests).
//! 4. Assign positional authority: opening chunks carry orientation value,
//!    closing chunks carry conclusion value (validated in benchmarks).
//!
//! The chunker is a pure function: `(text, config) -> Vec<ProposedChunk>`.
//! No IO, no randomness — fully deterministic and unit-testable.

use crate::config::Config;
use crate::ingestion::hash_text;

/// A chunk proposed by the chunker, before embedding/storage.
#[derive(Debug, Clone)]
pub struct ProposedChunk {
    /// Chunk text.
    pub text: String,
    /// Section title in effect for this chunk.
    pub section_title: Option<String>,
    /// Semantic kind.
    pub kind: crate::types::ChunkKind,
    /// Start offset in the normalized text.
    pub start_offset: usize,
    /// End offset (exclusive).
    pub end_offset: usize,
    /// Authority multiplier.
    pub authority: f32,
}

/// Chunker version (recorded per document; bump to force re-chunk).
pub const CHUNKER_VERSION: &str = "chunk-v1.1.0";

/// Chunk a document body.
pub fn chunk_document(
    text: &str,
    doc_type: crate::types::DocType,
    cfg: &Config,
) -> Vec<ProposedChunk> {
    let blocks = match doc_type {
        crate::types::DocType::Code => code_blocks(text),
        _ => prose_blocks(text),
    };
    let mut chunks = aggregate(blocks, cfg);
    // Post-pass: merge tiny trailing chunks forward where possible.
    merge_tiny(&mut chunks, cfg);
    // Authority pass (positional, deterministic).
    assign_authority(&mut chunks);
    chunks
}

#[derive(Debug, Clone)]
struct Block {
    text: String,
    section: Option<String>,
    kind: crate::types::ChunkKind,
    start: usize,
    end: usize,
}

// ---------------------------------------------------------------------------
// prose
// ---------------------------------------------------------------------------

fn prose_blocks(text: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let bytes_len = text.len();
    let mut current_section: Option<String> = None;
    let mut para_start: Option<usize> = None;
    let mut i = 0usize;

    let lines: Vec<&str> = text.split('\n').collect();
    let mut consumed = 0usize; // char offsets handled via byte cursor

    for line in &lines {
        let line_start = i;
        i += line.len() + 1; // +1 for '\n' (safe: we split on it)
        let trimmed = line.trim();

        // Markdown heading?
        if trimmed.starts_with('#') && trimmed.len() > 1 {
            let level = trimmed.chars().take_while(|&c| c == '#').count();
            let title = trimmed.trim_start_matches('#').trim().to_string();
            if !title.is_empty() && level <= 4 {
                if let Some(s) = para_start.take() {
                    push_prose_block(&mut blocks, text, s, line_start, current_section.clone());
                }
                let mut b = Block {
                    text: format!("## {title}"),
                    section: Some(title.clone()),
                    kind: crate::types::ChunkKind::Heading,
                    start: line_start,
                    end: i,
                };
                b.section = Some(title.clone());
                blocks.push(b);
                current_section = Some(title);
                consumed = i;
                continue;
            }
        }

        // ALL-CAPS or trailing-colon standalone line = section heading heuristic.
        if is_heading_like(trimmed) {
            if let Some(s) = para_start.take() {
                push_prose_block(&mut blocks, text, s, line_start, current_section.clone());
            }
            let title = trimmed.trim_end_matches(':').to_string();
            blocks.push(Block {
                text: format!("## {title}"),
                section: Some(title.clone()),
                kind: crate::types::ChunkKind::Heading,
                start: line_start,
                end: i,
            });
            current_section = Some(title);
            consumed = i;
            continue;
        }

        if trimmed.is_empty() {
            if let Some(s) = para_start.take() {
                push_prose_block(&mut blocks, text, s, line_start, current_section.clone());
                consumed = line_start;
            }
        } else if para_start.is_none() {
            para_start = Some(line_start);
        }
    }
    if let Some(s) = para_start.take() {
        push_prose_block(&mut blocks, text, s, bytes_len, current_section.clone());
    }
    let _ = consumed;
    blocks
}

fn push_prose_block(
    blocks: &mut Vec<Block>,
    text: &str,
    start: usize,
    end: usize,
    section: Option<String>,
) {
    let slice = &text[start..end.min(text.len())];
    let t = slice.trim();
    if t.is_empty() {
        return;
    }
    blocks.push(Block {
        text: t.to_string(),
        section,
        kind: crate::types::ChunkKind::Prose,
        start,
        end: start + t.len(),
    });
}

/// Heading heuristics (deterministic, from the original LKOS design):
/// 1. Standalone line in ALL CAPS containing letters, <= 60 chars.
/// 2. Standalone line ending with ':' with <= 60 chars, <= 6 words.
fn is_heading_like(line: &str) -> bool {
    if line.is_empty() || line.len() > 60 || line.contains(". ") || line.starts_with('#') {
        return false;
    }
    let has_letters = line.chars().any(|c| c.is_alphabetic());
    let all_caps = line
        .chars()
        .filter(|c| c.is_alphabetic())
        .all(|c| c.is_uppercase());
    if has_letters && all_caps && line.split_whitespace().count() <= 8 {
        return true;
    }
    let ends_colon = line.ends_with(':');
    let words = line.split_whitespace().count();
    ends_colon && words <= 6 && words > 0 && line.chars().next().is_some_and(|c| c.is_alphabetic())
}

// ---------------------------------------------------------------------------
// code
// ---------------------------------------------------------------------------

/// Top-level unit openers per language family (best-effort, deterministic).
fn code_blocks(text: &str) -> Vec<Block> {
    // Split into top-level blocks: a new block starts at a line matching unit
    // openers at column 0. Comments/blank lines attach to the following unit.
    let openers = [
        "fn ",
        "pub fn ",
        "pub(crate) fn ",
        "impl ",
        "struct ",
        "enum ",
        "trait ",
        "mod ",
        "class ",
        "def ",
        "async def ",
        "function ",
        "export function ",
        "export class ",
        "public class ",
        "private class ",
        "interface ",
        "type ",
        "package ",
        "import ",
        "#include",
        "func ",
        "var ",
        "const ",
        "let ",
        "SELECT",
        "CREATE",
        "INSERT",
        "UPDATE",
    ];
    let mut blocks = Vec::new();
    let mut block_start: Option<usize> = None;
    let mut i = 0usize;
    for line in text.split('\n') {
        let line_start = i;
        i += line.len() + 1;
        let trimmed = line.trim_end();
        let is_open = openers.iter().any(|o| trimmed.starts_with(o));
        if is_open {
            if let Some(s) = block_start.take() {
                push_code_block(&mut blocks, text, s, line_start);
            }
            block_start = Some(line_start);
        }
    }
    let end = text.len();
    if let Some(s) = block_start.take() {
        push_code_block(&mut blocks, text, s, end);
    } else if blocks.is_empty() {
        push_code_block(&mut blocks, text, 0, end);
    }
    blocks
}

fn push_code_block(blocks: &mut Vec<Block>, text: &str, start: usize, end: usize) {
    let slice = &text[start..end.min(text.len())];
    let t = slice.trim();
    if t.is_empty() {
        return;
    }
    let kind = if is_code_unit(t) {
        crate::types::ChunkKind::CodeUnit
    } else {
        crate::types::ChunkKind::Code
    };
    blocks.push(Block {
        text: t.to_string(),
        section: None,
        kind,
        start,
        end: start + t.len(),
    });
}

/// Heuristic: does this block look like a named unit (fn/class/def) we can title?
fn is_code_unit(block: &str) -> bool {
    let first = block.lines().next().unwrap_or("");
    [
        "fn ",
        "pub fn ",
        "def ",
        "class ",
        "function ",
        "func ",
        "impl ",
        "struct ",
        "trait ",
        "enum ",
    ]
    .iter()
    .any(|p| first.contains(p))
}

// ---------------------------------------------------------------------------
// aggregation
// ---------------------------------------------------------------------------

fn aggregate(blocks: Vec<Block>, cfg: &Config) -> Vec<ProposedChunk> {
    let mut chunks: Vec<ProposedChunk> = Vec::new();
    let mut cur_text = String::new();
    let mut cur_section: Option<String> = None;
    let mut cur_kind = crate::types::ChunkKind::Prose;
    let mut cur_start = 0usize;
    let mut cur_end = 0usize;
    let mut open = false;

    for b in blocks {
        let is_heading = b.kind == crate::types::ChunkKind::Heading;
        let fits = cur_text.len() + b.text.len() + 2 <= cfg.chunk_max_chars;
        let same_kind = b.kind == cur_kind || !open;

        if open && (!fits || is_heading || !same_kind) {
            chunks.push(ProposedChunk {
                text: cur_text.clone(),
                section_title: cur_section.clone(),
                kind: cur_kind,
                start_offset: cur_start,
                end_offset: cur_end,
                authority: 1.0,
            });
            cur_text.clear();
            open = false;
        }

        if is_heading {
            // Headings become part of the section context, not standalone chunks
            // unless nothing follows: keep them as context markers.
            cur_section = b.section.clone();
            cur_start = b.start;
            cur_end = b.end;
            cur_text.push_str(&b.text);
            cur_text.push_str("\n\n");
            open = true;
            // Heading-only chunk: mark as heading kind, the next body block will
            // attach to it (same-kind check disabled by resetting kind below).
            cur_kind = crate::types::ChunkKind::Prose;
            continue;
        }

        if !open {
            cur_start = b.start;
            cur_section = b.section.clone();
            cur_kind = b.kind;
        }
        cur_end = b.end;
        cur_text.push_str(&b.text);
        cur_text.push('\n');
        open = true;
    }
    if open && !cur_text.trim().is_empty() {
        chunks.push(ProposedChunk {
            text: cur_text.trim().to_string(),
            section_title: cur_section,
            kind: cur_kind,
            start_offset: cur_start,
            end_offset: cur_end,
            authority: 1.0,
        });
    }
    // Oversized single blocks were not split here (prose_blocks emits paragraph
    // blocks which are naturally bounded); enforce a hard cap by sentence split.
    chunks = split_oversized(chunks, cfg);
    chunks
}

fn split_oversized(chunks: Vec<ProposedChunk>, cfg: &Config) -> Vec<ProposedChunk> {
    let mut out = Vec::with_capacity(chunks.len());
    for c in chunks {
        if c.text.len() <= cfg.chunk_max_chars * 2 {
            out.push(c);
            continue;
        }
        // Sentence-boundary split.
        let start = 0usize;
        let mut last_break = 0usize;
        let text = c.text.as_str();
        let mut split_at: Vec<usize> = Vec::new();
        for (i, ch) in text.char_indices() {
            if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
                let next_is_space = text[i + ch.len_utf8()..]
                    .chars()
                    .next()
                    .map_or(true, |n| n.is_whitespace());
                if next_is_space && i - last_break >= cfg.chunk_max_chars {
                    split_at.push(i + ch.len_utf8());
                    last_break = i;
                }
            }
        }
        let mut prev = 0usize;
        for s in split_at {
            let piece = text[prev..s].trim();
            if !piece.is_empty() {
                out.push(ProposedChunk {
                    text: piece.to_string(),
                    section_title: c.section_title.clone(),
                    kind: c.kind,
                    start_offset: c.start_offset + prev,
                    end_offset: c.start_offset + s,
                    authority: 1.0,
                });
            }
            prev = s;
        }
        if prev < start + text.len() {
            let piece = text[prev..].trim();
            if !piece.is_empty() {
                out.push(ProposedChunk {
                    text: piece.to_string(),
                    section_title: c.section_title.clone(),
                    kind: c.kind,
                    start_offset: c.start_offset + prev,
                    end_offset: c.start_offset + text.len(),
                    authority: 1.0,
                });
            }
        }
    }
    out
}

fn merge_tiny(chunks: &mut Vec<ProposedChunk>, cfg: &Config) {
    if chunks.len() < 2 {
        return;
    }
    let mut i = 0usize;
    while i + 1 < chunks.len() {
        if chunks[i].text.len() < cfg.chunk_min_chars
            && chunks[i].kind != crate::types::ChunkKind::CodeUnit
        {
            let merged_text = format!("{}\n{}", chunks[i].text, chunks[i + 1].text);
            chunks[i + 1].text = merged_text;
            chunks[i + 1].start_offset = chunks[i].start_offset;
            if chunks[i + 1].section_title.is_none() {
                chunks[i + 1].section_title = chunks[i].section_title.clone();
            }
            chunks.remove(i);
        } else {
            i += 1;
        }
    }
}

fn assign_authority(chunks: &mut [ProposedChunk]) {
    let n = chunks.len();
    let tail_start = if n > 10 { n - (n / 10).max(1) } else { n };
    for (i, c) in chunks.iter_mut().enumerate() {
        c.authority = if i < 3 {
            1.2
        } else if i >= tail_start || c.kind == crate::types::ChunkKind::CodeUnit {
            1.1
        } else {
            1.0
        };
    }
}

/// Convenience: content hash of a chunk's text.
pub fn chunk_hash(text: &str) -> String {
    hash_text(text)
}
