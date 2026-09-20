//! Document ingestion: byte-level extraction and normalization.
//!
//! v0.1 extractors (deterministic, dependency-free):
//! - plain text (`.txt`)
//! - markdown (`.md`, `.markdown`) — structural cues preserved for the chunker
//! - source code (`.rs`, `.py`, `.js`, `.ts`, `.tsx`, `.java`, `.c`, `.cpp`, `.h`,
//!   `.go`, `.json`, `.toml`, `.yaml`, `.yml`, `.sh`, ...)
//! - data (`.csv`, `.xml`, `.html` — HTML tags stripped)
//!
//! Unknown extensions are sniffed: if content is valid UTF-8 with < 10% control
//! characters it is ingested as text; otherwise rejected (binary formats like
//! PDF/DOCX are *deliberately deferred* — see `docs/NON_GOALS.md` and
//! `docs/MULTIMODAL.md`).
//!
//! Normalization (`normalize`) is lossless w.r.t. words: CRLF→LF, strip NUL and
//! other C0 control characters except newline/tab, collapse 3+ blank lines.

use crate::error::{LkosError, Result};
use sha2::{Digest, Sha256};

/// The normalized output of extraction.
#[derive(Debug, Clone)]
pub struct ExtractedDocument {
    /// Normalized plain text (the canonical indexing substrate).
    pub text: String,
    /// Coarse document type.
    pub doc_type: crate::types::DocType,
    /// Detected programming language (code docs only).
    pub language: Option<&'static str>,
    /// Original file name.
    pub filename: String,
    /// SHA-256 hex of the ORIGINAL bytes.
    pub content_hash: String,
    /// Raw byte size.
    pub size: u64,
    /// Original path when extracted from disk.
    pub path_for_storage: Option<String>,
}

/// Extractor implementation version (bump when behavior changes to trigger reindex).
pub const EXTRACTOR_VERSION: &str = "ingest-v1.0.0";

/// Extensions recognized as markdown.
pub const MARKDOWN_EXT: &[&str] = &["md", "markdown", "mdown", "mkd"];
/// Extensions recognized as source code.
pub const CODE_EXT: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "java", "kt", "go", "c", "h", "cpp", "hpp", "cc", "cs",
    "rb", "php", "swift", "m", "sh", "bash", "zsh", "sql", "toml", "yaml", "yml", "ini", "cfg",
];
/// Extensions recognized as data files.
pub const DATA_EXT: &[&str] = &["json", "csv", "tsv", "xml", "html", "htm"];

/// Extract and normalize a document from raw bytes.
pub fn extract(filename: &str, bytes: &[u8]) -> Result<ExtractedDocument> {
    if bytes.is_empty() {
        return Err(LkosError::InvalidInput("empty file".into()));
    }
    let content_hash = content_hash(bytes);
    let size = bytes.len() as u64;
    let ext = filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();

    let (text, doc_type, language) = if MARKDOWN_EXT.contains(&ext.as_str()) {
        let raw = decode_utf8(bytes)?;
        (raw, crate::types::DocType::Markdown, None)
    } else if CODE_EXT.contains(&ext.as_str()) {
        let raw = decode_utf8(bytes)?;
        (raw, crate::types::DocType::Code, code_language(&ext))
    } else if DATA_EXT.contains(&ext.as_str()) {
        let raw = decode_utf8(bytes)?;
        let cleaned = if ext == "html" || ext == "htm" || ext == "xml" {
            strip_markup(&raw)
        } else {
            raw
        };
        (cleaned, crate::types::DocType::Data, None)
    } else {
        // Sniff: valid UTF-8 and mostly printable → text.
        match std::str::from_utf8(bytes) {
            Ok(s) if control_ratio(s) < 0.1 => (s.to_string(), crate::types::DocType::Text, None),
            _ => {
                return Err(LkosError::UnsupportedFileType(format!(
                    "'{filename}': binary or unsupported format (PDF/DOCX are on the roadmap; \
                     see docs/NON_GOALS.md)"
                )))
            }
        }
    };

    let normalized = normalize(&text);
    if normalized.trim().is_empty() {
        return Err(LkosError::InvalidInput(format!(
            "'{filename}' contains no extractable text"
        )));
    }
    Ok(ExtractedDocument {
        text: normalized,
        doc_type,
        language,
        filename: filename.to_string(),
        content_hash,
        size,
        path_for_storage: None,
    })
}

/// Extract from a filesystem path.
pub fn extract_file(path: impl AsRef<std::path::Path>) -> Result<ExtractedDocument> {
    let path = path.as_ref();
    let filename = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .ok_or_else(|| LkosError::InvalidInput("path has no file name".into()))?;
    let bytes = std::fs::read(path)?;
    let mut doc = extract(&filename, &bytes)?;
    doc.path_for_storage = Some(path.to_string_lossy().to_string());
    Ok(doc)
}

/// SHA-256 hex digest of raw bytes.
pub fn content_hash(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 hex of a string (chunk hashing).
pub fn hash_text(s: &str) -> String {
    content_hash(s.as_bytes())
}

fn decode_utf8(bytes: &[u8]) -> Result<String> {
    String::from_utf8(bytes.to_vec())
        .map_err(|_| LkosError::InvalidInput("invalid UTF-8".into()))
}

fn control_ratio(s: &str) -> f64 {
    if s.is_empty() {
        return 1.0;
    }
    let ctrl = s
        .chars()
        .filter(|&c| c.is_control() && c != '\n' && c != '\t' && c != '\r')
        .count();
    ctrl as f64 / s.chars().count() as f64
}

fn code_language(ext: &str) -> Option<&'static str> {
    Some(match ext {
        "rs" => "rust",
        "py" => "python",
        "js" | "jsx" => "javascript",
        "ts" | "tsx" => "typescript",
        "java" => "java",
        "kt" => "kotlin",
        "go" => "go",
        "c" | "h" => "c",
        "cpp" | "hpp" | "cc" => "cpp",
        "cs" => "csharp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "m" => "objc",
        "sh" | "bash" | "zsh" => "shell",
        "sql" => "sql",
        "toml" | "ini" | "cfg" => "toml",
        "yaml" | "yml" => "yaml",
        _ => return None,
    })
}

/// Very small HTML/XML tag stripper (documented limitation: no entity decoding).
fn strip_markup(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

/// Normalize text: CRLF→LF, drop control chars (keep \n, \t), collapse blank runs.
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank_run = 0usize;
    for line in s.replace("\r\n", "\n").replace('\r', "\n").split('\n') {
        let trimmed_end = line.trim_end();
        if trimmed_end.is_empty() {
            blank_run += 1;
            if blank_run > 2 {
                continue;
            }
            out.push('\n');
        } else {
            blank_run = 0;
            for ch in trimmed_end.chars() {
                if ch == '\u{0}' || (ch.is_control() && ch != '\t') {
                    continue;
                }
                out.push(ch);
            }
            out.push('\n');
        }
    }
    out
}
