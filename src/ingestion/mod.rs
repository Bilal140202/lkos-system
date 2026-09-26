//! Document ingestion: byte-level extraction and normalization.
//!
//! v0.9 extractors (deterministic, bounded):
//! - plain text (`.txt`), markdown (`.md`), source code (20+ languages)
//! - data (`.csv`, `.tsv`, `.json`)
//! - HTML/XML: script/style removal + entity decoding (`&amp;` …)
//! - **DOCX**: OOXML `w:t` runs, paragraphs → lines, with size caps
//! - **XLSX**: shared strings + inline cells, sheet-scoped lines
//! - **PPTX**: slide text runs, one slide per line group
//! - **EPUB**: XHTML spine entries stripped to text
//! - **PDF**: behind the optional `pdf` feature (pure-Rust `pdf-extract`)
//!
//! Security hardening (see docs/SECURITY.md):
//! - Archive-based formats are rejected beyond [`MAX_INPUT_BYTES`] raw and
//!   [`MAX_DECOMPRESSED_BYTES`] inflated (decompression-bomb guard).
//! - Entry counts and per-entry size caps stop nested-zip and header-lying
//!   attacks; OOXML/HTML/XML are handled by hand-rolled string scanning (no XML
//!   dependency, no entity expansion, no DTD processing).
//!
//! Normalization (`normalize`) is lossless w.r.t. words: CRLF→LF, strip NUL and
//! other C0 control characters except newline/tab, collapse 3+ blank lines.

use crate::error::{LkosError, Result};
use sha2::{Digest, Sha256};
use std::io::Read;

/// Hard cap on raw input bytes (64 MiB).
pub const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;
/// Hard cap on decompressed archive content (256 MiB) — decompression-bomb guard.
pub const MAX_DECOMPRESSED_BYTES: usize = 256 * 1024 * 1024;
/// Maximum entries accepted inside one archive (nested-zip guard).
const MAX_ARCHIVE_ENTRIES: usize = 4096;

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
pub const EXTRACTOR_VERSION: &str = "ingest-v2.0.0";

/// Extensions recognized as markdown.
pub const MARKDOWN_EXT: &[&str] = &["md", "markdown", "mdown", "mkd"];
/// Extensions recognized as source code.
pub const CODE_EXT: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "java", "kt", "go", "c", "h", "cpp", "hpp", "cc", "cs",
    "rb", "php", "swift", "m", "sh", "bash", "zsh", "sql", "toml", "yaml", "yml", "ini", "cfg",
];
/// Extensions recognized as data files.
pub const DATA_EXT: &[&str] = &["json", "csv", "tsv", "xml", "html", "htm"];
/// OOXML / zip-container extensions handled by the archive extractors.
pub const ARCHIVE_EXT: &[&str] = &["docx", "xlsx", "pptx", "epub"];

/// Extract and normalize a document from raw bytes.
pub fn extract(filename: &str, bytes: &[u8]) -> Result<ExtractedDocument> {
    if bytes.is_empty() {
        return Err(LkosError::InvalidInput("empty file".into()));
    }
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(LkosError::InvalidInput(format!(
            "'{filename}' is {} bytes (cap {MAX_INPUT_BYTES})",
            bytes.len()
        )));
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
    } else if ext == "html" || ext == "htm" || ext == "xml" {
        let raw = decode_utf8(bytes)?;
        (strip_markup(&raw), crate::types::DocType::Data, None)
    } else if ext == "docx" {
        (
            extract_docx(bytes, filename)?,
            crate::types::DocType::Text,
            None,
        )
    } else if ext == "xlsx" {
        (
            extract_xlsx(bytes, filename)?,
            crate::types::DocType::Data,
            None,
        )
    } else if ext == "pptx" {
        (
            extract_pptx(bytes, filename)?,
            crate::types::DocType::Text,
            None,
        )
    } else if ext == "epub" {
        (
            extract_epub(bytes, filename)?,
            crate::types::DocType::Text,
            None,
        )
    } else if DATA_EXT.contains(&ext.as_str()) {
        let raw = decode_utf8(bytes)?;
        (raw, crate::types::DocType::Data, None)
    } else if ext == "pdf" {
        return extract_pdf(filename, bytes);
    } else {
        // Sniff: valid UTF-8 and mostly printable → text.
        match std::str::from_utf8(bytes) {
            Ok(s) if control_ratio(s) < 0.1 => (s.to_string(), crate::types::DocType::Text, None),
            _ => {
                return Err(LkosError::UnsupportedFileType(format!(
                    "'{filename}': binary or unsupported format (supported: text, md, code, \
                     html, csv/json/tsv/xml, docx, xlsx, pptx, epub"
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
    String::from_utf8(bytes.to_vec()).map_err(|_| LkosError::InvalidInput("invalid UTF-8".into()))
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

/// Structured HTML/XML → text: drops `script`/`style` bodies entirely,
/// block-level tags become line breaks, and named/numeric entities are
/// decoded (`&amp;` → `&`). v0.1 toggled on `<`/`>`, which kept script
/// bodies and mangled attributes.
fn strip_markup(s: &str) -> String {
    let lower = s.to_ascii_lowercase();
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut dropped: Option<String> = None;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if let Some(end_rel) = lower[i..].find('>') {
                let tag = &lower[i + 1..i + end_rel];
                let name: String = tag
                    .trim_start_matches('/')
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric())
                    .collect();
                // Skip the element bodies of non-content containers.
                if dropped.is_none()
                    && matches!(name.as_str(), "script" | "style" | "head" | "noscript")
                    && !tag.starts_with('/')
                {
                    let close = format!("</{name}");
                    if let Some(close_rel) = lower[i..].find(&close) {
                        i += close_rel + close.len();
                        continue;
                    } else {
                        dropped = Some(name);
                        i += end_rel + 1;
                        continue;
                    }
                } else if let Some(d) = &dropped {
                    if tag.starts_with('/') && tag.trim_start_matches('/') == d {
                        dropped = None;
                    }
                    i += end_rel + 1;
                    continue;
                }
                if matches!(
                    name.as_str(),
                    "p" | "br"
                        | "div"
                        | "li"
                        | "tr"
                        | "h1"
                        | "h2"
                        | "h3"
                        | "h4"
                        | "h5"
                        | "h6"
                        | "table"
                        | "section"
                        | "article"
                        | "header"
                        | "footer"
                        | "blockquote"
                ) && !tag.starts_with('/')
                {
                    out.push('\n');
                }
                i += end_rel + 1;
                continue;
            } else {
                break; // unterminated tag: drop the tail
            }
        }
        if dropped.is_none() {
            let ch = s[i..].chars().next().unwrap_or(' ');
            if ch == '&' {
                // Entity decode (bounded look-ahead).
                let rest = &s[i + 1..i + 12.min(s.len() - i)];
                if let Some(semi) = rest.find(';') {
                    let ent = &rest[..semi];
                    let decoded = match ent {
                        "amp" => Some('&'),
                        "lt" => Some('<'),
                        "gt" => Some('>'),
                        "quot" => Some('"'),
                        "apos" => Some('\''),
                        "nbsp" => Some(' '),
                        _ => {
                            if let Some(hex) =
                                ent.strip_prefix("#x").or_else(|| ent.strip_prefix("#X"))
                            {
                                u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
                            } else if let Some(dec) = ent.strip_prefix('#') {
                                dec.parse::<u32>().ok().and_then(char::from_u32)
                            } else {
                                None
                            }
                        }
                    };
                    if let Some(d) = decoded {
                        out.push(d);
                        i += semi + 2;
                        continue;
                    }
                }
            }
            out.push(ch);
            i += ch.len_utf8();
        } else {
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// OOXML / zip-container extractors (DOCX, XLSX, PPTX, EPUB)
// ---------------------------------------------------------------------------

fn open_zip<'a>(
    bytes: &'a [u8],
    filename: &str,
) -> Result<zip::ZipArchive<std::io::Cursor<&'a [u8]>>> {
    zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| {
        LkosError::InvalidInput(format!(
            "'{filename}' is not a valid OOXML/zip container: {e}"
        ))
    })
}

fn read_zip_entry(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    name: &str,
) -> Result<String> {
    let file = archive.by_name(name).map_err(|_| {
        LkosError::InvalidInput(format!("archive entry '{name}' missing or unreadable"))
    })?;
    if file.size() as usize > MAX_DECOMPRESSED_BYTES {
        return Err(LkosError::InvalidInput(
            "archive entry exceeds decompression limit (possible decompression bomb)".into(),
        ));
    }
    let mut buf = String::new();
    let mut take = file.take(MAX_DECOMPRESSED_BYTES as u64);
    std::io::Read::read_to_string(&mut take, &mut buf).map_err(|e| {
        LkosError::InvalidInput(format!("archive entry '{name}' is not decodable text: {e}"))
    })?;
    Ok(buf)
}

/// DOCX: concatenate `<w:t>` runs from `word/document.xml`, one line per paragraph.
fn extract_docx(bytes: &[u8], filename: &str) -> Result<String> {
    let mut archive = open_zip(bytes, filename)?;
    if archive.len() > MAX_ARCHIVE_ENTRIES {
        return Err(LkosError::InvalidInput(
            "archive has too many entries".into(),
        ));
    }
    let xml = read_zip_entry(&mut archive, "word/document.xml")?;
    Ok(ooxml_text(&xml, "w", "p", "t"))
}

/// XLSX: resolve the shared-strings table then emit `<c>` cells per `<row>`.
fn extract_xlsx(bytes: &[u8], filename: &str) -> Result<String> {
    let mut archive = open_zip(bytes, filename)?;
    if archive.len() > MAX_ARCHIVE_ENTRIES {
        return Err(LkosError::InvalidInput(
            "archive has too many entries".into(),
        ));
    }
    let shared = if archive.file_names().any(|n| n == "xl/sharedStrings.xml") {
        read_zip_entry(&mut archive, "xl/sharedStrings.xml")?
    } else {
        String::new()
    };
    let strings = collect_shared_strings(&shared);
    let sheet = read_zip_entry(&mut archive, "xl/worksheets/sheet1.xml")?;
    Ok(sheet_text(&sheet, &strings))
}

/// PPTX: text runs per slide, slides separated by blank lines.
fn extract_pptx(bytes: &[u8], filename: &str) -> Result<String> {
    let mut archive = open_zip(bytes, filename)?;
    if archive.len() > MAX_ARCHIVE_ENTRIES {
        return Err(LkosError::InvalidInput(
            "archive has too many entries".into(),
        ));
    }
    let mut names: Vec<String> = archive
        .file_names()
        .filter(|n| n.starts_with("ppt/slides/slide") && n.ends_with(".xml"))
        .map(|n| n.to_string())
        .collect();
    names.sort();
    let mut out = String::new();
    for name in &names {
        let xml = read_zip_entry(&mut archive, name)?;
        out.push_str(&ooxml_text(&xml, "a", "p", "t"));
        out.push_str("\n\n");
    }
    Ok(out)
}

/// EPUB: strip every XHTML content document to text.
fn extract_epub(bytes: &[u8], filename: &str) -> Result<String> {
    let mut archive = open_zip(bytes, filename)?;
    if archive.len() > MAX_ARCHIVE_ENTRIES {
        return Err(LkosError::InvalidInput(
            "archive has too many entries".into(),
        ));
    }
    let mut names: Vec<String> = archive
        .file_names()
        .filter(|n| (n.ends_with(".xhtml") || n.ends_with(".html")) && n.contains("content"))
        .map(|n| n.to_string())
        .collect();
    if names.is_empty() {
        names = archive
            .file_names()
            .filter(|n| n.ends_with(".xhtml") || n.ends_with(".html"))
            .map(|n| n.to_string())
            .collect();
    }
    names.sort();
    let mut out = String::new();
    for name in names.iter().take(512) {
        let html = read_zip_entry(&mut archive, name)?;
        out.push_str(&strip_markup(&html));
        out.push_str("\n\n");
    }
    Ok(out)
}

/// Extract `<w:p>`/`<a:p>` paragraphs' `<w:t>`/`<a:t>` runs into lines.
fn ooxml_text(xml: &str, ns: &str, para: &str, text: &str) -> String {
    let mut out = String::new();
    let p_open_a = format!("<{ns}:{para}>");
    let p_open_b = format!("<{ns}:{para} ");
    let p_close = format!("</{ns}:{para}>");
    let t_open = format!("<{ns}:{text}");
    let t_close = format!("</{ns}:{text}>");
    let mut rest = xml;
    while let Some(pos) = rest
        .find(&p_open_a)
        .map(|p| (p, p_open_a.len()))
        .or_else(|| {
            rest.find(&p_open_b).map(|p| {
                (
                    p,
                    rest[p..].find('>').map(|g| g + 1).unwrap_or(p_open_b.len()),
                )
            })
        })
    {
        let (start, open_len) = pos;
        let after = &rest[start + open_len..];
        let para_body = match after.find(&p_close) {
            Some(end) => &after[..end],
            None => after,
        };
        let mut line = String::new();
        let mut trun = para_body;
        while let Some(t_start) = trun.find(&t_open) {
            let t_body = &trun[t_start + t_open.len()..];
            if let Some(t_end) = t_body.find(&t_close) {
                let mut inner = t_body[..t_end].to_string();
                if let Some(gt) = inner.find('>') {
                    inner = inner[gt + 1..].to_string();
                }
                line.push_str(&decode_basic_entities(&inner));
                trun = &t_body[t_end + t_close.len()..];
            } else {
                break;
            }
        }
        if !line.trim().is_empty() {
            out.push_str(line.trim());
            out.push('\n');
        }
        rest = after;
    }
    out
}

/// Collect `<si>…<t>text</t>…</si>` entries of the XLSX shared-strings table.
fn collect_shared_strings(xml: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(si_start) = rest.find("<si>").or_else(|| rest.find("<si ")) {
        let after = &rest[si_start..];
        let si_body = match after.find("</si>") {
            Some(end) => &after[..end],
            None => break,
        };
        let mut text = String::new();
        let mut trun = si_body;
        while let Some(t_start) = trun.find("<t") {
            let t_body = &trun[t_start..];
            match t_body.find('>') {
                Some(gt) => match t_body[gt + 1..].find("</t>") {
                    Some(end) => {
                        text.push_str(&decode_basic_entities(&t_body[gt + 1..gt + 1 + end]));
                        trun = &t_body[gt + 1 + end + 4..];
                    }
                    None => break,
                },
                None => break,
            }
        }
        out.push(text);
        rest = &after[si_body.len()..];
    }
    out
}

/// Render XLSX sheet rows: cells resolved via shared strings or inline text.
fn sheet_text(xml: &str, strings: &[String]) -> String {
    let mut out = String::new();
    let mut rest = xml;
    while let Some(row_start) = rest.find("<row") {
        let after = &rest[row_start..];
        let row_body = match after.find("</row>") {
            Some(end) => &after[..end],
            None => break,
        };
        let mut cells: Vec<String> = Vec::new();
        let mut crun = row_body;
        while let Some(c_start) = crun.find("<c ").or_else(|| crun.find("<c>")) {
            let c_body = &crun[c_start..];
            let c_end = match c_body.find("</c>") {
                Some(end) => end,
                None => break,
            };
            let cell = &c_body[..c_end];
            let is_shared = cell.contains("t=\"s\"");
            let value =
                extract_tag(cell, "<v", "</v>").or_else(|| extract_tag(cell, "<is", "</is>"));
            if let Some(mut v) = value {
                if let Some(gt) = v.find('>') {
                    v = v[gt + 1..].to_string();
                }
                let decoded = decode_basic_entities(&v);
                if is_shared {
                    if let Ok(idx) = decoded.trim().parse::<usize>() {
                        if let Some(s) = strings.get(idx) {
                            cells.push(s.clone());
                        }
                    }
                } else if !decoded.trim().is_empty() {
                    cells.push(decoded.trim().to_string());
                }
            }
            crun = &c_body[c_end + 4..];
        }
        if !cells.is_empty() {
            out.push_str(&cells.join(" | "));
            out.push('\n');
        }
        rest = &after[row_body.len()..];
    }
    out
}

fn extract_tag(xml: &str, open: &str, close: &str) -> Option<String> {
    let start = xml.find(open)?;
    let body = &xml[start..];
    let end = body.find(close)?;
    Some(body[..end + close.len()].to_string())
}

/// Decode the five XML named entities + numeric forms (bounded, no DTD).
fn decode_basic_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.char_indices().peekable();
    while let Some((i, ch)) = chars.next() {
        if ch != '&' {
            out.push(ch);
            continue;
        }
        let rest = &s[i + 1..(i + 11).min(s.len())];
        let mut handled = false;
        if let Some(semi) = rest.find(';') {
            let ent = &rest[..semi];
            let decoded = match ent {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some(' '),
                _ => {
                    if let Some(hex) = ent.strip_prefix("#x").or_else(|| ent.strip_prefix("#X")) {
                        u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
                    } else if let Some(dec) = ent.strip_prefix('#') {
                        dec.parse::<u32>().ok().and_then(char::from_u32)
                    } else {
                        None
                    }
                }
            };
            if let Some(d) = decoded {
                out.push(d);
                for _ in 0..=(semi + 1) {
                    chars.next();
                }
                handled = true;
            }
        }
        if !handled {
            out.push('&');
        }
    }
    out
}

/// PDF text extraction (behind the optional `pdf` feature).
#[cfg(feature = "pdf")]
fn extract_pdf(filename: &str, bytes: &[u8]) -> Result<ExtractedDocument> {
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(LkosError::InvalidInput("pdf exceeds size cap".into()));
    }
    let text = pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| LkosError::InvalidInput(format!("'{filename}' pdf extraction failed: {e}")))?;
    let normalized = normalize(&text);
    if normalized.trim().is_empty() {
        return Err(LkosError::InvalidInput(format!(
            "'{filename}' contains no extractable text (scanned PDF? OCR is out of scope)"
        )));
    }
    Ok(ExtractedDocument {
        text: normalized,
        doc_type: crate::types::DocType::Text,
        language: None,
        filename: filename.to_string(),
        content_hash: content_hash(bytes),
        size: bytes.len() as u64,
        path_for_storage: None,
    })
}

#[cfg(not(feature = "pdf"))]
fn extract_pdf(filename: &str, _bytes: &[u8]) -> Result<ExtractedDocument> {
    Err(LkosError::UnsupportedFileType(format!(
        "'{filename}': PDF support is not compiled in; rebuild with `--features pdf`"
    )))
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
