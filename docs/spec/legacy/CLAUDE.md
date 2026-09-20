# CLAUDE.md — Technical Guide for Anthropic Claude / Autonomous Engineers

## Build & Test Commands
- Check compilation: `cargo check --manifest-path src-tauri/Cargo.toml`
- Run integration tests: `cargo test --manifest-path src-tauri/Cargo.toml --verbose`
- Build release: `cargo build --manifest-path src-tauri/Cargo.toml --release`

## Architecture Highlights
- **FastEmbed ONNX**: Model `all-MiniLM-L6-v2` runs locally via ONNX Runtime producing 384-dimensional `f32` vectors.
- **SQLite Engine**: Utilizes `rusqlite` bundled with SQLite 3.46+. Uses WAL mode, foreign keys enabled.
- **Hybrid RRF**: Combines dense vector cosine similarity and FTS5 BM25 keyword rankings using constant $k=60$ reciprocal rank fusion scaled by positional authority scores.
- **Subprocess Prompt Passing**: Prompts are passed to `llama-cli.exe` via temporary file `-f <path>` to bypass Windows 8,191-character CLI buffer overflows.

## Style Conventions
- Strict Rust 2021 edition.
- Error handling: Use `Result<T, String>` for IPC-facing APIs and typed errors (`thiserror`) internally.
- Zero N+1 queries: Always use batch SQL queries when enriching multi-chunk results.