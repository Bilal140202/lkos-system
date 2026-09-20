# LKOS — Local Knowledge Object System

> **Universal On-Device Neuro-Symbolic Hybrid Retrieval & Knowledge Graph Engine for Rust / Tauri Applications**

LKOS is an offline, privacy-first knowledge extraction, indexing, and retrieval architecture designed for edge devices and desktop AI applications. It unifies dense vector embeddings with sparse BM25 keyword matching, deterministic heading heuristics, position-based authority scoring, and zero-latency entity graph extraction into a unified SQLite storage substrate.

---

## Architectural Principles (Carmack First-Principles)

1. **Deterministic Edge Guarantees**: 100% on-device operation. Zero telemetry, zero external API dependencies for vector storage, search, or inference.
2. **Subprocess Pipe Isolation**: Local LLM execution is decoupled into an isolated `llama-cli` subprocess communicating via temporary memory/file descriptors (`-f`). This eliminates the Windows 8,191-character CLI buffer limit and prevents `STATUS_STACK_BUFFER_OVERRUN` (0xC0000409) crashes on long context prompts.
3. **Zero N+1 Query Overhead**: Reciprocal Rank Fusion (RRF) scores are combined with batch-loaded chunk authority scores in single SQL queries (`batch_authority_scores`).
4. **Resilient Background Pipelines**: Non-blocking document ingestion. Indexing completes immediately (`indexing`), followed by asynchronous background section detection, entity extraction, and optional background LLM summarization (`indexing` -> `ready` -> `summarizing` -> `complete`).

---

## The 5 Phases of LKOS

```
                 RAW DOCUMENT INGESTION
                           │
                 [Multi-Format Extractor]
             (PDF, DOCX, Code, Text, Images)
                           │
                 [Smart Boundary Chunker]
             (Preserves functions & classes)
                           │
                 [FastEmbed ONNX Engine]
                  (all-MiniLM-L6-v2)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 1: DUAL SUBSTRATE                     │
│  - Documents Table: File metadata, paths, size, doc_type    │
│  - Chunks Table: Text, source_page, 384-d f32 BLOB embedding│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 2: READINESS & SECTIONS               │
│  - Heuristic Section Header Parsing (Regex, zero LLM cost)  │
│  - Section title propagation per chunk                      │
│  - Readiness state: 'indexing' -> 'ready' -> 'complete'     │
│  - Optional executive summary caching                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 3: KNOWLEDGE ENRICHMENT               │
│  - KnowledgeObject Extraction:                              │
│    * Named Entities (Dates, Monetary values, Orgs, Names)   │
│    * Top Keywords (TF-IDF without stopwords)                │
│    * Positional Authority Scoring (Intro 1.25x, Concl 1.15x)│
│  - Stored in `chunks.knowledge_json`                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 4: HYBRID RRF RETRIEVAL               │
│  - Vector Cosine Similarity (Dense)                         │
│  - FTS5 Porter Stemmer / unicode61 (Sparse BM25)            │
│  - Parallel asynchronous execution via tokio::join!         │
│  - Weighted Reciprocal Rank Fusion:                         │
│    Score = [ (0.5 / (60 + R_vec)) + (0.5 / (60 + R_fts)) ]  │
│            * Authority_Score                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 5: RELATIONAL ENTITY GRAPH            │
│  - `entity_index` Table: (entity, doc_id, chunk_id, type)   │
│  - Cross-document entity search (`search_by_entity`)        │
│  - Document entity badges & relationship exploration        │
└─────────────────────────────────────────────────────────────┘
```

---

## Cross-Application Integration Guide

LKOS can be integrated into any Rust or Tauri v2 project.

### 1. Database Schema Initialization

LKOS automatically applies idempotent schema migrations on boot:
- `documents` & `chunks` base schema
- `001_rag_enhancements.sql`: FTS5 virtual table + triggers + indexes
- LKOS Phase 2 (`v1.2.0`): `readiness_state`, `section_count`, `summary`
- LKOS Phase 3 (`v1.3.0`): `knowledge_json`
- LKOS Phase 5 (`v1.4.0`): `entity_index`

### 2. Executing Search

```rust
use lkos_system::vector::store::VectorStore;
use lkos_system::vector::types::SearchFilters;

let store = VectorStore::global()?.lock();

// 1. Parallel Hybrid Search (Vector + FTS5 + Authority Weighting)
let results = store.hybrid_search("financial revenue 2026", query_embedding, 10, &SearchFilters::default())?;

// 2. Entity Graph Lookup
let entity_chunks = store.search_by_entity("Google", 5)?;
```

---

## Directory Layout

```
lkos-system/
├── .github/workflows/backend-ci.yml   # Automated Windows CI (cargo check + test)
├── AGENTS.md                          # Multi-agent collaboration protocols
├── CLAUDE.md                          # Claude & LLM agent technical guidance
├── README.md                          # Architecture & integration specification
└── src-tauri/                         # Complete Rust implementation
    ├── Cargo.toml                     # Dependencies (rusqlite, fastembed, tauri v2)
    ├── bin/                           # llama-cli.exe runtime runners
    ├── src/
    │   ├── background.rs              # Asynchronous LKOS pipeline coordinator
    │   ├── commands/                  # Tauri IPC handler layer
    │   ├── document/                  # Extractors & syntax-aware chunker
    │   ├── embeddings/                # FastEmbed ONNX runner
    │   ├── llm/                       # Isolated llama.cpp runner & templates
    │   ├── rag/                       # KnowledgeObject & Domain prompts
    │   ├── vector/                    # SQLite store, FTS5, RRF & Entity graph
    │   ├── lib.rs                     # Library entrypoint
    │   └── main.rs                    # Process binary entrypoint
```