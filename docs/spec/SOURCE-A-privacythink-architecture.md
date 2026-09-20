# PrivacyThink — LKOS Architecture
## Local Knowledge Operating System

> *"PrivacyThink is not a RAG chatbot. It is a Privacy-First Local Knowledge Operating System — a system that continuously transforms documents into structured organizational knowledge."*

---

## Vision

A traditional RAG system answers queries at runtime by retrieving chunks and passing them to an LLM. LKOS front-loads the intelligence:

```
RAG (Old):   Upload → Embed → [query] → Search → LLM → Answer
LKOS (New):  Upload → Embed → Pipeline → Knowledge Stored
                                                    ↓
                                          [query] → Instant Answer
```

The user experience goal: **ask any question and get a confident, grounded answer in under 3 seconds.**

---

## Implementation Status

### ✅ Phase 1 — RAG Foundation (COMPLETE)
*Was the baseline before LKOS work began.*

| Component | Status | Notes |
|---|---|---|
| Document extraction (PDF, DOCX, TXT, code) | ✅ | `document/chunker.rs` |
| Chunk embedding (local sentence-transformers) | ✅ | `embeddings/generator.rs` |
| SQLite vector store with cosine similarity | ✅ | `vector/store.rs` |
| FTS5 hybrid search (vector + keyword + RRF fusion) | ✅ | `vector/store.rs` |
| Streaming LLM inference via llama-run subprocess | ✅ | `llm/inference.rs` |
| Document-grounded prompt builder | ✅ | `Chat.tsx` |
| English-anchored responses | ✅ | Prompt ends with `Answer in English:` |

**Bugs fixed during Phase 1:**
- Removed `chunk_index < 3` filter from `get_all_document_starts` (caused context starvation)
- Increased `MAX_CONTEXT_CHARS` from 2000 → 3000
- Bumped summary chunk count from 5 → 10
- Fixed FeedbackModal CSS collision (`.btn-send` scoped)
- Bumped UI inference timeout from 60s → 300s

---

### ✅ Phase 2 — Background Knowledge Pipeline (COMPLETE — TESTING)
*Implemented July 2026.*

**Core idea:** Documents are summarized *at index time*, not per query. Summary queries become instant.

#### DB Schema (migration v1.2.0)
New columns added to `documents` table:
```sql
summary TEXT                    -- Pre-built LLM summary
readiness_state TEXT            -- 'indexing'|'ready'|'summarizing'|'complete'
section_count INTEGER           -- Number of detected headings
summary_generated_at TEXT       -- Timestamp of summary generation
```

#### Backend Changes

| File | What Changed |
|---|---|
| `vector/types.rs` | `DocumentInfo` has `summary`, `readiness_state`, `section_count`, `summary_generated_at` |
| `vector/store.rs` | Migration v1.2.0 · `add_document` inserts `readiness_state='indexing'` · `list_documents`/`get_document` fetch new columns · 5 new helper methods |
| `background.rs` *(new)* | Async pipeline: detect sections → mark ready → generate LLM summary → mark complete |
| `commands/vector.rs` | `index_document` now accepts `AppHandle` and fires pipeline as fire-and-forget Tokio task · New `get_document_summary` command · New `get_document_readiness` command |
| `lib.rs` | `pub mod background` registered · 2 new commands in handler |

#### Pipeline State Machine
```
Document uploaded
      │
      ▼
  [indexing]  ← embeddings + SQLite writes
      │
      ▼ (spawn_blocking completes)
   [ready]  ←── Tauri event: "document-ready"
      │         Users can ask questions NOW
      │
      ▼ (if LLM model loaded)
[summarizing] ←── Tauri event: "document-status"
      │
      ▼ (LLM generates 3-4 sentence summary)
  [complete] ←── Tauri event: "document-summary-ready"
                  Summary stored in DB permanently
```

#### Section Detection (regex-based, zero-cost)
Heading heuristics applied per chunk:
1. Line is ALL CAPS and contains letters
2. Line ends with a colon `:`
3. Line starts with a keyword: `section`, `chapter`, `introduction`, `conclusion`, `summary`, `background`, `appendix`, `abstract`, `overview`, `methodology`, `results`, `discussion`, `references`

Section titles propagate forward across chunks until a new heading is detected.

#### Frontend Changes

| File | What Changed |
|---|---|
| `Library.tsx` | Listens to 3 Tauri events · Renders readiness badges per document |
| `Library.css` | Badge styles for all 5 states (indexing/ready/summarizing/complete/error) with micro-animations |
| `Chat.tsx` | **Pre-built summary fast path:** calls `get_document_summary` before RAG; returns instantly if found |

#### Pre-built Summary Fast Path (Chat.tsx)
```
User: "summarise my document"
         │
         ▼
  queryType === 'summary'?
         │ YES
         ▼
  invoke('get_document_summary', { docId })
         │
   ┌─────┴──────┐
   │ Found!     │ Not found yet
   ▼            ▼
Instant reply  Fall through to RAG chunks → LLM
(no LLM call)
```

---

### 🔲 Phase 3 — KnowledgeObject Abstraction (PLANNED)

**Core idea:** Every chunk becomes a `KnowledgeObject` with rich metadata, not just raw text.

```rust
struct KnowledgeObject {
    chunk_id: String,
    text: String,
    section_title: Option<String>,    // Already built in Phase 2
    entities: Vec<Entity>,            // Names, dates, amounts, orgs
    authority_score: f32,             // Intro/conclusion chunks score higher
    relevance_threshold: f32,         // Filter low-quality matches
    keywords: Vec<String>,            // Extracted key terms
}
```

**What this enables:**
- Context assembly filters out low-quality chunks (no more filler)
- Entity-aware answers ("what does John Smith say about...?")
- Smarter citation: "Source: Introduction → Methodology → Results"
- Authority weighting: conclusion chunks get higher retrieval weight

**Implementation plan:**
1. `src-tauri/src/rag/knowledge_object.rs` — new struct + extraction logic
2. Entity extraction: regex for names (capitalized consecutive words), dates (`\d{4}`, `Jan 2024`), amounts (`$\d+`, `\d+%`)
3. Authority scoring: chunk_index 0-2 → 1.2x, last 10% → 1.1x, middle → 1.0x
4. Store as JSON in new `chunks.knowledge_json` column (migration v1.3.0)
5. Update hybrid search to include authority score in RRF ranking

---

### 🔲 Phase 4 — Parallel Query Engine (PLANNED)

**Core idea:** Fan out search queries in parallel instead of sequential.

```
Current:  embed → vector_search → fts_search → RRF → assemble → LLM
Planned:  embed ──┬── vector_search ──┐
                  ├── fts_search ─────┤ → RRF → dedupe → assemble → LLM
                  └── summary_fetch ──┘
```

**Expected gain:** 40-60% latency reduction on query response.

**Implementation:**
- `tokio::join!` for parallel search arms
- Pre-fetch document summary concurrently with search
- Deduplicate overlapping chunks before context assembly

---

### 🔲 Phase 5 — Cross-Document Intelligence (FUTURE)

- Cross-document entity linkage ("Person X appears in 3 documents...")
- Topic clustering across the library
- Contradiction detection between documents
- Knowledge graph visualization

---

## Testing Checklist (Phase 2)

### Manual Tests

| Test | Expected Result | Status |
|---|---|---|
| Upload a document | Library shows badge: `⏳ Indexing...` | 🔲 |
| After indexing completes | Badge changes to: `⚡ Ready` | 🔲 |
| If LLM model loaded during upload | Badge transitions `🧠 Building summary...` then `✨ Summary ready` | 🔲 |
| Ask "summarise my document" (model loaded, doc complete) | Instant reply — **no spinner**, no LLM wait | 🔲 |
| Ask "summarise my document" (doc still summarizing) | Falls through to RAG chunk path | 🔲 |
| Ask a specific question | Normal hybrid search — unaffected | 🔲 |
| Delete document | Removes from list, cleanup works | 🔲 |
| Reload Library page | Existing docs show correct badge state from DB | 🔲 |
| No model loaded at upload time | Doc goes to `⚡ Ready`, no crash | 🔲 |

### Console Logs to Watch

```
🔄 LKOS background pipeline starting for doc: <id>
📑 Detected N sections in doc: <id>
✅ Doc marked ready: <id>
🤖 Generating LLM summary (prompt: N chars)
🧠 Summary saved for doc: <id>
⚡ Pre-built summary found — returning instantly   ← This is the win
```

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────┐
│                    PrivacyThink LKOS                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  INGESTION LAYER                                           │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐ │
│  │  Document    │───▶│  Chunker     │──▶│  Embedder    │ │
│  │  Extractor   │    │  (chunker.rs)│   │ (generator)  │ │
│  └──────────────┘    └──────────────┘   └──────┬───────┘ │
│                                                │           │
│  STORAGE LAYER                                 ▼           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │            SQLite Vector Store                      │  │
│  │  documents: id, filename, summary✨, readiness✨    │  │
│  │  chunks: id, text, embedding, section_title✨       │  │
│  │  chunks_fts: (FTS5 full-text index)                 │  │
│  └─────────────────────────────────────────────────────┘  │
│                            │                               │
│  BACKGROUND PIPELINE ✨    │  (async, fire-and-forget)    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  detect_sections → mark_ready → generate_summary    │  │
│  │  → store_summary → mark_complete → emit_events      │  │
│  └─────────────────────────────────────────────────────┘  │
│                            │                               │
│  QUERY LAYER               ▼                               │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐ │
│  │  Pre-built   │    │  Hybrid      │   │  Context     │ │
│  │  Summary✨   │    │  Search      │   │  Assembler   │ │
│  │  (instant)   │    │  (vec + fts) │   │              │ │
│  └──────┬───────┘    └──────┬───────┘   └──────┬───────┘ │
│         │                   │                   │          │
│  GENERATION LAYER           ▼                   ▼          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         LLM (llama-run subprocess)                  │  │
│  │         Streaming · English-anchored · 300s timeout │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘

✨ = Added in Phase 2
```

---

## Key Constraints

| Constraint | Value | Reason |
|---|---|---|
| Max context chars | 3000 | Small model (Qwen 0.5B) context window |
| Summary chunks | 10 | More context for better summaries |
| Summary prompt cap | 3000 chars | Fits within LLM context |
| Summary max tokens | 256 | Dense but concise output |
| Summary temperature | 0.2 | Factual, low-variance output |
| Inference timeout | 300s | Prevents UI hang on slow hardware |
| Emoji in prompts | ❌ | Confuses small models — plain ASCII only |
