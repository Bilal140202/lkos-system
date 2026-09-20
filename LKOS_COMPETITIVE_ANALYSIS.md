# LKOS Competitive Analysis (v0.9)

Honest positioning against the systems a user would actually compare.

| System | What it is | Where LKOS differs |
|---|---|---|
| LlamaIndex / LangChain retrievers | App-framework retrieval pipelines | LKOS is an embeddable engine with persistent typed knowledge (entities/claims/graph/provenance), not per-request plumbing |
| Qdrant / Milvus | Networked vector databases | LKOS is single-file, offline, no server; vector search is one channel of a hybrid stack |
| sqlite-vec | SQLite vector extension | complementary future substrate for ANN; LKOS currently brute-forces with a cap (documented) |
| Tantivy / Tantivy-based tools | Rust full-text engine | LKOS uses SQLite FTS5 (single-file durability, WAL, SQL filters); learned-sparse remains out of scope |
| GraphRAG-style pipelines | LLM-driven graph summarization | LKOS builds deterministic typed graphs at ingest; no LLM in the pipeline |
| AnythingLLM / Open WebUI | Chat UIs over documents | LKOS is the engine those apps could embed (knowledge DB + search + evidence + optional AI) |

Differentiation summary: local-first single-file knowledge substrate with
typed evidence and provenance — none of the compared systems ship all four
of (entities, claims-with-taxonomy, typed graph, span provenance) offline in
one portable file. LKOS's honest gaps (ANN scale, NER-grade extraction,
pretrained encoders) are the same gaps its roadmap sequences.
