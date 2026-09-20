# NON_GOALS.md — What LKOS v0.1 Deliberately Does Not Do

The project standard (spec §88, §127): *every abstraction must earn its existence.*
This register lists tempting complexity that is **not** implemented, with the entry
criteria that would justify building it. Documentation never outruns implementation.

## Not built (and why)

| Non-goal | Why not now | Entry criterion |
|---|---|---|
| Distributed / cluster mode | Local-first thesis; single-file portability is the product | A demonstrated multi-device sync demand from an embedding app |
| External vector database | SQLite + brute force is measured sufficient at v0.1 scale (dense p50 ≈ 14 ms @ 3.6k chunks) | > 250k chunks (`max_dense_scan` ceiling) in a real library |
| HNSW / ANN index | Same as above; brute force is honest O(N) and cache-friendly | Same, plus ANN benchmark showing >10× dense speedup at recall ≥ 0.99 |
| PDF / DOCX / OCR extraction | Binary inputs are rejected with a typed error; format scope is md/txt/code/data | A maintained extractor (e.g. `pdf-extract`) passing the fuzz + golden pipeline |
| Graph database (Neo4j-style) | 1-hop co-occurrence in SQL covers demonstrated demand (ADR-007) | Multi-hop traversal queries proven by real workloads |
| Embedding-based entity linking | Suffix/case folding resolves the measured collisions; embedding linking adds false merges | A labeled alias dataset showing fold-only recall < 90% |
| Multi-user, permissions, namespaces | Single-user local engine is the target | An application needs isolated domains (schema has room: per-row metadata) |
| Connectors (Drive/Notion/Slack/web) | Core must not depend on connectors | Connector feeds the public ingest API only; never the reverse |
| Plugin runtime | Security boundary is a project, not a flag | Signed-plugin design doc + capability model |
| Python / TypeScript bindings | Clean core first (spec §54) | API stabilized one minor cycle after v0.2 |
| Multimodal (image/audio/video) | Text pipeline must be boring and solid first | Embedding abstraction proven with a second text provider |
| Kubernetes / Kafka / microservices | Category error for an embedded engine | Never — unless the engine is forked into a server, which would be a different product |
| LLM inside the query path | Planner is deterministic by design (ADR-008) | Ablation shows an LLM planner beats rules on a held-out query set |
| Human-labeled benchmark claims | Self-supervised numbers only, labeled as such | A committed golden dataset with human judgments |

---

## v0.9 deltas (what changed)

Moved OUT of non-goals (now implemented): semantic embeddings (LSA provider),
BM25 score preservation, entity alias/merge APIs, graph-correct deletion,
DOCX/XLSX/PPTX/EPUB ingestion, job backoff/dead-letter/cancellation.

Still non-goals (with status):
- **ANN/HNSW** — brute force with documented cap; adoption gated on measured
  crossover (research frontier 2).
- **Pretrained neural encoders as default** — local-first zero-download
  contract; ONNX/fastembed stays an optional-provider design task.
- **NER-grade extraction / NLI contradiction detection** — deterministic
  regexes, honestly labeled; frontier item 5.
- **OCR, audio, video** — multimodal foundation later.
- **Multi-tenant ACLs, E2E encryption** — single-user local file first.
- **Learned sparse retrieval (SPLADE-style)** — requires model artifacts.
- **Graph database engines** — SQL traversal sufficient at current scale;
  must benchmark better before adoption.
