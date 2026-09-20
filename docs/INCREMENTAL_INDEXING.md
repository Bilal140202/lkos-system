# Incremental Indexing

## What is incremental today (implemented)

1. **Unchanged documents**: SHA-256 content hash short-circuits re-ingestion
   (test-asserted idempotency, including 5x repeats).
2. **Model migration**: per-chunk `embedding_model` lineage (schema v5);
   `reembed_stale_chunks` migrates stale vectors in batches of 256 with
   progress + cancellation at 22.9k chunks/s measured.
3. **Changed documents**: new document row (old path released), fresh
   chunks+embeddings+knowledge; delete is graph-correct (subgraph recompute).

## What is deferred (with reasons)

- **Extractor-version-driven re-extraction**: original bytes are not retained
  (only hashes), so bumping EXTRACTOR_VERSION cannot re-extract historical
  documents; reopening the source and re-ingesting is the supported path.
  Retaining raw sources has a privacy cost and is an open design issue.
- **Chunk-level embedding reuse across documents** (same text hash re-embeds
  free): recorded as reserved API; not yet wired into the pipeline.

## Job plumbing for bulk work

train_semantic -> chains reembed; both report progress and honor
cancellation; failures back off exponentially and dead-letter.
