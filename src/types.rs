//! Core data types of the LKOS knowledge model.
//!
//! These types form the *canonical data model* (see `docs/architecture/DATA_MODEL.md`).
//! Every artifact that LKOS derives from a source document is representable,
//! versioned, and traceable back to its evidence via provenance records.

use serde::{Deserialize, Serialize};

/// Lifecycle state of a document inside the background pipeline.
///
/// State machine: `indexing -> ready -> summarizing -> complete`
/// (`error` is a terminal failure state, `pending` means queued, not started).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadinessState {
    /// Queued, no work done yet.
    Pending,
    /// Chunks + embeddings are being written.
    Indexing,
    /// Indexed and searchable (embeddings + FTS are consistent).
    Ready,
    /// LLM summary being generated (searchable meanwhile).
    Summarizing,
    /// Summary stored; pipeline finished.
    Complete,
    /// Pipeline failed for this document (error message in events/logs).
    Error,
}

impl ReadinessState {
    /// Canonical string form used in the database.
    pub fn as_str(&self) -> &'static str {
        match self {
            ReadinessState::Pending => "pending",
            ReadinessState::Indexing => "indexing",
            ReadinessState::Ready => "ready",
            ReadinessState::Summarizing => "summarizing",
            ReadinessState::Complete => "complete",
            ReadinessState::Error => "error",
        }
    }

    /// Parse from the database string form.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(ReadinessState::Pending),
            "indexing" => Some(ReadinessState::Indexing),
            "ready" => Some(ReadinessState::Ready),
            "summarizing" => Some(ReadinessState::Summarizing),
            "complete" => Some(ReadinessState::Complete),
            "error" => Some(ReadinessState::Error),
            _ => None,
        }
    }
}

/// Coarse content type assigned at ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocType {
    /// Plain text.
    Text,
    /// Markdown / reStructuredText-like structured prose.
    Markdown,
    /// Source code (language detected by extension).
    Code,
    /// JSON / CSV / XML data files.
    Data,
    /// Unknown binary content (rejected unless forced).
    Binary,
}

impl DocType {
    /// Canonical string form.
    pub fn as_str(&self) -> &'static str {
        match self {
            DocType::Text => "text",
            DocType::Markdown => "markdown",
            DocType::Code => "code",
            DocType::Data => "data",
            DocType::Binary => "binary",
        }
    }
}

/// Semantic category of a chunk produced by the chunker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkKind {
    /// Prose paragraph.
    Prose,
    /// Document heading (short, structural).
    Heading,
    /// Generic source code block.
    Code,
    /// A top-level code unit: function, class, struct, impl, etc.
    CodeUnit,
    /// Key/value or tabular row material.
    DataRow,
}

impl ChunkKind {
    /// Canonical string form.
    pub fn as_str(&self) -> &'static str {
        match self {
            ChunkKind::Prose => "prose",
            ChunkKind::Heading => "heading",
            ChunkKind::Code => "code",
            ChunkKind::CodeUnit => "code_unit",
            ChunkKind::DataRow => "data_row",
        }
    }

    /// Parse from the database string form.
    pub fn from_db(s: &str) -> Self {
        match s {
            "heading" => ChunkKind::Heading,
            "code" => ChunkKind::Code,
            "code_unit" => ChunkKind::CodeUnit,
            "data_row" => ChunkKind::DataRow,
            _ => ChunkKind::Prose,
        }
    }
}

/// Describes a stored document (one row of `documents`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInfo {
    /// Stable numeric id.
    pub id: i64,
    /// Original filesystem path (may be empty for in-memory ingestion).
    pub path: String,
    /// File name.
    pub filename: String,
    /// Coarse type.
    pub doc_type: String,
    /// Raw size in bytes.
    pub size: u64,
    /// SHA-256 of the raw bytes, hex encoded. Uniqueness drives incremental indexing.
    pub content_hash: String,
    /// Extracted plain-text length in characters.
    pub text_chars: usize,
    /// Pipeline readiness state.
    pub readiness_state: String,
    /// Optional LLM summary (empty until `complete`).
    pub summary: Option<String>,
    /// When the summary was generated.
    pub summary_generated_at: Option<String>,
    /// Number of detected sections.
    pub section_count: i64,
    /// Number of chunks.
    pub chunk_count: i64,
    /// ISO-8601 creation timestamp (first ingestion).
    pub created_at: String,
    /// ISO-8601 timestamp of last successful (re)index.
    pub indexed_at: Option<String>,
}

/// Describes a stored chunk (one row of `chunks`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    /// Stable numeric id (rowid).
    pub id: i64,
    /// Owning document.
    pub document_id: i64,
    /// Position within the document (0-based).
    pub chunk_index: i64,
    /// Chunk text.
    pub text: String,
    /// Forward-propagated section heading, if any.
    pub section_title: Option<String>,
    /// Semantic kind.
    pub kind: String,
    /// Byte offsets into the normalized source text.
    pub start_offset: i64,
    /// End offset (exclusive).
    pub end_offset: i64,
    /// Positional authority multiplier (see RETRIEVAL.md).
    pub authority_score: f32,
    /// SHA-256 of chunk text (incremental reuse of embeddings).
    pub content_hash: String,
}

/// One retrieval result with full explainability metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Chunk id.
    pub chunk_id: i64,
    /// Document id.
    pub document_id: i64,
    /// Document filename.
    pub document: String,
    /// Section title, if known.
    pub section: Option<String>,
    /// Chunk text (possibly truncated for display).
    pub text: String,
    /// Final fused score (higher is better).
    pub score: f32,
    /// 1-based final rank.
    pub rank: usize,
    /// Which channels matched and at what internal ranks.
    pub matched_by: Vec<MatchSource>,
    /// Authority multiplier applied.
    pub authority_score: f32,
}

/// Explanation of why a chunk was retrieved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchSource {
    /// Matched by dense vector cosine similarity.
    Vector {
        /// 1-based rank inside the dense channel.
        rank: usize,
        /// Raw cosine similarity in [-1, 1].
        cosine: f32,
    },
    /// Matched by lexical BM25 (FTS5) search.
    Fts {
        /// 1-based rank inside the lexical channel.
        rank: usize,
        /// FTS5 bm25 score (negative; lower is better).
        bm25: f32,
    },
    /// Boosted because a known entity appears in the chunk.
    Entity {
        /// Matched entity display name.
        name: String,
    },
    /// Boosted because the exact query phrase occurs in the chunk.
    Phrase,
    /// Boosted because the query matched the document's section title.
    SectionTitle,
}

/// High-level query intent detected by the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryIntent {
    /// "summarise my document" style queries.
    Summary,
    /// Entity-centric queries ("what does OpenAI say...", "who is John Smith").
    Entity,
    /// Temporal queries ("what changed in 2026", "revenue before 2024").
    Temporal,
    /// Exact lookups: quoted phrases, identifiers, file names.
    Exact,
    /// Default natural-language semantic/prose queries.
    Semantic,
    /// Cross-document comparison / aggregation.
    Comparative,
}

impl QueryIntent {
    /// Canonical string form.
    pub fn as_str(&self) -> &'static str {
        match self {
            QueryIntent::Summary => "summary",
            QueryIntent::Entity => "entity",
            QueryIntent::Temporal => "temporal",
            QueryIntent::Exact => "exact",
            QueryIntent::Semantic => "semantic",
            QueryIntent::Comparative => "comparative",
        }
    }
}

/// Retrieval mode overriding automatic planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalMode {
    /// Detect intent and plan automatically (default).
    Auto,
    /// Only lexical BM25.
    LexicalOnly,
    /// Only dense vector.
    VectorOnly,
    /// Both channels fused with RRF (the default plan body).
    Hybrid,
    /// Entity index lookup only.
    EntityLookup,
}

/// Filters restricting the candidate set.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Filters {
    /// Restrict to these document ids.
    pub document_ids: Option<Vec<i64>>,
    /// Restrict to these document types ("markdown", "code", ...).
    pub doc_types: Option<Vec<String>>,
    /// Require this phrase (case-insensitive) in chunk text.
    pub must_contain: Option<String>,
    /// Require all listed entity names to be mentioned in the chunk's document.
    pub entities: Option<Vec<String>>,
    /// Temporal filter: only chunks whose document was indexed at/before this RFC-3339 date.
    pub as_of: Option<String>,
}

/// A structured query request (the "power mode" API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Natural language query text.
    pub text: String,
    /// Number of hits to return (default 8).
    pub top_k: usize,
    /// Retrieval mode.
    pub mode: RetrievalMode,
    /// Candidate filters.
    pub filters: Filters,
    /// Include per-hit provenance in the response.
    pub include_provenance: bool,
    /// Include graph context (related entities) in the response.
    pub include_graph: bool,
    /// Character budget for the assembled context block.
    pub context_budget: usize,
    /// Maximum chunks per document in the assembled context (diversity).
    pub max_per_document: usize,
}

impl QueryRequest {
    /// Create a request with defaults.
    pub fn new(text: impl Into<String>) -> Self {
        QueryRequest {
            text: text.into(),
            top_k: 8,
            mode: RetrievalMode::Auto,
            filters: Filters::default(),
            include_provenance: false,
            include_graph: false,
            context_budget: 3000,
            max_per_document: 3,
        }
    }

    /// Override top_k.
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Override retrieval mode.
    pub fn mode(mut self, mode: RetrievalMode) -> Self {
        self.mode = mode;
        self
    }
}

/// Provenance record attached to a hit when `include_provenance` is set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitProvenance {
    /// Artifact type (always "chunk" for hits).
    pub artifact_type: String,
    /// Document filename.
    pub document: String,
    /// Document content hash.
    pub content_hash: String,
    /// Extractor chain that produced this chunk.
    pub extractor: String,
    /// Offsets of the chunk inside the normalized text.
    pub offsets: (i64, i64),
}

/// Full query response: hits + assembled context + explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Detected intent (or the forced one).
    pub intent: QueryIntent,
    /// Human-readable plan description ("why these channels").
    pub plan_explanation: String,
    /// Ranked hits.
    pub hits: Vec<SearchHit>,
    /// Assembled context block (budgeted, de-duplicated, doc-diverse).
    pub context: String,
    /// Per-hit provenance, when requested.
    pub provenance: Vec<HitProvenance>,
    /// Entities related to the query (when requested or detected).
    pub related_entities: Vec<EntitySummary>,
    /// Latency of the full query in microseconds.
    pub elapsed_us: u128,
}

/// Compact entity summary for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySummary {
    /// Entity id.
    pub id: i64,
    /// Display name.
    pub display_name: String,
    /// Type string.
    pub entity_type: String,
    /// Total mention count.
    pub mention_count: i64,
}

/// A resolved entity (row of `entities`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    /// Entity id.
    pub id: i64,
    /// Canonical key (normalized for resolution).
    pub canonical_key: String,
    /// Best display surface.
    pub display_name: String,
    /// Type label: person|organization|location|date|money|percent|email|url|concept.
    pub entity_type: String,
    /// Number of mentions across the library.
    pub mention_count: i64,
    /// Known aliases (surface forms).
    pub aliases: Vec<String>,
}

/// An entity mention with provenance (row of `entity_mentions`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMention {
    /// Mentioning chunk.
    pub chunk_id: i64,
    /// Mentioning document.
    pub document_id: i64,
    /// Surface form as written.
    pub surface: String,
    /// Entity type.
    pub entity_type: String,
    /// Character offsets within the chunk text.
    pub offsets: (i64, i64),
    /// Heuristic confidence in [0,1] (labeled `heuristic`).
    pub confidence: f32,
}

/// An extracted claim (row of `claims`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimRecord {
    /// Claim id.
    pub id: i64,
    /// Subject phrase.
    pub subject: String,
    /// Predicate phrase (normalized verb or relation).
    pub predicate: String,
    /// Object phrase.
    pub object: String,
    /// Original sentence.
    pub sentence: String,
    /// Source document.
    pub document_id: i64,
    /// Source chunk.
    pub chunk_id: i64,
    /// Heuristic confidence (labeled as such).
    pub confidence: f32,
    /// Extractor identifier (e.g. `claim-heuristics-v1`).
    pub extractor: String,
    /// Optional validity start parsed from the sentence.
    pub valid_from: Option<String>,
    /// Optional validity end parsed from the sentence.
    pub valid_until: Option<String>,
    /// Creation timestamp.
    pub created_at: String,
}

/// A detected numeric conflict between two claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRecord {
    /// Conflict id.
    pub id: i64,
    /// Subject key shared by both claims.
    pub subject_key: String,
    /// Predicate key shared by both claims.
    pub predicate_key: String,
    /// First claim id.
    pub claim_a: i64,
    /// Second claim id.
    pub claim_b: i64,
    /// Relative numeric delta.
    pub delta: f32,
    /// Human-readable explanation including any temporal explanation.
    pub explanation: String,
}

/// A relationship edge between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipRecord {
    /// Relationship id.
    pub id: i64,
    /// Source entity id.
    pub source_entity_id: i64,
    /// Target entity id.
    pub target_entity_id: i64,
    /// Relationship type (e.g. `CO_OCCURS_WITH`).
    pub relationship_type: String,
    /// Aggregate weight (co-occurrence count).
    pub weight: f32,
    /// Last document contributing to this edge.
    pub last_document_id: i64,
}

/// A provenance record — the evidence trail of every derived artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// Provenance row id.
    pub id: i64,
    /// Artifact type: chunk|entity_mention|claim|relationship|summary.
    pub artifact_type: String,
    /// Artifact id as string.
    pub artifact_id: String,
    /// Source document id.
    pub document_id: i64,
    /// Source chunk id (when applicable).
    pub chunk_id: Option<i64>,
    /// Offsets in the chunk/document text.
    pub offsets: Option<(i64, i64)>,
    /// Extractor name.
    pub extractor: String,
    /// Extractor version.
    pub extractor_version: String,
    /// Creation time.
    pub created_at: String,
}

/// Background pipeline event delivered to subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Event name, e.g. `document.status`, `entity.discovered`, `claim.extracted`.
    pub name: String,
    /// Subject id (document/entity/claim id).
    pub subject_id: i64,
    /// Human payload (filename, entity name, status, ...).
    pub payload: String,
    /// Event time.
    pub at: String,
}

/// Aggregate library statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryStats {
    /// Number of documents.
    pub documents: i64,
    /// Number of chunks.
    pub chunks: i64,
    /// Number of entities.
    pub entities: i64,
    /// Number of entity mentions.
    pub entity_mentions: i64,
    /// Number of claims.
    pub claims: i64,
    /// Number of detected conflicts.
    pub conflicts: i64,
    /// Number of relationship edges.
    pub relationships: i64,
    /// Number of provenance records.
    pub provenance_records: i64,
    /// Database file size in bytes.
    pub db_size_bytes: u64,
}

/// Version bundle recorded per document for incremental indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Content hash of raw bytes.
    pub content_hash: String,
    /// Extractor implementation version.
    pub extractor_version: String,
    /// Chunker implementation version.
    pub chunker_version: String,
    /// Embedding provider name.
    pub embedding_model: String,
    /// Knowledge extractor version.
    pub knowledge_version: String,
}
