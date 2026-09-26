# LKOS Research Atlas (v0.9; v0.10 addition: Malkov & Yashunin)

Per-source records for the works that materially shaped v0.9 decisions.
(Full 2.0 rewrite with per-paper reading notes tracked as issue #19.)
(YouTube/lecture sources were consulted conceptually; only citable,
verifiable references are listed — nothing invented.)

| Source | Year | Key idea | Adopt/Modify/Reject | LKOS evidence |
|---|---|---|---|---|
| Deerwester et al., Indexing by LSA (JASIS 41(6)) | 1990 | corpus-latent semantic space | Adopt | embeddings/lsa.rs; determinism + topical-separation tests |
| Halko/Martinsson/Tropp, Randomized SVD (SIAM Rev 53(2)) | 2011 | O(nnz*l*q) factorization | Adopt | fixed-seed LCG + 4 power iterations in lsa::train |
| Robertson/Zaragoza, BM25 (FnTIR 3(4)) | 2009 | tf saturation, length norm | Adopt | FTS5 bm25() preserved in MatchSource::Fts |
| Cormack et al., RRF (SIGIR) | 2009 | rank fusion without calibration | Adopt | k=60 default, per-intent weights |
| Carbonell/Goldstein, MMR (SIGIR) | 1998 | relevance/diversity tradeoff | Adopt | lambda 0.7, full-window prefetch |
| Weinberger et al., Feature Hashing (ICML) | 2009 | hashing trick | Adopt (fallback) | hashing-lex-v1 cold start |
| Winkler, Record Linkage metrics (ASA) | 1989 | JW string similarity | Modify | 0.93 threshold + type guard + blocking |
| Kleppmann, DDIA | 2017 | durable queues, WAL, recovery | Adopt | jobs backoff/dead-letter; SQLite WAL |
| Lewis et al., RAG (NeurIPS) | 2020 | runtime retrieval + LLM | Modify | invert: front-load knowledge; LLM optional consumer |
| Karpukhin et al., DPR (EMNLP) | 2020 | dense retrieval strengths | Reject-for-now | needs pretrained encoder; frontier item |
| Malkov & Yashunin, HNSW (TPAMI) | 2016 | layered small-world ANN graphs | **Adopt (v0.10)** | pure-Rust deterministic build: splitmix64 levels, hash-order insertion, cosine parity (`src/ann.rs`; ADR-010; crossover bench: 48x at 100k, recall@10 0.967 @ ef=128) |

Search-methodology note: arXiv/ACL/NeurIPS/SIGIR-style literature and major
open-source RAG/vector engines informed the priority ranking in section 8 of
the whitepaper; per-source adoption decisions above are the durable record.
