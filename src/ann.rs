//! Deterministic in-memory HNSW approximate nearest-neighbour index.
//!
//! Addresses the v0.9 dense-channel ceiling: brute-force cosine scan refused
//! beyond `max_dense_scan` (250k chunks) instead of degrading (whitepaper
//! 7.3, GitHub issue #2). This module implements a pure-Rust Hierarchical
//! Navigable Small World graph (Malkov & Yashunin, 2016) with the following
//! engineering contract:
//!
//! - **Deterministic.** No RNG anywhere: layer assignment derives from a
//!   fixed splitmix64 hash of the chunk id (the same trick hnswlib uses,
//!   replacing its PRNG), nodes are inserted in ascending id order, and
//!   every heap/sort tie-breaks by distance then id. Two builds over the
//!   same (id, vector) pairs produce bit-identical graphs and identical
//!   query outputs on any machine.
//! - **Cosine distance.** Distance is `1 - cos(q, v)` — the same similarity
//!   the brute-force channel reports, so RRF fusion semantics are unchanged.
//!   Stored vector norms are precomputed once; the query norm once per search.
//! - **Positive-similarity parity.** The brute-force channel drops
//!   `cosine <= 0` results; the ANN search applies the identical filter, so
//!   channel output shape (possibly fewer than `k` hits) matches.
//! - **Immutable per model.** Chunk vectors are written at insert or at
//!   model-change migration only (engine invariant), so a cache keyed by
//!   `(model, COUNT, MAX(chunk id))` is exact: inserts raise COUNT/MAX,
//!   deletes lower COUNT, migrations change the model's membership. See
//!   `docs/adr/ADR.md` (ADR-010) and the invalidation regression tests.
//! - **Rebuild-on-mutation.** Deletions and re-embeddings invalidate the
//!   cache; the next dense query rebuilds (measured O(N log N), see
//!   `benchmarks/results/ann-crossover-*.txt`). Local desktop workloads
//!   mutate rarely and read often, so amortized rebuild cost wins over
//!   per-mutation graph surgery; the ADR records this decision.
//!
//! Parameters: `M` (max upper-layer degree), `M0 = 2M` (layer-0 degree),
//! `ef_construction`, `ef_search`. Defaults follow the reference paper.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Fixed seed for level-assignment hashing (splitmix64 finalizer input).
const LEVEL_SEED: u64 = 0x9E_37_79_B9_7F_4A_7C_15;

/// Deterministic splitmix64 — no RNG state, same id -> same level forever.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E_37_79_B9_7F_4A_7C_15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF_58_47_6D_1C_E4_E5_B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94_D0_49_BB_13_31_11_EB);
    x ^ (x >> 31)
}

/// Exponential level for a node: `floor(-ln(u01) * mL)` with u01 in (0,1)
/// derived from 53 hash bits. Identical to hnswlib's distribution.
fn level_for(id: i64, ml: f64) -> u32 {
    let h = splitmix64((id as u64) ^ LEVEL_SEED);
    let u01 = ((h >> 11) as f64 + 0.5) / ((1u64 << 53) as f64);
    (-u01.ln() * ml).floor() as u32
}

/// Precomputed-unit-norm helper: distance via dot / (nq * nv).
#[inline]
fn dist(qv: &[f32], qnorm: f32, vv: &[f32], vnorm: f32) -> f32 {
    if qv.len() != vv.len() || qnorm < f32::EPSILON || vnorm < f32::EPSILON {
        // Zero-norm or dimension mismatch => cosine 0.0 => distance 1.0.
        return 1.0;
    }
    let dot: f32 = qv.iter().zip(vv.iter()).map(|(a, b)| a * b).sum();
    // Clamp at 0: float rounding can push sim epsilon-past 1.0 for identical
    // vectors, and negative distances would break the to_bits heap ordering.
    (1.0 - dot / (qnorm * vnorm)).max(0.0)
}

#[inline]
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Max-heap entry for the kept top-`ef` results. `BinaryHeap` is a max-heap,
/// so `Ord` is FORWARD here: `peek()` = worst kept (for stop/accept gates)
/// and `pop()` = evict worst (on overflow). Deterministic: distance first,
/// then node id. Distances are non-negative (1 - cos), so `to_bits` is
/// monotonic.
#[derive(Clone, PartialEq)]
struct Best {
    dist: f32,
    node: u32,
}
impl Eq for Best {}
impl Ord for Best {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .to_bits()
            .cmp(&other.dist.to_bits())
            .then_with(|| self.node.cmp(&other.node))
    }
}
impl PartialOrd for Best {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Min-heap entry for the candidate frontier. `BinaryHeap` is a max-heap,
/// so `Ord` is REVERSED here: `pop()` yields the smallest (dist, id) first.
#[derive(PartialEq)]
struct Cand {
    dist: f32,
    node: u32,
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .to_bits()
            .cmp(&self.dist.to_bits())
            .then_with(|| other.node.cmp(&self.node))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A built, immutable HNSW graph over one embedding model's chunks.
pub struct HnswIndex {
    /// Chunk ids, ascending; node index into these vectors.
    ids: Vec<i64>,
    vectors: Vec<Vec<f32>>,
    norms: Vec<f32>,
    levels: Vec<u32>,
    /// neighbors[node][layer] -> linked nodes (layer 0 has M0 capacity).
    neighbors: Vec<Vec<Vec<u32>>>,
    entry: Option<u32>,
    max_level: u32,
    m: usize,
    m0: usize,
    ef_construction: usize,
}

impl HnswIndex {
    /// Build a deterministic index from (chunk id, vector) pairs.
    ///
    /// Node layout is ascending by id; insertion order is a deterministic
    /// hash shuffle (see below). Duplicate ids keep the first occurrence;
    /// zero-norm vectors are kept (they can never match a query: distance
    /// 1.0, filtered out by the positive-similarity rule).
    pub fn build(mut pairs: Vec<(i64, Vec<f32>)>, m: usize, ef_construction: usize) -> HnswIndex {
        pairs.sort_by_key(|a| a.0);
        pairs.dedup_by(|a, b| a.0 == b.0);
        let n = pairs.len();
        let m = m.max(2);
        let m0 = m * 2;
        let ml = 1.0 / (m as f64).ln();

        let mut idx = HnswIndex {
            ids: Vec::with_capacity(n),
            vectors: Vec::with_capacity(n),
            norms: Vec::with_capacity(n),
            levels: Vec::with_capacity(n),
            neighbors: Vec::with_capacity(n),
            entry: None,
            max_level: 0,
            m,
            m0,
            ef_construction: ef_construction.max(m0).max(16),
        };
        for (id, v) in pairs {
            let lv = if n == 1 { 0 } else { level_for(id, ml) };
            idx.ids.push(id);
            idx.norms.push(norm(&v));
            idx.levels.push(lv);
            idx.neighbors.push(vec![Vec::new(); lv as usize + 1]);
            idx.vectors.push(v);
        }
        // Insertion order: deterministic pseudo-random (splitmix64 of the
        // node's own id). Reference HNSW inserts in random order — with
        // cluster-sequential corpora, ascending-id insertion builds isolated
        // per-cluster subgraphs and starves cross-cluster bridges, which
        // collapses recall at scale (measured: ef-independent recall
        // saturation; see ADR-010). A hash-order shuffle interleaves clusters
        // from the first inserts while remaining fully deterministic.
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&node| splitmix64((idx.ids[node as usize] as u64) ^ LEVEL_SEED));
        for node in order {
            idx.insert_node(node);
        }
        idx
    }

    /// Insert a node (called in deterministic hash order from `build`).
    fn insert_node(&mut self, node: u32) {
        let level = self.levels[node as usize];
        let Some(ep) = self.entry else {
            self.entry = Some(node);
            self.max_level = level;
            return;
        };

        // Greedy descent from the top to level+1 (ef = 1).
        // (Cloned: the borrow must end before `link` mutates the graph.)
        let qv = self.vectors[node as usize].clone();
        let qn = self.norms[node as usize];
        let mut ep = ep;
        for lc in (level + 1..=self.max_level).rev() {
            ep = self.greedy_step(ep, &qv, qn, lc);
        }

        // Beam search + link on each shared layer, top-down.
        let mut eps = vec![ep];
        for lc in (0..=level.min(self.max_level)).rev() {
            let cands = self.search_layer(&qv, qn, &eps, self.ef_construction, lc);
            let selected = self.select_neighbors(&cands, self.m, true);
            for &c in &selected {
                self.link(c, node, lc);
                self.link(node, c, lc);
            }
            eps = cands.into_iter().map(|c| c.node).collect();
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(node);
        }
    }

    /// One ef=1 greedy move at a layer (returns the local minimum).
    fn greedy_step(&self, mut ep: u32, qv: &[f32], qn: f32, layer: u32) -> u32 {
        let mut best = dist(qv, qn, &self.vectors[ep as usize], self.norms[ep as usize]);
        loop {
            let mut improved = false;
            let links = self.neighbors[ep as usize]
                .get(layer as usize)
                .cloned()
                .unwrap_or_default();
            for nb in links {
                let d = dist(qv, qn, &self.vectors[nb as usize], self.norms[nb as usize]);
                if d < best {
                    best = d;
                    ep = nb;
                    improved = true;
                }
            }
            if !improved {
                return ep;
            }
        }
    }

    /// ef-bounded beam search at one layer. Returns up to `ef` best
    /// (deterministic order via (dist, id) heap ordering).
    fn search_layer(&self, qv: &[f32], qn: f32, eps: &[u32], ef: usize, layer: u32) -> Vec<Best> {
        let ef = ef.max(1);
        let mut visited: HashSet<u32> = HashSet::new();
        let mut cand: BinaryHeap<Cand> = BinaryHeap::new(); // min-heap (reverse Ord)
        let mut best: BinaryHeap<Best> = BinaryHeap::new(); // max-heap of current top-ef

        for &ep in eps {
            if visited.insert(ep) {
                let d = dist(qv, qn, &self.vectors[ep as usize], self.norms[ep as usize]);
                cand.push(Cand { dist: d, node: ep });
                best.push(Best { dist: d, node: ep });
            }
        }
        while best.len() > ef {
            best.pop();
        }

        while let Some(c) = cand.pop() {
            // Stop when the frontier's closest is worse than the worst kept.
            if best.len() >= ef && c.dist > best.peek().map(|b| b.dist).unwrap_or(f32::INFINITY) {
                break;
            }
            let links = self.neighbors[c.node as usize]
                .get(layer as usize)
                .cloned()
                .unwrap_or_default();
            for nb in links {
                if visited.insert(nb) {
                    let d = dist(qv, qn, &self.vectors[nb as usize], self.norms[nb as usize]);
                    if best.len() < ef || d < best.peek().map(|b| b.dist).unwrap_or(f32::INFINITY) {
                        cand.push(Cand { dist: d, node: nb });
                        best.push(Best { dist: d, node: nb });
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
            }
        }

        // Deterministic output: sort by (dist, node) ascending.
        let mut out: Vec<Best> = best.into_iter().collect();
        out.sort_by(|a, b| {
            a.dist
                .to_bits()
                .cmp(&b.dist.to_bits())
                .then_with(|| a.node.cmp(&b.node))
        });
        out
    }

    /// Malkov heuristic neighbour selection. `backfill` controls the paper's
    /// keepPrunedConnections: at insert time true (paper Algorithm 4), at
    /// shrink time false (hnswlib behaviour — pruning without re-adding
    /// preserves the diversity that keeps the graph navigable).
    fn select_neighbors(&self, cands: &[Best], m: usize, backfill: bool) -> Vec<u32> {
        if cands.len() <= m {
            return cands.iter().map(|c| c.node).collect();
        }
        let mut selected: Vec<Best> = Vec::with_capacity(m);
        let mut pruned: Vec<Best> = Vec::new();
        for c in cands {
            if selected.len() >= m {
                pruned.push(c.clone());
                continue;
            }
            let d_q = c.dist;
            let mut good = true;
            // The candidate must be closer to the query than to any
            // already-selected neighbour (evaluated on this layer's metric).
            for s in &selected {
                let d_se = dist(
                    &self.vectors[s.node as usize],
                    self.norms[s.node as usize],
                    &self.vectors[c.node as usize],
                    self.norms[c.node as usize],
                );
                if d_se < d_q {
                    good = false;
                    break;
                }
            }
            if good {
                selected.push(c.clone());
            } else {
                pruned.push(c.clone());
            }
        }
        // keepPruned: fill remaining slots with the closest pruned candidates.
        if backfill {
            for c in pruned {
                if selected.len() >= m {
                    break;
                }
                selected.push(c);
            }
        }
        selected.into_iter().map(|b| b.node).collect()
    }

    /// Add a bidirectional link, shrinking the target's list via the
    /// heuristic when it exceeds its layer capacity.
    fn link(&mut self, from: u32, to: u32, layer: u32) {
        let cap = if layer == 0 { self.m0 } else { self.m };
        let list = &mut self.neighbors[from as usize][layer as usize];
        if list.contains(&to) {
            return;
        }
        list.push(to);
        if list.len() <= cap {
            return;
        }
        // Re-select among the node's own links w.r.t. the node itself.
        let fv = self.vectors[from as usize].clone();
        let fn_ = self.norms[from as usize];
        let mut scored: Vec<Best> = list
            .iter()
            .map(|&nb| Best {
                dist: dist(
                    &fv,
                    fn_,
                    &self.vectors[nb as usize],
                    self.norms[nb as usize],
                ),
                node: nb,
            })
            .collect();
        scored.sort_by(|a, b| {
            a.dist
                .to_bits()
                .cmp(&b.dist.to_bits())
                .then_with(|| a.node.cmp(&b.node))
        });
        let chosen = self.select_around(&scored, cap);
        self.neighbors[from as usize][layer as usize] = chosen;
    }

    /// select_neighbors variant scored against the linking node itself:
    /// candidates are compared on dist(candidate, linking-node) instead of
    /// dist(candidate, global query). Shrink path: pure prune, no backfill
    /// (hnswlib behaviour — keeps the link set diverse).
    fn select_around(&self, cands: &[Best], m: usize) -> Vec<u32> {
        let mut selected: Vec<Best> = Vec::with_capacity(m);
        for c in cands {
            if selected.len() >= m {
                break;
            }
            let mut good = true;
            for s in &selected {
                let d_se = dist(
                    &self.vectors[s.node as usize],
                    self.norms[s.node as usize],
                    &self.vectors[c.node as usize],
                    self.norms[c.node as usize],
                );
                if d_se < c.dist {
                    good = false;
                    break;
                }
            }
            if good {
                selected.push(c.clone());
            }
        }
        selected.into_iter().map(|b| b.node).collect()
    }

    /// ANN search: top `k` (chunk id, cosine similarity), similarity > 0,
    /// sorted similarity-desc then id-asc — the brute-force channel's exact
    /// output contract.
    pub fn search(&self, qv: &[f32], k: usize, ef: usize) -> Vec<(i64, f32)> {
        if k == 0 || self.ids.is_empty() || self.entry.is_none() {
            return Vec::new();
        }
        let ef = ef.max(k).max(16);
        let qn = norm(qv);
        let mut ep = self.entry.unwrap();
        for lc in (1..=self.max_level).rev() {
            ep = self.greedy_step(ep, qv, qn, lc);
        }
        let eps = [ep];
        let cands = self.search_layer(qv, qn, &eps, ef, 0);
        let mut scored: Vec<(i64, f32)> = cands
            .into_iter()
            .map(|c| {
                let sim = 1.0 - c.dist;
                (self.ids[c.node as usize], sim)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the index holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Structural stats (for tests, bench, and the ADR evidence table).
    pub fn stats(&self) -> AnnStats {
        let edges: usize = self
            .neighbors
            .iter()
            .map(|layers| layers.iter().map(|l| l.len()).sum::<usize>())
            .sum();
        AnnStats {
            nodes: self.ids.len(),
            undirected_edges: edges / 2,
            max_level: self.max_level,
            levels_histogram: {
                let mut h: HashMap<u32, usize> = HashMap::new();
                for l in &self.levels {
                    *h.entry(*l).or_insert(0) += 1;
                }
                h
            },
            memory_estimate_bytes: self.vectors.len()
                * self.vectors.first().map_or(0, |v| v.len() * 4)
                + self.norms.len() * 4
                + edges * 4
                + self.ids.len() * 8,
        }
    }
}

/// Structural summary of a built index.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AnnStats {
    /// Number of indexed nodes.
    pub nodes: usize,
    /// Undirected edge count across all layers.
    pub undirected_edges: usize,
    /// Highest layer with at least one node.
    pub max_level: u32,
    /// Layer -> node count (should follow the exponential ml distribution).
    pub levels_histogram: HashMap<u32, usize>,
    /// Estimated heap footprint: vectors + norms + edges + id table.
    pub memory_estimate_bytes: usize,
}

/// Cache entry stored on the engine: the index plus the fingerprint that
/// produced it. Exactness argument (engine invariant): within one model
/// name, chunk vectors are written only at insert time or at model-change
/// migration, so `(COUNT, MAX(id))` fully determines the vector set —
/// inserts/deletes/migrations all move one of the two counters.
#[derive(Clone)]
pub struct AnnCacheEntry {
    /// Embedding model the index was built over.
    pub model: String,
    /// Fingerprint component: chunk count for `model` at build time.
    pub count: i64,
    /// Fingerprint component: max chunk id at build time.
    pub max_id: i64,
    /// The built graph.
    pub index: std::sync::Arc<HnswIndex>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clustered synthetic corpus: `clusters` unit centers, `per` points
    /// each with seeded jitter. Deterministic. Returns (pairs, centers).
    #[allow(clippy::type_complexity)]
    fn synthetic(
        clusters: usize,
        per: usize,
        dim: usize,
        jitter: f32,
    ) -> (Vec<(i64, Vec<f32>)>, Vec<Vec<f32>>) {
        let mut out = Vec::with_capacity(clusters * per);
        let mut seed: u64 = 42;
        let mut rnd = move || {
            seed = splitmix64(seed);
            ((seed >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        let mut centers = Vec::with_capacity(clusters);
        let mut id = 0i64;
        for _ in 0..clusters {
            let center: Vec<f32> = (0..dim).map(|_| rnd()).collect();
            for _ in 0..per {
                let v: Vec<f32> = center.iter().map(|&x| x + jitter * rnd()).collect();
                out.push((id, v));
                id += 1;
            }
            centers.push(center);
        }
        (out, centers)
    }

    fn brute_top10(pairs: &[(i64, Vec<f32>)], q: &[f32], k: usize) -> Vec<i64> {
        let mut s: Vec<(i64, f32)> = pairs
            .iter()
            .map(|(id, v)| (*id, 1.0 - dist(q, norm(q), v, norm(v))))
            .filter(|(_, x)| *x > 0.0)
            .collect();
        s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
        s.truncate(k);
        s.into_iter().map(|(id, _)| id).collect()
    }

    #[test]
    fn build_is_deterministic_and_recall_is_high() {
        // 2000 nodes, 32-dim, 40 tight clusters. Queries jitter the ACTUAL
        // cluster centers (semantic queries land near data; far-field
        // queries on a collapsed manifold are not representative).
        let (pairs, centers) = synthetic(40, 50, 32, 0.05);
        let mut seed: u64 = 7;
        let mut rnd = move || {
            seed = splitmix64(seed);
            ((seed >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        let queries: Vec<Vec<f32>> = centers
            .iter()
            .cycle()
            .take(50)
            .map(|c| c.iter().map(|&x| x + 0.1 * rnd()).collect())
            .collect();

        let i1 = HnswIndex::build(pairs.clone(), 16, 128);
        let i2 = HnswIndex::build(pairs.clone(), 16, 128);
        let mut recall_sum = 0.0f64;
        for q in &queries {
            let r1 = i1.search(q, 10, 64);
            let r2 = i2.search(q, 10, 64);
            assert_eq!(r1, r2, "two builds must return identical results");
            let truth: std::collections::HashSet<i64> =
                brute_top10(&pairs, q, 10).into_iter().collect();
            let got: std::collections::HashSet<i64> = r1.iter().map(|(id, _)| *id).collect();
            recall_sum += truth.intersection(&got).count() as f64 / truth.len().max(1) as f64;
        }
        let recall = recall_sum / queries.len() as f64;
        assert!(
            recall >= 0.95,
            "mean recall@10 at ef=64 must be >= 0.95 on clustered data, got {recall:.3}"
        );
    }

    #[test]
    fn positive_cosine_filter_matches_brute_force_semantics() {
        // All corpus vectors orthogonal-ish to the query (dot ~ 0 after
        // sign cancellation => similarity not > 0 in the top results case).
        let pairs: Vec<(i64, Vec<f32>)> = (0..50)
            .map(|i| {
                let mut v = vec![0.0f32; 8];
                v[i % 8] = 1.0;
                v[(i + 4) % 8] = -1.0;
                (i as i64, v)
            })
            .collect();
        let idx = HnswIndex::build(pairs.clone(), 8, 64);
        let q = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ann = idx.search(&q, 10, 64);
        let brute = brute_top10(&pairs, &q, 10);
        // Every ANN hit must be a true positive-cosine neighbour.
        for id in &brute {
            let v = &pairs[*id as usize].1;
            let sim = 1.0 - dist(&q, norm(&q), v, norm(v));
            assert!(sim > 0.0);
        }
        // With ef >= N the ANN result must equal the brute-force ranking.
        let ann_full = idx.search(&q, 10, 128);
        assert_eq!(
            ann_full.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            brute,
            "ef >= N must reproduce exact ranking"
        );
        let _ = ann;
    }

    #[test]
    fn empty_and_single_node_edges() {
        let empty = HnswIndex::build(Vec::new(), 8, 32);
        assert!(empty.is_empty());
        assert_eq!(empty.search(&[1.0, 2.0], 5, 32), Vec::new());

        let one = HnswIndex::build(vec![(123, vec![1.0, 0.0])], 8, 32);
        assert_eq!(one.search(&[1.0, 0.0], 5, 32), vec![(123, 1.0)]);
        assert_eq!(one.search(&[-1.0, 0.0], 5, 32), Vec::new()); // sim = -1
        assert_eq!(one.search(&[0.0, 1.0], 5, 32), Vec::new()); // sim = 0
    }

    #[test]
    fn duplicate_ids_are_deduplicated() {
        let idx = HnswIndex::build(
            vec![
                (1, vec![1.0, 0.0]),
                (1, vec![0.0, 1.0]),
                (2, vec![1.0, 0.0]),
            ],
            4,
            32,
        );
        assert_eq!(idx.len(), 2);
    }
}
