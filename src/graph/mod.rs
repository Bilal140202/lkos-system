//! Knowledge graph read-side + traversal.
//!
//! Edge model (ADR-007, revised for v0.9):
//! - `CO_OCCURS_WITH` — symmetric, weight = co-mention chunk count.
//! - `RELATES_TO_<predicate>` — typed directed edges derived from claims
//!   whose subject and object both resolve to entities (e.g.
//!   `RELATES_TO_acquired`). Weight = supporting claim count.
//!
//! Reads: one-hop neighborhoods (unchanged), **multi-hop BFS traversal**
//! (depth ≤ 2 by default; every returned edge carries its provenance weight),
//! and degree centrality for entity ranking. SQLite relational tables remain
//! the substrate — see ADR-007 for why not a graph database.

use crate::error::Result;
use crate::storage::dao;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

/// A node in a neighborhood view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Entity id.
    pub id: i64,
    /// Display name.
    pub name: String,
    /// Entity type.
    pub entity_type: String,
}

/// An edge in a neighborhood view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source entity id.
    pub source: i64,
    /// Target entity id.
    pub target: i64,
    /// Edge type (`CO_OCCURS_WITH` or `RELATES_TO_<predicate>`).
    pub relationship: String,
    /// Weight (evidence count).
    pub weight: f32,
}

/// A one-hop neighborhood.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighborhood {
    /// Center entity.
    pub center: GraphNode,
    /// Neighbors with connecting edges.
    pub neighbors: Vec<(GraphNode, GraphEdge)>,
    /// Documents mentioning the center entity.
    pub documents: Vec<i64>,
}

/// Result of a multi-hop traversal: entities discovered per depth level with
/// the connecting edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Traversal {
    /// Starting entity.
    pub start: GraphNode,
    /// Level 1 = direct neighbors, level 2 = neighbors-of-neighbors, ...
    pub levels: Vec<Vec<(GraphNode, GraphEdge)>>,
    /// Distinct entities visited (including the start).
    pub visited: usize,
}

/// Build the one-hop neighborhood view of an entity.
pub fn neighborhood(conn: &Connection, entity_id: i64, limit: usize) -> Result<Neighborhood> {
    let entity = dao::get_entity(conn, entity_id)?;
    let center = GraphNode {
        id: entity.id,
        name: entity.display_name.clone(),
        entity_type: entity.entity_type.clone(),
    };
    let neighbors_raw = dao::neighbors(conn, entity_id, limit)?;
    let mut neighbors = Vec::new();
    for (summary, rel) in neighbors_raw {
        neighbors.push((
            GraphNode {
                id: summary.id,
                name: summary.display_name,
                entity_type: summary.entity_type,
            },
            GraphEdge {
                source: rel.source_entity_id,
                target: rel.target_entity_id,
                relationship: rel.relationship_type,
                weight: rel.weight,
            },
        ));
    }
    let documents = dao::documents_for_entity(conn, entity_id)?;
    Ok(Neighborhood {
        center,
        neighbors,
        documents,
    })
}

/// Breadth-first traversal up to `depth` hops (1..=3; deeper levels are
/// exponentially noisy for co-occurrence graphs). Deduplicates visited ids
/// and caps each level at `limit` nodes ordered by edge weight.
pub fn traverse(
    conn: &Connection,
    entity_id: i64,
    depth: usize,
    limit: usize,
) -> Result<Traversal> {
    let depth = depth.clamp(1, 3);
    let start_rec = dao::get_entity(conn, entity_id)?;
    let start = GraphNode {
        id: start_rec.id,
        name: start_rec.display_name.clone(),
        entity_type: start_rec.entity_type.clone(),
    };
    let mut visited: std::collections::HashSet<i64> = std::collections::HashSet::new();
    visited.insert(entity_id);
    let mut levels = Vec::new();
    let mut frontier = vec![entity_id];
    for _ in 0..depth {
        let mut next_level: Vec<(GraphNode, GraphEdge)> = Vec::new();
        let mut next_frontier: Vec<i64> = Vec::new();
        for node in &frontier {
            for (summary, rel) in dao::neighbors(conn, *node, limit)? {
                let other = if rel.source_entity_id == *node {
                    rel.target_entity_id
                } else {
                    rel.source_entity_id
                };
                if visited.contains(&other) {
                    continue;
                }
                if visited.insert(other) {
                    next_frontier.push(other);
                    next_level.push((
                        GraphNode {
                            id: summary.id,
                            name: summary.display_name.clone(),
                            entity_type: summary.entity_type.clone(),
                        },
                        GraphEdge {
                            source: rel.source_entity_id,
                            target: rel.target_entity_id,
                            relationship: rel.relationship_type,
                            weight: rel.weight,
                        },
                    ));
                }
            }
        }
        if next_level.is_empty() {
            break;
        }
        next_frontier.sort_unstable();
        frontier = next_frontier;
        levels.push(next_level);
    }
    Ok(Traversal {
        start,
        visited: visited.len(),
        levels,
    })
}

/// Degree centrality for one entity (sum of edge weights).
pub fn degree_centrality(conn: &Connection, entity_id: i64) -> Result<f32> {
    let w: Option<f64> = conn
        .query_row(
            "SELECT SUM(weight) FROM relationships
             WHERE source_entity_id = ?1 OR target_entity_id = ?1",
            rusqlite::params![entity_id],
            |r| r.get(0),
        )
        .optional()?
        .flatten();
    Ok(w.unwrap_or(0.0) as f32)
}
