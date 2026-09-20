//! Knowledge graph read-side (edges are written by `entities::link_cooccurrences`).
//!
//! v0.1 graph model (ADR-007): SQLite relational tables with a symmetric
//! CO_OCCURS_WITH edge type and weights. This is *deliberately* not a graph
//! database — relational representation covers 1-hop entity pages, "show me
//! everything about X", and entity-intersection queries, which is the
//! demonstrated demand (see docs/GRAPH.md and NON_GOALS.md).

use crate::error::Result;
use crate::storage::dao;
use rusqlite::Connection;
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
    /// Edge type.
    pub relationship: String,
    /// Weight.
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
