"""Utilities for storing and representing the knowledge graph.

This module provides helpers to:
- Serialize an ExtractionResult to JSON
- Store the KG into a Neo4j database
- Build an in-memory NetworkX graph
- A unified `persist_kg` function to optionally use all three
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional
import json

import networkx as nx
from neo4j import GraphDatabase

from grapheval.kg_construction.llm_extractor import ExtractionResult
from grapheval.models.entities import Entity, RelationTriple


# ----------------------------- JSON storage -----------------------------


def _entity_to_dict(entity: Entity) -> Dict[str, Any]:
    return {
        "text": entity.text,
        "type": entity.type,
        "id": entity.id,
    }


def _triple_to_dict(triple: RelationTriple) -> Dict[str, Any]:
    return {
        "head": triple.head.text,
        "relation": triple.relation,
        "tail": triple.tail.text,
        "confidence": triple.confidence,
    }


def save_kg_to_json(result: ExtractionResult, file_path: str) -> None:
    """Serialize the KG (entities + triples) to a JSON file."""

    data = {
        "entities": [_entity_to_dict(e) for e in result.entities],
        "triples": [_triple_to_dict(t) for t in result.triples],
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ----------------------------- NetworkX graph -----------------------------


def build_networkx_graph(result: ExtractionResult, directed: bool = True) -> nx.Graph:
    """Build an in-memory NetworkX graph from the KG.

    Nodes correspond to entities (keyed by their text), edges to relation triples.
    """

    G: nx.Graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Add nodes
    for ent in result.entities:
        node_id = ent.text
        if node_id not in G:
            G.add_node(node_id, text=ent.text, type=ent.type, id=ent.id)

    # Add edges
    for triple in result.triples:
        head_id = triple.head.text
        tail_id = triple.tail.text
        if head_id not in G:
            G.add_node(head_id, text=triple.head.text, type=triple.head.type, id=triple.head.id)
        if tail_id not in G:
            G.add_node(tail_id, text=triple.tail.text, type=triple.tail.type, id=triple.tail.id)
        G.add_edge(
            head_id,
            tail_id,
            relation=triple.relation,
            confidence=triple.confidence,
        )

    return G


# ----------------------------- Neo4j storage -----------------------------


def _neo4j_merge_entity(tx, entity: Entity) -> None:
    query = (
        "MERGE (e:Entity {text: $text}) "
        "SET e.type = coalesce($type, e.type)"
    )
    tx.run(query, text=entity.text, type=entity.type)


def _neo4j_merge_triple(tx, triple: RelationTriple) -> None:
    query = (
        "MATCH (h:Entity {text: $head_text}), (t:Entity {text: $tail_text}) "
        "MERGE (h)-[r:RELATION {name: $relation}]->(t) "
        "SET r.confidence = $confidence"
    )
    tx.run(
        query,
        head_text=triple.head.text,
        tail_text=triple.tail.text,
        relation=triple.relation,
        confidence=triple.confidence,
    )


def store_kg_in_neo4j(
    result: ExtractionResult,
    uri: str,
    user: str,
    password: str,
    database: Optional[str] = None,
) -> Dict[str, Any]:
    """Store the KG into a Neo4j database.

    This function creates/updates :Entity nodes and :RELATION relationships.
    Returns a small summary (counts) for convenience.
    """

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) if database else driver.session() as session:
            # Upsert entities
            for ent in result.entities:
                session.execute_write(_neo4j_merge_entity, ent)
            # Upsert triples
            for triple in result.triples:
                session.execute_write(_neo4j_merge_triple, triple)

    finally:
        driver.close()

    return {
        "entities_written": len(result.entities),
        "triples_written": len(result.triples),
    }


# ----------------------------- Unified persistence API -----------------------------


def persist_kg(
    result: ExtractionResult,
    json_path: Optional[str] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    neo4j_database: Optional[str] = None,
    build_networkx: bool = False,
) -> Dict[str, Any]:
    """Persist the KG to one or more backends.

    - If `json_path` is provided, write entities/triples to that JSON file.
    - If `neo4j_uri`/`neo4j_user`/`neo4j_password` are provided, store into Neo4j.
    - If `build_networkx` is True, build and return a NetworkX graph.

    Returns a dict summarizing what was done, e.g.:
    {
      "json_path": "..." or None,
      "neo4j_summary": {...} or None,
      "networkx_graph": <nx.Graph> or None,
    }
    """

    summary: Dict[str, Any] = {
        "json_path": None,
        "neo4j_summary": None,
        "networkx_graph": None,
    }

    if json_path is not None:
        save_kg_to_json(result, json_path)
        summary["json_path"] = json_path

    if neo4j_uri and neo4j_user and neo4j_password:
        neo4j_info = store_kg_in_neo4j(
            result,
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
        )
        summary["neo4j_summary"] = neo4j_info

    if build_networkx:
        G = build_networkx_graph(result)
        summary["networkx_graph"] = G

    return summary
