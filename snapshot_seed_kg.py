"""
MKGE -- Seed KG Snapshot (read-only)
============================================================
Exports the CURRENT Neo4j graph to a local JSON file so it can be
restored exactly, as many times as needed, for controlled baseline vs.
memory-gated experiments.

This script is STRICTLY READ-ONLY. It never writes, deletes, or resets
anything in Neo4j -- it only runs MATCH queries.

Why not elementId()?
---------------------
Neo4j's elementId() is only guaranteed stable for the lifetime of the
underlying node/relationship record. After a DETACH DELETE + recreate
cycle (which restore_seed_kg.py will do every time you want to reset
back to this seed), every node gets a BRAND NEW elementId. So instead
of storing elementIds, every node is assigned a stable, sequential
LOCAL snapshot id (0, 1, 2, ...) at export time, and relationships
reference each other using those local ids instead. That local id is
what makes replaying relationships correct after a full delete+recreate.

What's captured
----------------
- Every node: ALL labels (not just __Entity__) + ALL properties as-is,
  including Chunk.text and Chunk.embedding (list of floats) and any
  Document node properties.
- Every relationship: its type, its full property dict (in case any
  relationship carries properties), and the (start, end) pair expressed
  as local snapshot ids rather than elementIds.

Run with:  python snapshot_seed_kg.py
(same .env as the rest of the project -- no new env vars needed)
"""

import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

OUTPUT_FILE = "clean_seed_kg.json"


def export_nodes(driver) -> tuple:
    """
    Reads every node in the graph, regardless of label, and assigns each
    one a stable local snapshot_id based on the order Neo4j returns them
    in this single query (deterministic within this one read).

    Returns
    -------
    nodes : list[dict]
        [{"snapshot_id": int, "labels": [str, ...], "properties": {...}}, ...]
    element_id_to_snapshot_id : dict[str, int]
        Lookup used by export_relationships() below to translate this
        session's elementIds into the stable local ids -- this map is
        only ever used within this same script run; it is NOT saved to
        the JSON file itself (elementIds are session-specific and would
        be meaningless on restore).
    """
    nodes = []
    element_id_to_snapshot_id = {}

    with driver.session() as session:
        result = session.run("MATCH (n) RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props")
        for snapshot_id, record in enumerate(result):
            element_id_to_snapshot_id[record["eid"]] = snapshot_id
            nodes.append({
                "snapshot_id": snapshot_id,
                "labels": record["labels"],
                "properties": dict(record["props"]),
            })

    return nodes, element_id_to_snapshot_id


def export_relationships(driver, element_id_to_snapshot_id: dict) -> list:
    """
    Reads every relationship in the graph (any type, any direction as
    stored), translating its endpoints from elementIds to the stable
    local snapshot ids built in export_nodes() above.

    Uses a DIRECTED MATCH ()-[r]->() so the original direction of each
    relationship is preserved on restore -- undirected traversal
    (as stage2's load_entity_graph uses for graph-metric purposes) would
    lose that information, which restore_seed_kg.py needs to recreate
    the graph faithfully rather than just its undirected shape.
    """
    relationships = []

    with driver.session() as session:
        result = session.run(
            "MATCH (n)-[r]->(m) "
            "RETURN elementId(n) AS start_eid, elementId(m) AS end_eid, "
            "type(r) AS rel_type, properties(r) AS props"
        )
        for record in result:
            start_eid = record["start_eid"]
            end_eid = record["end_eid"]

            # Defensive: every node in the graph should already be in
            # this map since export_nodes() reads ALL nodes with no
            # label filter -- but guard anyway rather than silently
            # dropping a relationship or crashing with a raw KeyError.
            if start_eid not in element_id_to_snapshot_id or end_eid not in element_id_to_snapshot_id:
                print(f"    [warn] skipping relationship with unmapped endpoint: "
                      f"{start_eid} -> {end_eid} ({record['rel_type']})")
                continue

            relationships.append({
                "start_snapshot_id": element_id_to_snapshot_id[start_eid],
                "end_snapshot_id": element_id_to_snapshot_id[end_eid],
                "type": record["rel_type"],
                "properties": dict(record["props"]),
            })

    return relationships


def summarize_labels(nodes: list) -> dict:
    """Per-label node counts, for the verification printout. A node with
    multiple labels (e.g. an __Entity__ node that's also something else)
    is counted once under EACH of its labels, so these counts can sum to
    more than len(nodes) -- that's expected and worth knowing, not a bug.
    """
    counts: dict = {}
    for node in nodes:
        for label in node["labels"]:
            counts[label] = counts.get(label, 0) + 1
    return counts


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j (read-only session).\n")

    print("[Snapshot] Reading all nodes...")
    nodes, element_id_to_snapshot_id = export_nodes(driver)
    print(f"[Snapshot] Read {len(nodes)} node(s).\n")

    print("[Snapshot] Reading all relationships...")
    relationships = export_relationships(driver, element_id_to_snapshot_id)
    print(f"[Snapshot] Read {len(relationships)} relationship(s).\n")

    snapshot = {
        "format_version": 1,
        "nodes": nodes,
        "relationships": relationships,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(snapshot, f)

    file_size_bytes = os.path.getsize(OUTPUT_FILE)

    label_counts = summarize_labels(nodes)
    rel_type_counts: dict = {}
    for rel in relationships:
        rel_type_counts[rel["type"]] = rel_type_counts.get(rel["type"], 0) + 1

    chunks_with_embedding = sum(
        1 for n in nodes
        if "Chunk" in n["labels"] and n["properties"].get("embedding") is not None
    )
    chunks_total = sum(1 for n in nodes if "Chunk" in n["labels"])
    documents_total = sum(1 for n in nodes if "Document" in n["labels"])

    print("=" * 60)
    print(f"[Snapshot] Saved to '{OUTPUT_FILE}' ({file_size_bytes:,} bytes)")
    print(f"[Snapshot] Total nodes         : {len(nodes)}")
    print(f"[Snapshot] Total relationships : {len(relationships)}")
    print()
    print("[Snapshot] Node counts by label:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:<20}: {count}")
    print()
    print("[Snapshot] Relationship counts by type:")
    for rel_type, count in sorted(rel_type_counts.items()):
        print(f"    {rel_type:<20}: {count}")
    print()
    print(f"[Snapshot] Document nodes captured        : {documents_total}")
    print(f"[Snapshot] Chunk nodes captured            : {chunks_total}")
    print(f"[Snapshot] Chunk nodes WITH embedding      : {chunks_with_embedding}")
    if chunks_total > 0 and chunks_with_embedding < chunks_total:
        print(f"    [warn] {chunks_total - chunks_with_embedding} chunk(s) missing an embedding.")
    print("=" * 60)
    print("\n[Snapshot] Done. Database was NOT modified -- read-only session throughout.")

    driver.close()


if __name__ == "__main__":
    main()