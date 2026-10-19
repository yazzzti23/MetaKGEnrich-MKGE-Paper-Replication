"""
MKGE -- Seed KG Restore (from clean_seed_kg.json)
============================================================
Restores the CURRENT Neo4j database to the exact frozen state captured
by snapshot_seed_kg.py. Intended to be run once per experimental
condition (baseline run, then memory-gated run, etc.) so both start
from an identical graph.

RESTORE SEQUENCE
-----------------
1. Load clean_seed_kg.json (read-only -- this script never writes to it).
2. Reset the graph completely, reusing reset_graph.py's own
   delete_all_indexes() / clear_graph() functions (imported, not
   duplicated or modified) so the reset step matches the rest of this
   project exactly.
3. Recreate every node with its original labels + properties, plus a
   TEMPORARY `_snapshot_id` property (the value assigned at export
   time) -- needed because Neo4j gives every recreated node a brand
   new elementId, so `_snapshot_id` is the only reliable way to wire
   relationships back up correctly in the next step.
4. Recreate every relationship by matching its two endpoints on
   `_snapshot_id`, with the exact original type, direction, and
   properties from the snapshot.
5. Remove `_snapshot_id` from every node -- it was only ever a
   restore-time scaffold, not part of the real seed KG, so the restored
   graph ends up byte-for-byte equivalent to the original in terms of
   labels/properties (nothing extra left behind).
6. Verify: print total node/relationship counts and per-label counts,
   and compare them against the snapshot file's own recorded counts.

Run with:  python restore_seed_kg.py
(same .env and same folder as the rest of the project -- must be run
alongside reset_graph.py, which this script imports from)
"""

import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

from reset_graph import delete_all_indexes, clear_graph

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

SNAPSHOT_FILE = "clean_seed_kg.json"


def load_snapshot(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def recreate_nodes(driver, nodes: list) -> None:
    """
    Recreates every node from the snapshot, one CREATE per node.
    Each node's original labels are baked directly into the Cypher
    label syntax (labels aren't parameterizable in Cypher), and its
    original properties are passed as a parameterized map so property
    VALUES are never string-interpolated. A temporary `_snapshot_id`
    property is set alongside the original properties -- this is the
    join key recreate_relationships() below uses, and it's removed by
    strip_snapshot_ids() once relationships are wired up.
    """
    with driver.session() as session:
        for node in nodes:
            labels = node["labels"]
            label_clause = "".join(f":`{label}`" for label in labels)
            session.run(
                f"CREATE (n{label_clause}) "
                f"SET n = $props, n._snapshot_id = $snapshot_id",
                props=node["properties"],
                snapshot_id=node["snapshot_id"],
            )
    print(f"[Restore] Recreated {len(nodes)} node(s).")


def recreate_relationships(driver, relationships: list) -> int:
    """
    Recreates every relationship by matching its two endpoints on the
    temporary `_snapshot_id` property set in recreate_nodes() above.
    The relationship TYPE is baked into the Cypher (not parameterizable),
    same reasoning as node labels; its properties are passed as a
    parameterized map, same as node properties.

    Returns the number of relationships successfully created, so the
    caller can compare it against len(relationships) and flag any
    endpoint that failed to match (which would indicate a corrupted or
    hand-edited snapshot file, not normal operation).
    """
    created = 0
    with driver.session() as session:
        for rel in relationships:
            rel_type = rel["type"]
            result = session.run(
                f"MATCH (a {{_snapshot_id: $start_id}}), (b {{_snapshot_id: $end_id}}) "
                f"CREATE (a)-[r:`{rel_type}`]->(b) "
                f"SET r = $props "
                f"RETURN r",
                start_id=rel["start_snapshot_id"],
                end_id=rel["end_snapshot_id"],
                props=rel["properties"],
            )
            if result.single() is not None:
                created += 1
            else:
                print(f"    [warn] could not create relationship {rel['start_snapshot_id']} "
                      f"-[{rel_type}]-> {rel['end_snapshot_id']} -- endpoint(s) not found")
    print(f"[Restore] Recreated {created} of {len(relationships)} relationship(s).")
    return created


def strip_snapshot_ids(driver) -> None:
    """Removes the temporary `_snapshot_id` property from every node --
    it was only ever restore-time scaffolding, not part of the real
    seed KG, so the restored graph doesn't carry any extra property
    that the original snapshot didn't have.
    """
    with driver.session() as session:
        session.run("MATCH (n) WHERE n._snapshot_id IS NOT NULL REMOVE n._snapshot_id")
    print("[Restore] Removed temporary _snapshot_id property from all nodes.")


def verify_restore(driver, expected_node_count: int, expected_rel_count: int) -> None:
    """Prints total node/relationship counts and per-label counts, and
    compares the totals against the snapshot file's own recorded counts
    so any mismatch is immediately visible rather than silently assumed
    correct.
    """
    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        label_rows = session.run(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c ORDER BY label"
        ).data()

    print()
    print("=" * 60)
    print(f"[Restore] Total nodes         : {node_count}  (expected {expected_node_count})")
    print(f"[Restore] Total relationships : {rel_count}  (expected {expected_rel_count})")
    print()
    print("[Restore] Node counts by label:")
    for row in label_rows:
        print(f"    {row['label']:<20}: {row['c']}")

    if node_count != expected_node_count or rel_count != expected_rel_count:
        print()
        print("    [warn] restored counts do NOT match the snapshot file -- "
              "investigate before running any experiment against this graph.")
    else:
        print()
        print("[Restore] Counts match the snapshot exactly.")
    print("=" * 60)


def main():
    snapshot = load_snapshot(SNAPSHOT_FILE)
    nodes = snapshot["nodes"]
    relationships = snapshot["relationships"]
    print(f"[Restore] Loaded '{SNAPSHOT_FILE}': {len(nodes)} node(s), "
          f"{len(relationships)} relationship(s) on file.\n")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    print("[Restore] Resetting current graph...")
    delete_all_indexes(driver)
    clear_graph(driver)
    print()

    print("[Restore] Recreating nodes...")
    recreate_nodes(driver, nodes)

    print("[Restore] Recreating relationships...")
    recreate_relationships(driver, relationships)

    print("[Restore] Cleaning up temporary snapshot ids...")
    strip_snapshot_ids(driver)

    verify_restore(driver, expected_node_count=len(nodes), expected_rel_count=len(relationships))

    driver.close()


if __name__ == "__main__":
    main()