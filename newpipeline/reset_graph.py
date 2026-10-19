"""
MKGE Paper Replication -- Utility: Reset Between Questions
============================================================
Run this BEFORE switching to a new test question. Without it, Stage 1's
new seed KG gets built on top of whatever the previous question already
left in Neo4j -- Stage 2's metrics would then be computed over a mixed,
unrelated graph.

Mirrors the paper's own Listing 1, which calls
`delete_all_indexes(driver); _clear_graph(driver)` at the start of every
question in their loop.

Run with:  python reset_graph.py
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

CACHE_FILE = "answers_cache.json"  # from stage5_evaluation.py


def delete_all_indexes(driver) -> None:
    with driver.session() as session:
        indexes = session.run("SHOW INDEXES YIELD name").data()
        for record in indexes:
            name = record["name"]
            try:
                session.run(f"DROP INDEX `{name}` IF EXISTS")
            except Exception as e:
                print(f"  (skipped dropping index {name}: {e})")
    print(f"[Reset] Dropped {len(indexes)} index(es).")


def clear_graph(driver) -> None:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("[Reset] Deleted all nodes and relationships.")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    delete_all_indexes(driver)
    clear_graph(driver)

    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"[Reset] Removed stale {CACHE_FILE} (old before/after answers).")

    print("\n[Reset] Done. Graph is clean -- ready for a new question.")
    driver.close()


if __name__ == "__main__":
    main()