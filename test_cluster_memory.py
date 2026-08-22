"""
Quick manual test for Phase 1 + Phase 2 -- NOT part of the pipeline itself.

Loads the current graph from Neo4j, computes the node->cluster mapping
(Phase 1, stage2_sparsity_metrics.py), builds Cluster Memory from it
(Phase 2, cluster_memory.py), and prints what came out so you can eyeball
whether the clusters and Representative Memory look sensible.

Run with:  python test_cluster_memory.py
(same .env as the rest of the pipeline -- no new API keys needed unless
you pass an llm_client below)
"""

from neo4j import GraphDatabase

from stage2_sparsity_metrics import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    load_entity_graph,
    get_node_cluster_mapping,
)
from cluster_memory import build_cluster_memory_store


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    G = load_entity_graph(driver)
    print(f"[Test] Loaded entity graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges.\n")

    node_cluster_map = get_node_cluster_mapping(G)

    # No llm_client passed -> uses the free deterministic fallback for
    # topic_summary/keywords, no Groq API calls made by this test.
    store = build_cluster_memory_store(driver, node_cluster_map, G)

    print(f"[Test] Found {len(store)} cluster(s).\n")

    for cluster_id in sorted(store.all_cluster_ids()):
        cm = store.get(cluster_id)
        print(f"--- Cluster {cluster_id} ---")
        print("  entities  :", cm.representative.representative_entities)
        print("  relations :", cm.representative.representative_relations)
        print("  keywords  :", cm.representative.representative_keywords)
        print("  summary   :", cm.representative.topic_summary)
        print("  accepted exemplars:", len(cm.exemplars.accepted))
        print("  rejected exemplars:", len(cm.exemplars.rejected))
        print()

    driver.close()


if __name__ == "__main__":
    main()