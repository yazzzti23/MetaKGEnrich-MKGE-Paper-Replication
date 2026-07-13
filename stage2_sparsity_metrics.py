"""
MKGE Paper Replication -- Stage 2: Sparse-Node Detection
============================================================
Takes the seed KG built in Stage 1 and computes the paper's 7
graph-theoretic sparsity metrics (Table 1):
    clique, non-clique, clustering, degree, betweenness, diameter, louvain

A node is flagged "sparse" on a metric if its score is <= the median for
that metric -- exactly as the Methodology section defines it.

Run with:  python stage2_sparsity_metrics.py
(uses the same .env as Stage 1 -- no new API keys needed for this stage)
"""

import os
from collections import defaultdict

import networkx as nx
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

METRIC_NAMES = [
    "clique", "non_clique", "clustering", "degree",
    "betweenness", "diameter", "louvain",
]


def load_entity_graph(driver) -> nx.Graph:
    """Pull only the __Entity__ nodes (real concepts: Jack, Rose, Titanic...)
    out of Neo4j into a NetworkX graph. We deliberately EXCLUDE the
    Document/Chunk/Community/Network bookkeeping nodes that neo4j-graphrag
    adds for its own internal use -- those aren't "concepts" in the
    paper's sense, and including them would distort the metrics.
    """
    G = nx.Graph()

    with driver.session() as session:
        nodes = session.run(
            "MATCH (n:__Entity__) "
            "RETURN elementId(n) AS id, coalesce(n.name, n.id, 'unnamed') AS label"
        )
        for record in nodes:
            G.add_node(record["id"], label=record["label"])

        edges = session.run(
            "MATCH (n:__Entity__)-[r]-(m:__Entity__) "
            "RETURN elementId(n) AS source, elementId(m) AS target"
        )
        for record in edges:
            G.add_edge(record["source"], record["target"])

    return G


def compute_sparsity_metrics(G: nx.Graph) -> dict:
    """The 7 metrics from Table 1 of the paper. Returns
    {node_id: {metric_name: score}}.
    """
    scores: dict = defaultdict(dict)

    # 1. Clique -- size of the largest clique each node belongs to.
    #    Low score = "over-confidence in tight clusters" never even formed.
    for node, val in nx.node_clique_number(G).items():
        scores[node]["clique"] = val

    # 2. Non-Clique (isolates) -- raw degree; degree 0 = totally isolated
    #    "unknown unknown" the paper talks about.
    for node in G.nodes():
        scores[node]["non_clique"] = G.degree(node)

    # 3. Clustering coefficient -- local triadic closure (over-specialisation
    #    signal when low).
    for node, val in nx.clustering(G).items():
        scores[node]["clustering"] = val

    # 4. Degree centrality -- "fluency illusion" signal when low.
    for node, val in nx.degree_centrality(G).items():
        scores[node]["degree"] = val

    # 5. Betweenness centrality -- "missing bridge" signal when low.
    for node, val in nx.betweenness_centrality(G).items():
        scores[node]["betweenness"] = val

    # 6. Component diameter -- each node inherits the diameter of ITS
    #    connected component (the paper treats this as a per-node signal).
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        diam = nx.diameter(subgraph) if subgraph.number_of_nodes() > 1 else 0
        for node in component:
            scores[node]["diameter"] = diam

    # 7. Louvain community size -- each node inherits the size of the
    #    community Louvain's algorithm assigned it to.
    for community in nx.community.louvain_communities(G, seed=42):
        for node in community:
            scores[node]["louvain"] = len(community)

    return scores


def flag_sparse_nodes(scores: dict):
    """Median-threshold rule from the paper: sparse if score <= median."""
    medians = {}
    for metric in METRIC_NAMES:
        values = sorted(scores[node][metric] for node in scores)
        n = len(values)
        medians[metric] = (
            values[n // 2] if n % 2 == 1
            else (values[n // 2 - 1] + values[n // 2]) / 2
        )

    sparse_flags: dict = defaultdict(dict)
    for node in scores:
        for metric in METRIC_NAMES:
            sparse_flags[node][metric] = scores[node][metric] <= medians[metric]

    return sparse_flags, medians


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    G = load_entity_graph(driver)
    print(f"[Stage 2] Loaded entity graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges.\n")

    if G.number_of_nodes() == 0:
        print("No __Entity__ nodes found -- did Stage 1 actually run first?")
        driver.close()
        return

    scores = compute_sparsity_metrics(G)
    sparse_flags, medians = flag_sparse_nodes(scores)
    labels = nx.get_node_attributes(G, "label")

    print("[Stage 2] Median thresholds per metric:")
    for metric, median in medians.items():
        print(f"  {metric:<12}: {median:.3f}")
    print()

    print("[Stage 2] Per-node scores and sparsity flags (SPARSE = at/below median):\n")
    header = f"{'Node':<20} " + " ".join(f"{m:<12}" for m in METRIC_NAMES)
    print(header)
    print("-" * len(header))
    for node in scores:
        label = str(labels.get(node, node))[:18]
        row = f"{label:<20} "
        for metric in METRIC_NAMES:
            flag = "SPARSE" if sparse_flags[node][metric] else "ok"
            row += f"{flag:<12} "
        print(row)

    driver.close()


if __name__ == "__main__":
    main()
