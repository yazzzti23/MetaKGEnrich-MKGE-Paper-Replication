"""
MKGE -- OUR PIPELINE -- Stage 2: Sparse-Node Detection
============================================================
Our improved version of Stage 2. The original, untouched version stays
in the original pipeline folder and is what run A (original MKGE) uses.

Computes the paper's 7 graph-theoretic sparsity metrics (Table 1):
    clique, non-clique, clustering, degree, betweenness, diameter, louvain
A node is flagged "sparse" on a metric if its score is <= the median.
(The metric maths is unchanged from the original.)

Changes vs the original Stage 2:
  [F1] One shared naming rule. Every node's readable name comes from
       name -> title -> id, and if none exist a unique fallback like
       "Movie:115" -- never the shared word "unnamed". The same rule is
       available as a Cypher expression (cypher_label) for queries and
       as a Python function (python_label) for nodes that are not in
       Neo4j yet (GatedKGWriter). Other files should import these
       instead of writing their own coalesce(...).
  [F2] Stable clusters across restores and runs:
         1. nodes and edges are loaded in a fixed (sorted-by-name) order,
            so Louvain gets identical input every time;
         2. clusters are numbered by their alphabetically-first member,
            not by Louvain's list order;
         3. clusters can be SAVED on the seed as a `seed_cluster`
            property (save_seed_clusters) and READ back
            (load_seed_clusters). The snapshot keeps every property, so
            each restore brings back exactly the same clusters.

Run with (from the project root):
    python newpipeline\\stage2_sparsity_metrics2.py                  # metrics + clusters, read-only
    python newpipeline\\stage2_sparsity_metrics2.py --save-clusters  # also save clusters on the seed
Use --save-clusters ONCE on the new seed, right before snapshotting it.
"""

import os
import sys
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


# =============================================================================
# [F1] One shared naming rule
# =============================================================================
# The ONLY place the order of identity properties is written down. Both
# functions below follow it, so the Cypher and Python versions can never
# disagree.
IDENTITY_KEYS = ("name", "title", "id")


def cypher_label(var: str = "n") -> str:
    """[F1] Cypher expression giving node `var` a readable, unique name.

    Tries name, then title, then id. If none exist, builds
    "<Type>:<short id>" (e.g. "Movie:115") from the node's first
    non-internal label and the last part of its elementId -- unique per
    node, never the shared word "unnamed".

    Use inside a query, e.g.
        f"MATCH (n:__Entity__) RETURN {cypher_label('n')} AS label"
    """
    keys = ", ".join(f"{var}.{key}" for key in IDENTITY_KEYS)
    fallback = (
        f"coalesce(head([l IN labels({var}) WHERE NOT l STARTS WITH '__']), 'Entity')"
        f" + ':' + last(split(elementId({var}), ':'))"
    )
    return f"toString(coalesce({keys}, {fallback}))"


def python_label(properties: dict, type_label: str = None, node_id=None) -> str:
    """[F1] Same rule as cypher_label, for nodes that are not in Neo4j yet
    (e.g. extracted nodes inside GatedKGWriter, before they are written).

    properties : the node's property dict
    type_label : its type, e.g. "Person" or "Work" (used only for the fallback)
    node_id    : any id for the node (used only for the fallback)
    """
    properties = properties or {}
    for key in IDENTITY_KEYS:
        value = properties.get(key)
        if value is not None and str(value).strip():
            return str(value)
    short_id = str(node_id).split(":")[-1] if node_id is not None else "?"
    return f"{type_label or 'Entity'}:{short_id}"


# =============================================================================
# Loading the graph  ([F1] names, [F2] fixed order)
# =============================================================================
def load_entity_graph(driver) -> nx.Graph:
    """Pull only the __Entity__ nodes (real concepts) out of Neo4j into a
    NetworkX graph. Document/Chunk bookkeeping nodes are excluded -- they
    aren't "concepts" in the paper's sense and would distort the metrics.

    [F1] names come from cypher_label (no more "unnamed").
    [F2] nodes are added sorted by name, and edges sorted by the names of
         their two ends. Neo4j gives nodes new elementIds after every
         restore and may return them in a different order; sorting by
         name means NetworkX (and therefore Louvain) sees identical
         input every time.
    """
    G = nx.Graph()

    with driver.session() as session:
        node_records = session.run(
            f"MATCH (n:__Entity__) "
            f"RETURN elementId(n) AS id, {cypher_label('n')} AS label"
        ).data()

        edge_records = session.run(
            "MATCH (n:__Entity__)-[r]-(m:__Entity__) "
            "RETURN elementId(n) AS source, elementId(m) AS target"
        ).data()

    # Sort by name; elementId only breaks ties between identically named nodes.
    node_records.sort(key=lambda r: (r["label"], r["id"]))
    labels = {}
    for record in node_records:
        G.add_node(record["id"], label=record["label"])
        labels[record["id"]] = record["label"]

    # The undirected pattern returns every edge twice (a-b and b-a): keep
    # one copy, with its two ends in name order, then sort all edges.
    edges = set()
    for record in edge_records:
        a, b = record["source"], record["target"]
        if a not in labels or b not in labels:
            continue
        pair = tuple(sorted((a, b), key=lambda node: (labels[node], node)))
        edges.add(pair)
    for a, b in sorted(edges, key=lambda pair: (labels[pair[0]], labels[pair[1]], pair)):
        G.add_edge(a, b)

    return G


# =============================================================================
# The 7 metrics and the median rule (unchanged from the original)
# =============================================================================
def compute_sparsity_metrics(G: nx.Graph) -> dict:
    """The 7 metrics from Table 1 of the paper. Returns
    {node_id: {metric_name: score}}.
    """
    scores: dict = defaultdict(dict)

    # 1. Clique -- size of the largest clique each node belongs to.
    for node, val in nx.node_clique_number(G).items():
        scores[node]["clique"] = val

    # 2. Non-Clique (isolates) -- raw degree; degree 0 = totally isolated.
    for node in G.nodes():
        scores[node]["non_clique"] = G.degree(node)

    # 3. Clustering coefficient -- local triadic closure.
    for node, val in nx.clustering(G).items():
        scores[node]["clustering"] = val

    # 4. Degree centrality.
    for node, val in nx.degree_centrality(G).items():
        scores[node]["degree"] = val

    # 5. Betweenness centrality.
    for node, val in nx.betweenness_centrality(G).items():
        scores[node]["betweenness"] = val

    # 6. Component diameter -- each node inherits its component's diameter.
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        diam = nx.diameter(subgraph) if subgraph.number_of_nodes() > 1 else 0
        for node in component:
            scores[node]["diameter"] = diam

    # 7. Louvain community size.
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


# =============================================================================
# Clusters  ([F2] fixed numbering + save/load on the seed)
# =============================================================================
SEED_CLUSTER_PROPERTY = "seed_cluster"


def get_node_cluster_mapping(G: nx.Graph) -> dict:
    """Runs Louvain (seed 42) and returns {node_id: cluster_id}.

    [F2] Clusters are numbered by their alphabetically-first member name,
    not by Louvain's list order, so the same group always gets the same
    number. Together with load_entity_graph's sorted loading, the same
    seed always gives the same mapping (by member names).

    Meant to be called ONCE on the verified seed -- rerunning it on a
    graph that has grown during enrichment would re-partition everything.
    For runs, prefer load_seed_clusters() / get_or_create_seed_clusters().
    """
    labels = nx.get_node_attributes(G, "label")
    communities = nx.community.louvain_communities(G, seed=42)

    def first_member_name(community):
        return min((labels.get(node, str(node)), node) for node in community)

    node_to_cluster: dict = {}
    for cluster_id, community in enumerate(sorted(communities, key=first_member_name)):
        for node in community:
            node_to_cluster[node] = cluster_id
    return node_to_cluster


def save_seed_clusters(driver, node_to_cluster: dict) -> int:
    """[F2] Writes each node's cluster onto the node itself as the
    `seed_cluster` property. Run ONCE on the new seed, before
    snapshotting it -- the snapshot then carries the clusters, and every
    restore brings back exactly the same ones. Returns how many nodes
    were updated.
    """
    rows = [{"id": node_id, "cluster": cluster_id}
            for node_id, cluster_id in node_to_cluster.items()]
    with driver.session() as session:
        record = session.run(
            f"UNWIND $rows AS row "
            f"MATCH (n:__Entity__) WHERE elementId(n) = row.id "
            f"SET n.{SEED_CLUSTER_PROPERTY} = row.cluster "
            f"RETURN count(n) AS updated",
            rows=rows,
        ).single()
    return record["updated"]


def load_seed_clusters(driver) -> dict:
    """[F2] Reads the saved clusters back: {node_id: cluster_id} for every
    entity that has a `seed_cluster` property. Nodes added during
    enrichment have no such property and are simply not in the result
    (Stage 4 assigns them to the nearest cluster in memory).
    Returns {} if no clusters were ever saved.
    """
    with driver.session() as session:
        records = session.run(
            f"MATCH (n:__Entity__) WHERE n.{SEED_CLUSTER_PROPERTY} IS NOT NULL "
            f"RETURN elementId(n) AS id, n.{SEED_CLUSTER_PROPERTY} AS cluster"
        ).data()
    return {record["id"]: record["cluster"] for record in records}


def get_or_create_seed_clusters(driver, G: nx.Graph) -> dict:
    """[F2] What Stage 4 should call: use the saved clusters if they exist;
    otherwise compute them on G, save them, and warn loudly (because
    that means the seed was frozen without clusters).
    """
    saved = load_seed_clusters(driver)
    if saved:
        return saved

    print("[Stage 2] WARNING: no saved seed clusters found -- computing and "
          "saving them now. Re-snapshot the seed so future restores carry them.")
    mapping = get_node_cluster_mapping(G)
    save_seed_clusters(driver, mapping)
    return mapping


def describe_clusters(G: nx.Graph, node_to_cluster: dict) -> dict:
    """{cluster_id: [sorted member names]} -- for logging and for checking
    that two restores gave the same clusters (compare by names, never by
    elementIds, which change on every restore)."""
    labels = nx.get_node_attributes(G, "label")
    clusters: dict = defaultdict(list)
    for node_id, cluster_id in node_to_cluster.items():
        clusters[cluster_id].append(labels.get(node_id, str(node_id)))
    return {cluster_id: sorted(members) for cluster_id, members in sorted(clusters.items())}


# =============================================================================
# main
# =============================================================================
def main():
    save_clusters = "--save-clusters" in sys.argv[1:]

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

    print("[Stage 2] Per-node sparsity flags (SPARSE = at/below median):\n")
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
    print()

    # [F2] Clusters: saved ones if they exist, otherwise freshly computed.
    saved = load_seed_clusters(driver)
    if save_clusters:
        mapping = get_node_cluster_mapping(G)
        updated = save_seed_clusters(driver, mapping)
        print(f"[Stage 2] Saved seed clusters on {updated} node(s). "
              f"Now snapshot the seed so restores carry them.\n")
    elif saved:
        mapping = saved
        print("[Stage 2] Using SAVED seed clusters.\n")
    else:
        mapping = get_node_cluster_mapping(G)
        print("[Stage 2] No saved clusters -- showing freshly computed ones "
              "(run with --save-clusters to save them).\n")

    for cluster_id, members in describe_clusters(G, mapping).items():
        print(f"  cluster {cluster_id}: {', '.join(members)}")

    unnamed = [l for l in labels.values() if l == "unnamed"]
    print(f"\n[Check] [F1] nodes named 'unnamed': {len(unnamed)} (should be 0)")

    driver.close()


if __name__ == "__main__":
    main()
