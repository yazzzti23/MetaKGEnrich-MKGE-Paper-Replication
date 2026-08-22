"""
Cluster Memory -- in-memory infrastructure for the MKGE memory extension
============================================================================
Implements Phase 1 (community -> node lookup, in stage2_sparsity_metrics.py)
and Phase 2 (this file) of the memory-augmented MKGE design: Representative
Memory + Exemplar Memory, one of each per Louvain cluster, held in a plain
Python ClusterMemoryStore.

SCOPE OF THIS FILE
-------------------
This file ONLY builds and holds Cluster Memory. It does not implement:
  - the Question Memory Gate
  - the Post-Extraction Gate
  - rejected-exemplar pruning logic
  - nearest-cluster assignment for nodes added during enrichment
All four are deliberately left as either absent or as explicit
NotImplementedError interface stubs -- see ClusterMemoryStore.assign_new_node
and ExemplarMemory.add_rejected below for where each later phase plugs in.

STORAGE
-------
Everything here is a plain in-memory Python object (dataclasses + a dict).
Nothing in this file writes to Neo4j. It only READS from Neo4j (via the
`driver` passed into build_cluster_memory_store) to construct Representative
Memory once, from the manually-verified seed KG.

INTENDED CALL SEQUENCE (later phases wire this up -- not done yet, and
nothing below executes on import; this is a library module, not a script)
-------------------------------------------------------------------------
    G = load_entity_graph(driver)                       # existing, stage2
    node_cluster_map = get_node_cluster_mapping(G)       # stage2, Phase 1
    store = build_cluster_memory_store(driver,
                                        node_cluster_map,
                                        G)                # this file, Phase 2
    # store is now ready for Phase 3's gates to read from / write to.
"""

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx


# =============================================================================
# Representative Memory
# =============================================================================
@dataclass
class RepresentativeMemory:
    """
    Compact semantic description of one cluster, built once from the
    verified seed KG. Deliberately does NOT include a stored embedding --
    see compute_cluster_embedding() below, which computes one on demand
    from Neo4j's existing Chunk embeddings whenever a caller needs it,
    rather than keeping a second copy here that could drift out of sync.
    """
    topic_summary: str
    representative_entities: list
    representative_relations: list
    representative_keywords: list


# =============================================================================
# Exemplar Memory
# =============================================================================
@dataclass
class AcceptedExemplar:
    """One successful enrichment cycle's record, for a given cluster."""
    sparse_node: str
    generated_question: str
    retrieved_evidence: str
    accepted_triple: str
    acceptance_reason: str


@dataclass
class RejectedExemplar:
    """One failed enrichment attempt's record, for a given cluster."""
    sparse_node: str
    generated_question: str
    failure_type: str
    rejection_reason: str
    confidence_score: float


@dataclass
class ExemplarMemory:
    """
    Holds both exemplar lists for one cluster. accepted/rejected are plain
    lists -- append-only for now.

    add_rejected() exists as its own method (rather than callers doing
    `exemplars.rejected.append(...)` directly) specifically so that
    pruning has a single, obvious place to be added later: once a cap is
    decided (see the earlier architecture review), the prune step slots
    in right after the append below, and every future call site
    automatically benefits without needing to change anything about how
    they call add_rejected(). No pruning logic exists yet, per this
    phase's scope.
    """
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)

    def add_accepted(self, exemplar: AcceptedExemplar) -> None:
        self.accepted.append(exemplar)

    def add_rejected(self, exemplar: RejectedExemplar) -> None:
        # Pruning belongs here once implemented (Phase 3+), e.g.:
        #   self.rejected.append(exemplar)
        #   self.rejected = self._prune(self.rejected)
        # Left as a plain append for now, intentionally.
        self.rejected.append(exemplar)


# =============================================================================
# Cluster Memory (Representative + Exemplar, per cluster)
# =============================================================================
@dataclass
class ClusterMemory:
    cluster_id: int
    representative: RepresentativeMemory
    exemplars: ExemplarMemory = field(default_factory=ExemplarMemory)


class ClusterMemoryStore:
    """
    Maps cluster_id -> ClusterMemory. Plain in-memory dict wrapper --
    no Neo4j involved. Kept as a thin class (rather than a bare dict)
    specifically so that a future migration to Neo4j-backed storage only
    requires changing what happens INSIDE add()/get()/all_cluster_ids(),
    not every call site that uses a ClusterMemoryStore.
    """

    def __init__(self):
        self._clusters: dict = {}

    def add(self, cluster_memory: ClusterMemory) -> None:
        self._clusters[cluster_memory.cluster_id] = cluster_memory

    def get(self, cluster_id: int) -> Optional[ClusterMemory]:
        return self._clusters.get(cluster_id)

    def all_cluster_ids(self) -> list:
        return list(self._clusters.keys())

    def __len__(self) -> int:
        return len(self._clusters)

    def assign_new_node(self, node_id: str, node_embedding) -> int:
        """
        INTERFACE ONLY -- intentionally not implemented in this phase.

        Later (once Stage 4's orchestration and the memory gates are being
        built), this is where a node that appears DURING enrichment --
        i.e. after the one-time initial community detection this file
        performs -- gets assigned to its nearest EXISTING cluster, most
        likely by comparing node_embedding against each cluster's
        on-demand embedding from compute_cluster_embedding() below and
        returning the closest cluster_id.

        Raising NotImplementedError here (rather than silently doing
        nothing, or a naive default) is deliberate: it makes it obvious
        and immediate if anything tries to call this before Phase 3 exists,
        instead of failing silently or producing wrong cluster
        assignments.
        """
        raise NotImplementedError(
            "assign_new_node() is a placeholder interface for "
            "nearest-existing-cluster assignment. Not implemented in this "
            "phase -- community detection is only run once, on the "
            "verified seed KG, via build_cluster_memory_store(). Wiring "
            "this up is a later-phase task."
        )


# =============================================================================
# On-demand cluster embedding (explicitly NOT stored anywhere above)
# =============================================================================
def compute_cluster_embedding(driver, member_node_ids: list):
    """
    Computes a cluster's embedding ON DEMAND by averaging the embeddings
    of the Chunk nodes that the cluster's member __Entity__ nodes were
    extracted from (the same Chunk.embedding property SimpleKGPipeline's
    embedder already writes during Stage 1 / Stage 4 ingestion).

    Nothing in this file calls this function yet -- it's infrastructure
    for the memory gates (Phase 3+), which will need cluster embeddings to
    compute semantic-similarity signals. It's included now, alongside the
    rest of Cluster Memory, because "compute embeddings on demand instead
    of storing them" is a property of how Cluster Memory is meant to work,
    not something that belongs bolted onto the gates later.

    Parameters
    ----------
    driver : neo4j.Driver
        Existing Neo4j driver (same one used throughout the pipeline).
    member_node_ids : list[str]
        elementIds of the __Entity__ nodes belonging to one cluster --
        i.e. the node_ids that map to a given cluster_id in the output of
        get_node_cluster_mapping() (stage2_sparsity_metrics.py).

    Returns
    -------
    list[float] or None
        The mean embedding vector across the cluster's member nodes'
        chunks, or None if no member node had an embedded chunk to draw
        from (e.g. an empty or degenerate cluster).
    """
    if not member_node_ids:
        return None

    vectors = []
    with driver.session() as session:
        for node_id in member_node_ids:
            result = session.run(
                "MATCH (n)-[:FROM_CHUNK]-(c:Chunk) "
                "WHERE elementId(n) = $node_id AND c.embedding IS NOT NULL "
                "RETURN c.embedding AS embedding LIMIT 1",
                node_id=node_id,
            )
            record = result.single()
            if record and record["embedding"] is not None:
                vectors.append(record["embedding"])

    if not vectors:
        return None

    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


# =============================================================================
# Cluster relation lookup (small helper, used only by initialization below)
# =============================================================================
def _get_cluster_relation_types(driver, member_node_ids: list) -> list:
    """
    Looks up the distinct relationship types connecting entities WITHIN
    one cluster.

    This can't be read off the graph object stage2's load_entity_graph()
    already builds, because that function's edge-loading Cypher query
    only checks THAT an edge exists between two __Entity__ nodes -- it
    never returns type(r). Rather than changing that existing query's
    return shape (which stage3/stage4 also depend on), this issues its
    own small, separate query.
    """
    if not member_node_ids:
        return []

    with driver.session() as session:
        result = session.run(
            "MATCH (n:__Entity__)-[r]-(m:__Entity__) "
            "WHERE elementId(n) IN $node_ids AND elementId(m) IN $node_ids "
            "RETURN DISTINCT type(r) AS rel_type",
            node_ids=member_node_ids,
        )
        return [record["rel_type"] for record in result]


# =============================================================================
# Topic summary + keywords (optional LLM call, used only by initialization)
# =============================================================================
def _summarize_cluster(llm_client, entities: list, relations: list):
    """
    Generates a one-sentence Topic Summary + a short keyword list for one
    cluster, using the same OpenAI-compatible client pattern already used
    throughout stage3/stage4/stage5 (Groq-hosted, plain-text output rather
    than structured output -- see stage1_seed_kg.py's
    `llm.supports_structured_output = False` for why this codebase avoids
    structured output with Groq).

    Called at most ONCE PER CLUSTER during build_cluster_memory_store()
    below -- not once per enrichment cycle -- so this does not add to the
    per-question LLM-call budget pressure already noted for Stage 4's loop
    (40s sleep between calls, QUESTIONS_PER_METRIC dropped from 5 to 2 for
    quota reasons). For a single seed KG the number of clusters is
    typically small, so this is a cheap, one-time cost.
    """
    entity_list = ", ".join(entities) if entities else "(no named entities)"
    relation_list = ", ".join(relations) if relations else "(no relation types found)"

    prompt = f"""You are summarizing one community (cluster) of a knowledge graph.

Entities in this cluster: {entity_list}
Relation types connecting them: {relation_list}

Respond in EXACTLY two lines, nothing else:
Line 1: a one-sentence topic summary of what this cluster is about.
Line 2: 3 to 6 comma-separated keywords describing this cluster.
"""

    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    lines = [
        line.strip()
        for line in response.choices[0].message.content.splitlines()
        if line.strip()
    ]

    topic_summary = lines[0] if len(lines) >= 1 else f"Cluster of entities: {entity_list}"
    keywords = [kw.strip() for kw in lines[1].split(",")] if len(lines) >= 2 else []

    return topic_summary, keywords


# =============================================================================
# Cluster Memory initialization -- the one-time entry point
# =============================================================================
def build_cluster_memory_store(driver, node_cluster_map: dict, G: nx.Graph,
                                llm_client=None) -> ClusterMemoryStore:
    """
    ONE-TIME initialization of Cluster Memory. Intended to run exactly
    once, immediately after the seed KG has been manually verified --
    NOT re-run per enrichment cycle. (This mirrors why
    get_node_cluster_mapping() in stage2_sparsity_metrics.py is also
    meant to be called once, on that same verified graph, rather than
    inside Stage 4's per-metric loop the way sparsity detection currently
    is.)

    Groups node_cluster_map by cluster, builds a RepresentativeMemory per
    cluster directly from the graph's own structure (entities, relation
    types) plus an optional LLM-generated summary, and returns a
    ClusterMemoryStore with every cluster's ExemplarMemory left empty --
    nothing has been accepted or rejected yet, since no gate has run.

    Parameters
    ----------
    driver : neo4j.Driver
        Existing Neo4j driver (same one used throughout the pipeline).
    node_cluster_map : dict[str, int]
        Output of get_node_cluster_mapping() (stage2_sparsity_metrics.py).
    G : networkx.Graph
        The SAME graph object node_cluster_map was computed from (i.e.
        load_entity_graph(driver)'s return value) -- reused here to read
        each node's label without an extra Neo4j round-trip.
    llm_client : optional
        An OpenAI-compatible client (e.g. the same Groq-backed
        `OpenAI(api_key=..., base_url="https://api.groq.com/openai/v1")`
        client already constructed in stage3/stage4/stage5's main()
        functions). If provided, used to generate Topic Summary +
        keywords per cluster via _summarize_cluster(). If None, a simple
        deterministic fallback is used instead (entity-name based), so
        this function -- and this whole module -- can be tested without
        any API key.

    Returns
    -------
    ClusterMemoryStore
    """
    clusters_to_nodes: dict = {}
    for node_id, cluster_id in node_cluster_map.items():
        clusters_to_nodes.setdefault(cluster_id, []).append(node_id)

    labels = nx.get_node_attributes(G, "label")

    store = ClusterMemoryStore()

    for cluster_id, member_node_ids in clusters_to_nodes.items():
        entities = [labels.get(nid, nid) for nid in member_node_ids]
        relations = _get_cluster_relation_types(driver, member_node_ids)

        if llm_client is not None:
            topic_summary, keywords = _summarize_cluster(llm_client, entities, relations)
        else:
            topic_summary = f"Cluster of entities: {', '.join(entities)}"
            keywords = list(dict.fromkeys(e.lower() for e in entities))

        representative = RepresentativeMemory(
            topic_summary=topic_summary,
            representative_entities=entities,
            representative_relations=relations,
            representative_keywords=keywords,
        )

        store.add(ClusterMemory(cluster_id=cluster_id, representative=representative))

    return store