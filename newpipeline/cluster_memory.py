"""
Cluster Memory -- in-memory infrastructure for the MKGE memory extension
============================================================================
Implements Phase 1 (community -> node lookup, in stage2_sparsity_metrics.py)
and Phase 2 (this file) of the memory-augmented MKGE design: Representative
Memory + Exemplar Memory, one of each per Louvain cluster, held in a plain
Python ClusterMemoryStore.

SCOPE OF THIS FILE
-------------------
This file builds and holds Cluster Memory: Representative + Exemplar
memory per cluster, rejected-exemplar pruning (FIFO, ExemplarMemory.
add_rejected), nearest-cluster assignment for new nodes
(ClusterMemoryStore.assign_new_node), and the cluster member register
(ClusterMemoryStore.members). The gates themselves live in memory_gates.py.

CHANGES (our pipeline)
-----------------------
  [S6] ClusterMemoryStore.members(cluster_id) is now the ONE register of
       who is in each cluster. Stage 4 used to keep its own second copy
       (clusters_to_nodes) that had to be updated by hand in parallel;
       it now asks the store instead, so the two can never disagree.
  Decided, not changed (see the plan doc): exemplar memory is NOT saved
  between runs (S7: each question has its own seed and clusters), and
  representative_entities stay frozen to the seed (S8: revisit with the
  Post-Extraction Gate redesign).

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
# Small helper (local copy -- see note below)
# =============================================================================
def _cosine_similarity(vec_a, vec_b) -> float:
    """Plain Python cosine similarity, no numpy -- matches this project's
    existing convention (compute_cluster_embedding's average below is
    also plain Python).

    This is intentionally a LOCAL copy of the identical helper already
    in memory_gates.py, not an import from it: memory_gates.py imports
    compute_cluster_embedding FROM this file, so importing back from
    memory_gates.py here would create a circular import. Kept as small,
    duplicated, pure logic rather than restructuring either file's
    imports.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


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
    """One successful enrichment cycle's record, for a given cluster.

    accepted_triple is Optional (default None) -- as of the Question
    Memory Gate's Stage 4 integration, this field is deliberately left
    unpopulated by that integration. Stage 4's ingestion currently goes
    through SimpleKGPipeline as one opaque extraction+ingestion step, so
    there is no clean access to the SPECIFIC triple that came out of a
    given snippet -- inventing one here would be fabricated data, not a
    real exemplar. The field stays on the dataclass for the future
    Post-Extraction Gate (a separate, later phase -- see this file's
    module docstring) to populate once it has real triple-level access;
    it just isn't required to construct an AcceptedExemplar today.
    """
    sparse_node: str
    generated_question: str
    retrieved_evidence: str
    acceptance_reason: str
    accepted_triple: Optional[str] = None


@dataclass
class RejectedExemplar:
    """One failed enrichment attempt's record, for a given cluster."""
    sparse_node: str
    generated_question: str
    failure_type: str
    rejection_reason: str
    confidence_score: float


# Configurable cap on rejected exemplars retained PER CLUSTER. Applies
# only to RejectedExemplar (see ExemplarMemory.add_rejected() below) --
# AcceptedExemplar has no cap, unchanged. Chosen as a plain module
# constant (not derived from anything else) so it's trivially
# overridable per-call in tests without needing to patch this module.
MAX_REJECTED_EXEMPLARS_PER_CLUSTER = 50


@dataclass
class ExemplarMemory:
    """
    Holds both exemplar lists for one cluster. accepted/rejected are plain
    lists.

    accepted is still append-only, unbounded -- pruning in this phase
    applies ONLY to rejected exemplars (see add_rejected() below); this
    was a specific, scoped requirement, not a general exemplar-memory
    change.

    add_rejected() exists as its own method (rather than callers doing
    `exemplars.rejected.append(...)` directly) specifically so that
    pruning could be added later without changing any call site -- that
    is exactly what happens below now.
    """
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)

    def add_accepted(self, exemplar: AcceptedExemplar) -> None:
        self.accepted.append(exemplar)

    def add_rejected(self, exemplar: RejectedExemplar, max_rejected: int = MAX_REJECTED_EXEMPLARS_PER_CLUSTER) -> None:
        """
        Appends exemplar, then enforces a maximum retained count for
        THIS cluster's rejected list only.

        PRUNING POLICY -- deterministic, oldest-first (FIFO)
        -------------------------------------------------------
        self.rejected is a plain list appended to in call order, so
        index 0 is always the least-recently-added (oldest) exemplar
        and index -1 is always the most-recently-added. When len(self.rejected)
        exceeds max_rejected after this append, the OLDEST entries are
        dropped from the front until exactly max_rejected remain --
        i.e. simple FIFO eviction. This is deterministic (no
        similarity/score-based selection, no randomness) and requires
        no new fields on RejectedExemplar itself -- append order IS the
        recency order, nothing extra needs tracking.

        Parameters
        ----------
        exemplar : RejectedExemplar
            Unchanged structure -- this function does not read or
            modify any of its fields.
        max_rejected : int
            Defaults to the module-level MAX_REJECTED_EXEMPLARS_PER_CLUSTER
            constant below, so every existing call site (Stage 4, via
            gate_cluster_memory.exemplars.add_rejected(...)) gets
            pruning automatically with no call-site change required.
            Accepting it as a parameter (rather than only reading the
            module constant inside the method) keeps this testable
            with small limits without needing to monkeypatch a global.
        """
        self.rejected.append(exemplar)
        if max_rejected is not None and len(self.rejected) > max_rejected:
            overflow = len(self.rejected) - max_rejected
            del self.rejected[:overflow]


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
        # cluster_id -> [member_node_id, ...]. Populated once by
        # build_cluster_memory_store() below, and then appended to by
        # assign_new_node() as nodes are assigned during enrichment.
        # This did not exist before assign_new_node() was implemented --
        # build_cluster_memory_store() previously computed this same
        # grouping locally and discarded it, since nothing needed it
        # after initialization. Now it's the minimal piece of state
        # nearest-cluster assignment needs (compute_cluster_embedding
        # requires a cluster's member_node_ids, and ClusterMemory itself
        # only stores human-readable labels, not node ids -- see
        # memory_gates.py's module docstring for the same reasoning).
        self._cluster_member_node_ids: dict = {}

    def add(self, cluster_memory: ClusterMemory) -> None:
        self._clusters[cluster_memory.cluster_id] = cluster_memory

    def get(self, cluster_id: int) -> Optional[ClusterMemory]:
        return self._clusters.get(cluster_id)

    def all_cluster_ids(self) -> list:
        return list(self._clusters.keys())

    def __len__(self) -> int:
        return len(self._clusters)

    def members(self, cluster_id: int) -> list:
        """[S6] The ONE register of who is in a cluster: its seed members
        plus any nodes assign_new_node() has added during this run.
        Returns a copy, so callers can't change the register by accident --
        only assign_new_node() adds to it. Empty list for an unknown id.
        """
        return list(self._cluster_member_node_ids.get(cluster_id, []))

    def assign_new_node(self, node_id: str, node_embedding, driver) -> Optional[int]:
        """
        Minimal nearest-cluster assignment for a node that appears DURING
        enrichment (i.e. after the one-time initial community detection
        in build_cluster_memory_store() already ran). Compares
        node_embedding against each existing cluster's on-demand
        embedding (compute_cluster_embedding() over that cluster's
        CURRENT member_node_ids) and returns the closest cluster_id.

        This does not redo or touch Louvain community detection -- it
        only decides which EXISTING cluster a new node joins, using the
        same on-demand embedding mechanism the gate's semantic signal
        already relies on. No new memory architecture, no new signal.

        Fail-open (returns None), matching the caller's existing
        fail-open handling for unclustered nodes:
          - node_embedding is falsy (e.g. this node has no embedded
            Chunk yet).
          - this store has no clusters at all, or none of them has a
            computable embedding yet (cold start -- e.g. right at the
            start of a run, before any cluster has an embedded member).
        Never raises, never guesses -- same "no fabricated assignment"
        principle as the rest of this file.

        On a successful assignment, node_id is appended to that
        cluster's tracked member_node_ids so this node counts toward
        that cluster's embedding (and toward the gate's semantic
        signal) from this point forward, and so a later call to this
        method for a DIFFERENT new node sees this one too.

        Parameters
        ----------
        node_id : str
            The new node's Neo4j elementId.
        node_embedding : list[float] or None
            The new node's own embedding -- in practice, obtained by the
            caller via compute_cluster_embedding(driver, [node_id]),
            which for a single-node list simply returns that node's own
            Chunk embedding (no new Cypher needed -- reuses the existing
            function as-is).
        driver : neo4j.Driver
            Existing driver, passed through to compute_cluster_embedding()
            for each candidate cluster.

        Returns
        -------
        int or None
            The assigned cluster_id, or None if assignment could not be
            performed (see fail-open cases above).
        """
        if not node_embedding or not self._cluster_member_node_ids:
            return None

        best_cluster_id = None
        best_similarity = -1.0

        for cluster_id, member_node_ids in self._cluster_member_node_ids.items():
            cluster_embedding = compute_cluster_embedding(driver, member_node_ids)
            if cluster_embedding is None:
                continue  # this cluster has no computable embedding yet -- skip, don't guess
            similarity = _cosine_similarity(node_embedding, cluster_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id

        if best_cluster_id is None:
            return None  # no cluster had a usable embedding -- fail-open

        self._cluster_member_node_ids[best_cluster_id].append(node_id)
        return best_cluster_id


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
        # Record this cluster's member node ids on the store itself so
        # assign_new_node() can later recompute this cluster's embedding
        # (and grow this same list) without needing node_cluster_map
        # threaded back in from outside. Previously this grouping was
        # computed into the local clusters_to_nodes dict above and then
        # discarded once the loop finished -- nothing read it afterward.
        store._cluster_member_node_ids[cluster_id] = list(member_node_ids)

    return store