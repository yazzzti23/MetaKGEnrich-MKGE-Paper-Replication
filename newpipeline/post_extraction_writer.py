"""
MKGE -- OUR PIPELINE -- Post-Extraction Gate writer (GatedKGWriter)
============================================================
SimpleKGPipeline accepts a custom `kg_writer`. GatedKGWriter is passed in
as that writer, so it receives every extraction result IN MEMORY, before
anything is written to Neo4j (confirmed against neo4j_graphrag==1.19.0:
the pipeline runs extractor -> pruner -> writer, and the writer gets the
full Neo4jGraph: extracted entities + facts AND the lexical graph, i.e.
Chunk/Document nodes and FROM_CHUNK / NEXT_CHUNK / FROM_DOCUMENT links).

[PG2] Post-Extraction Gate v2 -- what changed vs the previous writer
-------------------------------------------------------------------
Before: one score for the WHOLE page (share of already-known names +
semantic similarity, pass at 0.70) -> write everything or nothing.
Now: judge each ENTITY and FACT (memory_gates.post_extraction_filter):
  - anchors = target node + cluster members (names read from Neo4j) +
    the cluster's representative entities;
  - keep entities that are anchors or connected to one within 2 links in
    this page's extraction; keep facts whose two ends were kept;
  - skip facts that already exist in the graph (duplicate check against
    Neo4j); log conflicts on one-value relations (logged, not acted on);
  - per-fact relevance to the question is computed for the LOG only.
Then this writer builds a FILTERED graph and hands only that to the real
Neo4jWriter:
  - kept entities + kept new facts;
  - Chunk nodes that at least one kept entity came from (+ their
    FROM_CHUNK links to kept entities, NEXT_CHUNK links between kept
    chunks, FROM_DOCUMENT links, and the Document node if any chunk is
    kept). Chunks with no kept entity are dropped, so off-topic text never
    reaches Stage 5's vector retrieval either.
If nothing new survives, NOTHING is written (same as the old REJECT).

[F1] Entity names come from python_label (stage2_sparsity_metrics1):
name -> title -> id -> "<Type>:<id>", never the type label alone.

Fail-open is unchanged: with no gate context (cluster_memory=None), the
extraction is written as-is.

@validate_call on run() is REQUIRED: the base KGWriter.run uses it to turn
the plain dict the pipeline passes into a real Neo4jGraph.
"""

from typing import Optional

from pydantic import validate_call

from neo4j_graphrag.components.kg_writer import KGWriter, KGWriterModel, Neo4jWriter
from neo4j_graphrag.components.types import Neo4jGraph, LexicalGraphConfig

from memory_gates import (
    post_extraction_filter,
    PostExtractionGateResult,
    FUNCTIONAL_RELATIONS,
    fact_key,
    _norm_name,
    _cosine_similarity,
)


class GatedKGWriter(KGWriter):
    """Wraps a real Neo4jWriter and filters each extraction before writing.

    Stage 4 calls set_gate_context(...) right before each
    kg_builder.run_async(...), and reads `last_result` right after.
    """

    def __init__(self, driver, embedder, neo4j_database: Optional[str] = None):
        self._inner_writer = Neo4jWriter(driver=driver, neo4j_database=neo4j_database)
        self._driver = driver
        self._embedder = embedder

        self._node_label: Optional[str] = None
        self._cluster_memory = None
        self._cluster_member_node_ids: Optional[list] = None
        self._question: Optional[str] = None

        self.last_result: Optional[PostExtractionGateResult] = None

    def set_gate_context(self, node_label: Optional[str], cluster_memory,
                         cluster_member_node_ids: Optional[list], question: Optional[str] = None) -> None:
        """Called by Stage 4 immediately before each kg_builder.run_async().
        cluster_memory=None -> fail-open (write without gating).
        [PG2] question (optional) is used only for the logged per-fact
        relevance score."""
        self._node_label = node_label
        self._cluster_memory = cluster_memory
        self._cluster_member_node_ids = cluster_member_node_ids
        self._question = question
        self.last_result = None

    # ------------------------------------------------------------------
    # Neo4j lookups (skipped gracefully when there is no driver, e.g. tests)
    # ------------------------------------------------------------------
    def _member_names(self) -> list:
        """Current names of the cluster's members (seed + nodes assigned
        during this run), via the shared naming rule."""
        if self._driver is None or not self._cluster_member_node_ids:
            return []
        from stage2_sparsity_metrics1 import cypher_label
        with self._driver.session() as session:
            records = session.run(
                f"MATCH (n:__Entity__) WHERE elementId(n) IN $ids "
                f"RETURN {cypher_label('n')} AS label",
                ids=list(self._cluster_member_node_ids),
            ).data()
        return [r["label"] for r in records if r["label"]]

    def _existing_facts(self, entity_names: list):
        """Facts already in the graph whose two ends both have one of
        `entity_names` (normalised), plus existing values of one-value
        relations. Returns (existing_fact_keys, existing_functional)."""
        keys, functional = set(), {}
        names = sorted({_norm_name(n) for n in entity_names if n})
        if self._driver is None or not names:
            return keys, functional
        from stage2_sparsity_metrics1 import cypher_label
        a_label, b_label = cypher_label("a"), cypher_label("b")
        with self._driver.session() as session:
            records = session.run(
                f"MATCH (a:__Entity__)-[r]->(b:__Entity__) "
                f"WITH a, r, b, toLower({a_label}) AS an, toLower({b_label}) AS bn "
                f"WHERE an IN $names AND (bn IN $names OR type(r) IN $functional) "
                f"RETURN {a_label} AS a_name, type(r) AS rel, {b_label} AS b_name",
                names=names, functional=sorted(FUNCTIONAL_RELATIONS),
            ).data()
        for r in records:
            keys.add(fact_key(r["a_name"], r["rel"], r["b_name"]))
            if r["rel"].upper() in FUNCTIONAL_RELATIONS:
                functional.setdefault((_norm_name(r["a_name"]), r["rel"].upper()), set()).add(
                    _norm_name(r["b_name"]))
        return keys, functional

    def _relevance_fn(self):
        """Per-fact relevance for the LOG: cosine similarity between the
        verbalised fact and the question (or the target name)."""
        if self._embedder is None:
            return None
        reference = self._question or self._node_label
        if not reference:
            return None
        reference_vec = self._embedder.embed_query(reference)
        return lambda text: _cosine_similarity(self._embedder.embed_query(text), reference_vec)

    # ------------------------------------------------------------------
    @validate_call
    async def run(self, graph: Neo4jGraph,
                  lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig()) -> KGWriterModel:
        if self._cluster_memory is None:
            # Fail-open: no cluster context -> write as-is, no gate.
            self.last_result = None
            return await self._inner_writer.run(graph, lexical_graph_config)

        from stage2_sparsity_metrics1 import python_label  # [F1]

        lexical_labels = set(lexical_graph_config.lexical_graph_node_labels)
        lexical_rel_types = set(lexical_graph_config.lexical_graph_relationship_types)
        node_to_chunk = lexical_graph_config.node_to_chunk_relationship_type
        next_chunk = lexical_graph_config.next_chunk_relationship_type
        chunk_to_doc = lexical_graph_config.chunk_to_document_relationship_type
        chunk_label = lexical_graph_config.chunk_node_label
        document_label = lexical_graph_config.document_node_label

        lexical_nodes = {n.id: n for n in graph.nodes if n.label in lexical_labels}
        entity_nodes = {n.id: n for n in graph.nodes if n.label not in lexical_labels}

        entities = [{"id": n.id, "name": python_label(n.properties, n.label, n.id), "type": n.label}
                    for n in entity_nodes.values()]
        fact_rels = [r for r in graph.relationships
                     if r.type not in lexical_rel_types
                     and r.start_node_id in entity_nodes and r.end_node_id in entity_nodes]
        facts = [{"start": r.start_node_id, "type": r.type, "end": r.end_node_id} for r in fact_rels]

        rep = self._cluster_memory.representative
        anchor_names = list(rep.representative_entities or []) + self._member_names()
        existing_keys, existing_functional = self._existing_facts([e["name"] for e in entities])

        result = post_extraction_filter(
            target_label=self._node_label,
            anchor_names=anchor_names,
            entities=entities,
            facts=facts,
            existing_fact_keys=existing_keys,
            existing_functional=existing_functional,
            relevance_fn=self._relevance_fn(),
        )
        self.last_result = result

        if result.decision == "reject":
            # Nothing new and connected -> write nothing (not even the chunks).
            return KGWriterModel(
                status="SUCCESS",
                metadata={
                    "post_extraction_gate": "rejected",
                    "reason": result.signals.get("reason"),
                    "statistics": {"node_count": 0, "relationship_count": 0},
                },
            )

        # ---- Build the filtered graph.
        kept_entities = result.kept_entity_ids
        kept_fact_rels = [fact_rels[i] for i in result.kept_fact_indices]

        # Chunks that at least one kept entity came from.
        kept_chunk_ids = set()
        for r in graph.relationships:
            if r.type != node_to_chunk:
                continue
            ends = {r.start_node_id, r.end_node_id}
            if ends & kept_entities:
                kept_chunk_ids |= {i for i in ends if i in lexical_nodes
                                   and lexical_nodes[i].label == chunk_label}

        kept_lexical_rels, kept_doc_ids = [], set()
        for r in graph.relationships:
            if r.type not in lexical_rel_types:
                continue
            a, b = r.start_node_id, r.end_node_id
            if r.type == node_to_chunk:
                keep = ({a, b} & kept_entities) and ({a, b} & kept_chunk_ids)
            elif r.type == next_chunk:
                keep = a in kept_chunk_ids and b in kept_chunk_ids
            elif r.type == chunk_to_doc:
                keep = bool({a, b} & kept_chunk_ids)
                if keep:
                    kept_doc_ids |= {i for i in (a, b) if i in lexical_nodes
                                     and lexical_nodes[i].label == document_label}
            else:
                keep = False
            if keep:
                kept_lexical_rels.append(r)

        kept_nodes = ([entity_nodes[i] for i in entity_nodes if i in kept_entities]
                      + [lexical_nodes[i] for i in lexical_nodes
                         if i in kept_chunk_ids or i in kept_doc_ids])
        filtered = Neo4jGraph(nodes=kept_nodes, relationships=kept_fact_rels + kept_lexical_rels)

        return await self._inner_writer.run(filtered, lexical_graph_config)
