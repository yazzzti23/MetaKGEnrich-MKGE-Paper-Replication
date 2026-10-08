"""
MKGE Cluster-Memory Extension -- Post-Extraction Memory Gate Writer
============================================================
Implements the pre-ingestion interception point the Post-Extraction
Memory Gate needs. This file contains ONLY the KGWriter subclass that
does the interception + delegation -- the actual gate DECISION logic
lives in memory_gates.py's post_extraction_memory_gate(), unchanged in
spirit from how question_memory_gate() is a pure evaluator that Stage 4
calls and acts on.

WHY A CUSTOM KGWriter IS THE RIGHT (AND SMALLEST) INTERCEPTION POINT
----------------------------------------------------------------------
Inspected directly against the installed neo4j_graphrag==1.19.0 source:

  - SimpleKGPipeline.__init__ (experimental/pipeline/kg_builder.py)
    accepts an optional `kg_writer` component and passes it straight
    into its internal PipelineRunner config.
  - KGWriter (components/kg_writer.py) is an abstract Component whose
    only required method is `run(graph: Neo4jGraph, lexical_graph_config)`.
  - Neo4jWriter (the framework's own default KGWriter) receives the
    SAME fully-built Neo4jGraph -- containing every Neo4jNode /
    Neo4jRelationship the LLM extractor produced, each with .label /
    .properties (nodes) or .start_node_id/.end_node_id/.type/.properties
    (relationships) -- and its run() method is the exact place the
    upsert-to-Neo4j Cypher actually happens.

Passing a custom KGWriter subclass as `kg_writer=` therefore means this
class receives extraction results in-memory, BEFORE anything is written
to Neo4j -- genuine pre-ingestion interception, not a post-hoc filter
run after writing. No other part of SimpleKGPipeline needed touching
for this to work; this is the single smallest hook the library exposes
for this purpose (confirmed by reading the source, not assumed).

IMPORTANT -- @validate_call requirement (found via source inspection)
------------------------------------------------------------------------
The abstract base method KGWriter.run() is decorated with pydantic's
@validate_call:

    class KGWriter(Component):
        @abstractmethod
        @validate_call
        async def run(self, graph: Neo4jGraph, ...): ...

That decorator is what coerces the plain dict the pipeline internally
passes between components into a real Neo4jGraph instance before the
method body runs. Overriding run() WITHOUT @validate_call means this
class receives a raw dict instead of a Neo4jGraph object, and
`graph.nodes` then fails with `AttributeError: 'dict' object has no
attribute 'nodes'` -- confirmed by hitting exactly this error against
the live pipeline and then checking the installed package source.
@validate_call is therefore applied to run() below, matching the base
class's own contract exactly (this is what the real Neo4jWriter relies
on too).

WHAT THIS CLASS DOES NOT DO
-----------------------------
It does not change how extraction happens (LLM prompt, entity/relation
schema, chunking) -- only what happens to the extraction's OUTPUT
before it reaches Neo4j. It does not touch Neo4jWriter itself; on an
ACCEPT decision it simply delegates to a real, unmodified Neo4jWriter
instance to perform the actual write.
"""

from typing import Optional

from pydantic import validate_call

from neo4j_graphrag.components.kg_writer import KGWriter, KGWriterModel, Neo4jWriter
from neo4j_graphrag.components.types import Neo4jGraph, LexicalGraphConfig

from memory_gates import post_extraction_memory_gate, PostExtractionGateResult


class GatedKGWriter(KGWriter):
    """
    Wraps a real Neo4jWriter. On each run() call:
      - If no gate context has been set (see set_gate_context() below),
        accepts unconditionally and delegates straight to the inner
        Neo4jWriter -- this is the FAIL-OPEN path, for nodes with no
        cluster assignment, and is IDENTICAL in spirit to Stage 4's
        existing fail-open handling for the Question Memory Gate.
      - Otherwise, translates the extracted Neo4jGraph into the plain
        entity-label / relationship-type lists post_extraction_memory_gate()
        expects, runs the gate, and either delegates to the inner
        Neo4jWriter (ACCEPT) or returns a SUCCESS-with-zero-writes result
        WITHOUT calling the inner writer at all (REJECT) -- meaning a
        rejected extraction, including its Chunk/Document lexical-graph
        nodes, is never written to Neo4j.

    PER-CALL CONTEXT (why set_gate_context()/last_result exist)
    --------------------------------------------------------------
    Stage 4 constructs ONE SimpleKGPipeline (and one GatedKGWriter) for
    the entire run, then calls kg_builder.run_async() once per accepted
    question in a loop. The framework's run(graph, lexical_graph_config)
    signature has no room for "which node/cluster is THIS call for" --
    that context has to be threaded in some other way. set_gate_context()
    lets Stage 4 declare it immediately before each run_async() call;
    last_result lets Stage 4 read the gate's verdict back out immediately
    after, since KGWriterModel's own schema (status: SUCCESS/FAILURE,
    metadata: dict) has no accept/reject concept of its own to (ab)use
    for this.
    """

    def __init__(self, driver, embedder, neo4j_database: Optional[str] = None):
        self._inner_writer = Neo4jWriter(driver=driver, neo4j_database=neo4j_database)
        self._driver = driver
        self._embedder = embedder

        # Set by Stage 4 via set_gate_context() immediately before each
        # kg_builder.run_async() call. cluster_memory=None means
        # "no cluster context for this call" (fail-open case).
        self._node_label: Optional[str] = None
        self._cluster_memory = None
        self._cluster_member_node_ids: Optional[list] = None

        # Populated by run() after each call. None until the first
        # run() call; also explicitly reset to None by set_gate_context()
        # so a stale result from a PREVIOUS call is never mistaken for
        # this call's outcome if run() is (unexpectedly) not invoked.
        self.last_result: Optional[PostExtractionGateResult] = None

    def set_gate_context(self, node_label: Optional[str], cluster_memory,
                          cluster_member_node_ids: Optional[list]) -> None:
        """Called by Stage 4 immediately before each kg_builder.run_async()
        call, so this writer knows which node/cluster that specific
        extraction is being evaluated against. Pass cluster_memory=None
        for a fail-open (unclustered) node -- run() below then accepts
        unconditionally, without calling the gate at all.
        """
        self._node_label = node_label
        self._cluster_memory = cluster_memory
        self._cluster_member_node_ids = cluster_member_node_ids
        self.last_result = None

    @validate_call
    async def run(self, graph: Neo4jGraph,
                   lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig()) -> KGWriterModel:
        if self._cluster_memory is None:
            # Fail-open: no cluster context set for this call. Matches
            # Stage 4's existing fail-open handling for unclustered
            # nodes exactly -- never gates without context.
            self.last_result = None
            return await self._inner_writer.run(graph, lexical_graph_config)

        lexical_labels = lexical_graph_config.lexical_graph_node_labels
        # BUGFIX (Post-Extraction Gate entity_overlap always 0.0):
        # node.label is the generic ontology TYPE the extractor assigned
        # (e.g. "Person", "Work", "Date") -- confirmed against the
        # installed neo4j_graphrag==1.19.0 source: Neo4jNode.label docstring
        # ("The label of the node") and ERExtractionTemplate's own output
        # example, {"id": "0", "label": "Person", "properties": {"name": "John"}},
        # which is the one canonical shape the LLM is shown for every entity
        # type. The actual extracted entity NAME lives in node.properties["name"],
        # not node.label -- the same convention stage2_sparsity_metrics.py's
        # load_entity_graph() already relies on for already-written nodes via
        # coalesce(n.name, n.id, 'unnamed'). Falling back to node.label (not
        # crashing) only for the rare case "name" is absent from properties.
        entity_labels = [
            node.properties.get("name", node.label)
            for node in graph.nodes if node.label not in lexical_labels
        ]
        relationship_types = [rel.type for rel in graph.relationships]

        result = post_extraction_memory_gate(
            node_label=self._node_label,
            extracted_entity_labels=entity_labels,
            extracted_relationship_types=relationship_types,
            cluster_memory=self._cluster_memory,
            cluster_member_node_ids=self._cluster_member_node_ids,
            driver=self._driver,
            embedder=self._embedder,
        )
        self.last_result = result

        if result.decision == "reject":
            # Nothing from this extraction -- not even its Chunk/Document
            # lexical-graph nodes -- is written to Neo4j. Deliberate: a
            # rejected extraction means the retrieved text itself was
            # judged off-topic for this node/cluster, so keeping even
            # its Chunk record would leave an orphaned node behind that
            # nothing else in the graph meaningfully connects to.
            return KGWriterModel(
                status="SUCCESS",
                metadata={
                    "post_extraction_gate": "rejected",
                    "statistics": {"node_count": 0, "relationship_count": 0},
                },
            )

        return await self._inner_writer.run(graph, lexical_graph_config)