"""
Focused test for the Post-Extraction Memory Gate -- NOT part of the
pipeline, does NOT touch live Neo4j, does NOT call Groq/Tavily/any API.

Tests two things, at two levels:

1. post_extraction_memory_gate() (memory_gates.py) directly, with
   compute_cluster_embedding monkeypatched to return fixed vectors --
   covers the ACCEPT case, the REJECT-by-low-score case, and the
   immediate-REJECT-on-empty-extraction case.

2. GatedKGWriter.run() (post_extraction_writer.py) using a REAL
   neo4j_graphrag Neo4jGraph/Neo4jNode/Neo4jRelationship (so the test
   exercises the actual translation logic, not a hand-waved shape) but
   with GatedKGWriter's own inner Neo4jWriter replaced by a no-op fake
   that only records whether it was called -- confirming REJECT means
   nothing is delegated to the real writer, and fail-open (no gate
   context) always delegates regardless of content.

Run with:  python test_post_extraction_gate.py
(no .env needed -- nothing here talks to Neo4j, Groq, or Tavily)
"""

import asyncio

from neo4j_graphrag.components.kg_writer import KGWriterModel
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship, LexicalGraphConfig

import memory_gates
import cluster_memory
from memory_gates import post_extraction_memory_gate
from cluster_memory import ClusterMemory, RepresentativeMemory
from post_extraction_writer import GatedKGWriter


# =============================================================================
# Fakes -- no real embedder, no real Neo4j driver, no real LLM
# =============================================================================
class FakeEmbedder:
    """embed_query() returns a fixed vector keyed by whether "reliance"
    (case-insensitive) appears in the text -- just enough to make
    semantic similarity deterministic and controllable per test case,
    without needing a real SentenceTransformer model loaded.
    """
    def embed_query(self, text: str):
        return [1.0, 0.0] if "reliance" in text.lower() else [0.0, 1.0]


class FakeInnerWriter:
    """Stands in for GatedKGWriter's real Neo4jWriter. Records whether
    run() was called, so tests can assert a REJECT never reaches it."""
    def __init__(self):
        self.called = False
        self.received_graph = None

    async def run(self, graph, lexical_graph_config=LexicalGraphConfig()):
        self.called = True
        self.received_graph = graph
        return KGWriterModel(status="SUCCESS", metadata={"statistics": {
            "node_count": len(graph.nodes), "relationship_count": len(graph.relationships),
        }})


def _fake_cluster_embedding(matching_vector):
    """Returns a monkeypatch-ready replacement for compute_cluster_embedding
    that ignores its real args and always returns matching_vector (or None,
    to simulate cold-start)."""
    def _fn(driver, member_node_ids):
        return matching_vector
    return _fn


def _make_cluster_memory(representative_entities):
    return ClusterMemory(
        cluster_id=0,
        representative=RepresentativeMemory(
            topic_summary="Reliance Industries and its business empire",
            representative_entities=representative_entities,
            representative_relations=["OWNS", "LOCATED_IN"],
            representative_keywords=["reliance", "ambani", "conglomerate"],
        ),
    )


# =============================================================================
# Level 1: post_extraction_memory_gate() directly
# =============================================================================
def test_gate_accepts_on_topic_extraction():
    target_cluster_memory = _make_cluster_memory(["Reliance Industries", "Mukesh Ambani"])
    cluster_memory.compute_cluster_embedding = _fake_cluster_embedding([1.0, 0.0])  # "reliance"-like

    result = post_extraction_memory_gate(
        node_label="Reliance Industries",
        extracted_entity_labels=["Reliance Industries", "Mukesh Ambani", "Jio"],
        extracted_relationship_types=["OWNS"],
        cluster_memory=target_cluster_memory,
        cluster_member_node_ids=["n1", "n2"],
        driver=None,
        embedder=FakeEmbedder(),
    )

    assert result.decision == "accept", f"expected accept, got {result.decision} ({result.signals})"
    assert result.signals["semantic"] is not None
    assert result.kept_entity_labels == ["Reliance Industries", "Mukesh Ambani", "Jio"]
    print("test_gate_accepts_on_topic_extraction: PASS ->", result.decision, result.signals)


def test_gate_rejects_off_topic_extraction():
    target_cluster_memory = _make_cluster_memory(["Reliance Industries", "Mukesh Ambani"])
    # Cluster embedding is the OPPOSITE vector from what FakeEmbedder
    # returns for off-topic text -- low semantic score.
    cluster_memory.compute_cluster_embedding = _fake_cluster_embedding([1.0, 0.0])

    result = post_extraction_memory_gate(
        node_label="Reliance Industries",
        extracted_entity_labels=["Black Swan", "Darren Aronofsky"],  # movie-domain noise
        extracted_relationship_types=["DIRECTED"],
        cluster_memory=target_cluster_memory,
        cluster_member_node_ids=["n1", "n2"],
        driver=None,
        embedder=FakeEmbedder(),
    )

    assert result.decision == "reject", f"expected reject, got {result.decision} ({result.signals})"
    assert result.kept_entity_labels == []
    print("test_gate_rejects_off_topic_extraction: PASS ->", result.decision, result.signals)


def test_gate_rejects_empty_extraction_immediately():
    target_cluster_memory = _make_cluster_memory(["Reliance Industries"])
    cluster_memory.compute_cluster_embedding = _fake_cluster_embedding([1.0, 0.0])

    result = post_extraction_memory_gate(
        node_label="Reliance Industries",
        extracted_entity_labels=[],
        extracted_relationship_types=[],
        cluster_memory=target_cluster_memory,
        cluster_member_node_ids=["n1"],
        driver=None,
        embedder=FakeEmbedder(),
    )

    assert result.decision == "reject"
    assert result.signals == {"semantic": None, "entity_overlap": 0.0}
    print("test_gate_rejects_empty_extraction_immediately: PASS ->", result.decision, result.signals)


def test_gate_cold_start_semantic_still_decides_on_entity_overlap():
    target_cluster_memory = _make_cluster_memory(["Reliance Industries", "Mukesh Ambani"])
    cluster_memory.compute_cluster_embedding = _fake_cluster_embedding(None)  # cold start

    result = post_extraction_memory_gate(
        node_label="Reliance Industries",
        extracted_entity_labels=["Reliance Industries", "Mukesh Ambani"],
        extracted_relationship_types=["OWNS"],
        cluster_memory=target_cluster_memory,
        cluster_member_node_ids=["n1"],
        driver=None,
        embedder=FakeEmbedder(),
    )

    assert result.signals["semantic"] is None
    assert result.decision == "accept", f"expected accept on full entity overlap alone, got {result.decision}"
    print("test_gate_cold_start_semantic_still_decides_on_entity_overlap: PASS ->", result.decision, result.signals)


# =============================================================================
# Level 2: GatedKGWriter.run() -- interception + delegation behavior
# =============================================================================
def _make_extracted_graph(entity_labels, chunk_text="some chunk text"):
    nodes = [Neo4jNode(id="chunk-1", label="Chunk", properties={"text": chunk_text})]
    nodes += [Neo4jNode(id=f"e{i}", label=label, properties={"name": label})
              for i, label in enumerate(entity_labels)]
    relationships = [
        Neo4jRelationship(start_node_id="chunk-1", end_node_id="e0", type="FROM_CHUNK", properties={})
    ] if entity_labels else []
    return Neo4jGraph(nodes=nodes, relationships=relationships)


def test_writer_fail_open_delegates_without_gating():
    writer = GatedKGWriter.__new__(GatedKGWriter)  # bypass __init__'s real Neo4jWriter construction
    writer._inner_writer = FakeInnerWriter()
    writer._driver = None
    writer._embedder = FakeEmbedder()
    writer._node_label = None
    writer._cluster_memory = None  # no gate context set -- fail-open
    writer._cluster_member_node_ids = None
    writer.last_result = None

    graph = _make_extracted_graph(["Anything At All"])
    result = asyncio.run(writer.run(graph))

    assert writer._inner_writer.called is True, "fail-open must delegate to inner writer"
    assert writer.last_result is None, "fail-open must not run the gate at all"
    assert result.status == "SUCCESS"
    print("test_writer_fail_open_delegates_without_gating: PASS")


def test_writer_rejects_without_calling_inner_writer():
    target_cluster_memory = _make_cluster_memory(["Reliance Industries"])
    cluster_memory.compute_cluster_embedding = _fake_cluster_embedding([1.0, 0.0])

    writer = GatedKGWriter.__new__(GatedKGWriter)
    writer._inner_writer = FakeInnerWriter()
    writer._driver = None
    writer._embedder = FakeEmbedder()
    writer.set_gate_context(
        node_label="Reliance Industries",
        cluster_memory=target_cluster_memory,
        cluster_member_node_ids=["n1"],
    )

    graph = _make_extracted_graph(["Black Swan", "Darren Aronofsky"])  # off-topic
    result = asyncio.run(writer.run(graph))

    assert writer._inner_writer.called is False, "REJECT must never reach the inner writer"
    assert writer.last_result is not None and writer.last_result.decision == "reject"
    assert result.status == "SUCCESS" and result.metadata["post_extraction_gate"] == "rejected"
    print("test_writer_rejects_without_calling_inner_writer: PASS")


def test_writer_accepts_and_delegates_to_inner_writer():
    target_cluster_memory = _make_cluster_memory(["Reliance Industries", "Mukesh Ambani"])
    cluster_memory.compute_cluster_embedding = _fake_cluster_embedding([1.0, 0.0])

    writer = GatedKGWriter.__new__(GatedKGWriter)
    writer._inner_writer = FakeInnerWriter()
    writer._driver = None
    writer._embedder = FakeEmbedder()
    writer.set_gate_context(
        node_label="Reliance Industries",
        cluster_memory=target_cluster_memory,
        cluster_member_node_ids=["n1"],
    )

    graph = _make_extracted_graph(["Reliance Industries", "Mukesh Ambani"])  # on-topic
    result = asyncio.run(writer.run(graph))

    assert writer._inner_writer.called is True, "ACCEPT must delegate to the inner writer"
    assert writer.last_result.decision == "accept"
    assert result.status == "SUCCESS"
    print("test_writer_accepts_and_delegates_to_inner_writer: PASS")


# =============================================================================
# Level 3: Regression tests for the live bug -- generic ontology TYPE labels
# ("Person"/"Work"/"Date") were being read as if they were entity NAMES.
#
# _make_extracted_graph() above (Level 2) unrealistically puts the entity
# name directly in node.label (label="Black Swan"), so it could never have
# caught this bug -- that's not the shape SimpleKGPipeline's extractor
# actually produces. Confirmed against the installed neo4j_graphrag==1.19.0
# source (ERExtractionTemplate.DEFAULT_TEMPLATE): every extracted node is
# shaped {"label": "<generic ontology type>", "properties": {"name": "<real
# name>"}} -- label is ALWAYS the generic type, never the name.
# _make_realistic_extracted_graph() below builds graphs in that real shape.
# =============================================================================
class FakeGenericVsNameEmbedder:
    """Deterministic stand-in for a real embedding model, built specifically
    to make the ACCEPT/REJECT flip in these regression tests observable.
    Real embeddings of pure ontology-type text ("Person Work Work Date
    OWNS") land nowhere near a topic-specific cluster embedding; real
    embeddings of actual entity names do. This fake reproduces exactly that
    distinction deterministically: text containing ONLY generic
    ontology/relationship tokens embeds orthogonally ([0.0, 1.0]) to a
    [1.0, 0.0] cluster embedding (similarity 0.0, matching the live bug's
    near-zero semantic scores of 0.0561/0.0625); text containing any real
    name embeds as [1.0, 0.0] (similarity 1.0).
    """
    GENERIC_TOKENS = {
        "person", "work", "date", "event", "concept", "organization", "location",
        "related_to", "located_in", "part_of", "created_by", "owns", "caused_by",
        "occurred_on", "associated_with", "from_chunk",
    }

    def embed_query(self, text: str):
        words = set(text.lower().split())
        if words - self.GENERIC_TOKENS:
            return [1.0, 0.0]
        return [0.0, 1.0]


def _make_realistic_extracted_graph(entity_specs, chunk_text="some chunk text"):
    """entity_specs: list of (generic_label, name) tuples. Mirrors the REAL
    shape SimpleKGPipeline's extractor produces for a generic entity schema
    (entities=["Person","Organization","Location","Work","Event","Concept","Date"],
    same as this project's actual Stage 4 config) -- node.label is always the
    generic ontology type, node.properties["name"] holds the real extracted
    name. This is the shape the live bug's output
    (extracted_entities=['Person','Work','Work',...]) came from.
    """
    nodes = [Neo4jNode(id="chunk-1", label="Chunk", properties={"text": chunk_text})]
    nodes += [
        Neo4jNode(id=f"e{i}", label=generic_label, properties={"name": name})
        for i, (generic_label, name) in enumerate(entity_specs)
    ]
    relationships = [
        Neo4jRelationship(start_node_id="chunk-1", end_node_id="e0", type="FROM_CHUNK", properties={})
    ] if entity_specs else []
    return Neo4jGraph(nodes=nodes, relationships=relationships)


def test_generic_labels_used_as_identities_would_reject_pre_fix():
    """
    Reproduces the OLD (buggy) behavior directly at the gate level -- calling
    post_extraction_memory_gate() with generic ontology-type strings as
    extracted_entity_labels, exactly what post_extraction_writer.py used to
    hand it before the fix (node.label instead of
    node.properties.get("name", node.label)). Confirms this shape produces
    entity_overlap=0.0 and a REJECT, matching the live output for
    node='Darren Aronofsky' exactly. This is a negative-space check: it
    guards against ever regressing back to reading node.label directly, and
    documents precisely what the bug looked like.
    """
    target_cluster_memory = _make_cluster_memory(["Darren Aronofsky", "Black Swan"])
    cluster_memory.compute_cluster_embedding = _fake_cluster_embedding([1.0, 0.0])

    result = post_extraction_memory_gate(
        node_label="Darren Aronofsky",
        extracted_entity_labels=["Person", "Work", "Work", "Date"],  # generic TYPES, the bug
        extracted_relationship_types=["OWNS"],
        cluster_memory=target_cluster_memory,
        cluster_member_node_ids=["n1"],
        driver=None,
        embedder=FakeGenericVsNameEmbedder(),
    )

    assert result.signals["entity_overlap"] == 0.0, (
        "generic ontology type strings should never register as entity overlap"
    )
    assert result.decision == "reject", f"expected reject, got {result.decision} ({result.signals})"
    print("test_generic_labels_used_as_identities_would_reject_pre_fix: PASS ->",
          result.decision, result.signals)


def test_writer_extracts_actual_names_not_generic_labels():
    """
    THE regression test for the fix. Builds a Neo4jGraph shaped exactly like
    the live failure (generic node.label + real name in
    node.properties["name"]) and runs it through the real GatedKGWriter.run()
    -- exercising the actual translation logic in post_extraction_writer.py,
    not a hand-constructed entity_labels list. Confirms:
      1. The extraction is now correctly ACCEPTED (entity_overlap and
         semantic both recover once real names are used).
      2. kept_entity_labels contains the real extracted NAMES
         ("Darren Aronofsky", "Black Swan", ...).
      3. None of the generic ontology TYPE strings ("Person"/"Work"/"Date")
         leaked through as if they were entity identities.
    """
    target_cluster_memory = _make_cluster_memory(["Darren Aronofsky", "Black Swan"])
    cluster_memory.compute_cluster_embedding = _fake_cluster_embedding([1.0, 0.0])

    writer = GatedKGWriter.__new__(GatedKGWriter)
    writer._inner_writer = FakeInnerWriter()
    writer._driver = None
    writer._embedder = FakeGenericVsNameEmbedder()
    writer.set_gate_context(
        node_label="Darren Aronofsky",
        cluster_memory=target_cluster_memory,
        cluster_member_node_ids=["n1"],
    )

    graph = _make_realistic_extracted_graph([
        ("Person", "Darren Aronofsky"),
        ("Work", "Black Swan"),
        ("Work", "Requiem for a Dream"),
        ("Date", "2010"),
    ])
    result = asyncio.run(writer.run(graph))

    assert writer.last_result is not None
    assert writer.last_result.decision == "accept", (
        f"expected accept using real names, got {writer.last_result.decision} "
        f"({writer.last_result.signals})"
    )
    assert writer.last_result.kept_entity_labels == [
        "Darren Aronofsky", "Black Swan", "Requiem for a Dream", "2010"
    ], writer.last_result.kept_entity_labels
    for generic_type in ("Person", "Work", "Date"):
        assert generic_type not in writer.last_result.kept_entity_labels, (
            f"generic ontology type {generic_type!r} leaked through as an entity identity"
        )
    assert writer._inner_writer.called is True, "ACCEPT must still delegate to the inner writer"
    print("test_writer_extracts_actual_names_not_generic_labels: PASS ->",
          writer.last_result.decision, writer.last_result.kept_entity_labels)


def test_writer_off_topic_real_names_still_rejects():
    """
    Guards against the fix over-correcting into a rubber stamp: real,
    well-formed entity names that are genuinely off-topic for the target
    cluster must still be rejected, same as before the fix.
    """
    target_cluster_memory = _make_cluster_memory(["Darren Aronofsky", "Black Swan"])
    cluster_memory.compute_cluster_embedding = _fake_cluster_embedding([1.0, 0.0])

    writer = GatedKGWriter.__new__(GatedKGWriter)
    writer._inner_writer = FakeInnerWriter()
    writer._driver = None
    writer._embedder = FakeGenericVsNameEmbedder()
    writer.set_gate_context(
        node_label="Darren Aronofsky",
        cluster_memory=target_cluster_memory,
        cluster_member_node_ids=["n1"],
    )

    graph = _make_realistic_extracted_graph([
        ("Person", "Mukesh Ambani"),
        ("Organization", "Reliance Industries"),
    ])
    result = asyncio.run(writer.run(graph))

    assert writer.last_result is not None
    assert writer.last_result.decision == "reject", (
        f"expected reject for off-topic real names, got {writer.last_result.decision} "
        f"({writer.last_result.signals})"
    )
    assert writer._inner_writer.called is False, "REJECT must never reach the inner writer"
    print("test_writer_off_topic_real_names_still_rejects: PASS ->", writer.last_result.decision)


if __name__ == "__main__":
    test_gate_accepts_on_topic_extraction()
    test_gate_rejects_off_topic_extraction()
    test_gate_rejects_empty_extraction_immediately()
    test_gate_cold_start_semantic_still_decides_on_entity_overlap()
    test_writer_fail_open_delegates_without_gating()
    test_writer_rejects_without_calling_inner_writer()
    test_writer_accepts_and_delegates_to_inner_writer()
    test_generic_labels_used_as_identities_would_reject_pre_fix()
    test_writer_extracts_actual_names_not_generic_labels()
    test_writer_off_topic_real_names_still_rejects()
    print("\nALL POST-EXTRACTION GATE TESTS PASSED")