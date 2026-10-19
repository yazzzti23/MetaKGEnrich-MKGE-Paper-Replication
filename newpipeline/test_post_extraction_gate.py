"""
Offline tests for the Post-Extraction Gate v2 [PG2] -- no Neo4j, no Groq,
no Tavily. Replaces the old tests of the legacy whole-page gate (those
are still in git history).

Level 1: memory_gates.post_extraction_filter()  (pure logic)
Level 2: post_extraction_writer.GatedKGWriter.run() with a REAL
         neo4j_graphrag Neo4jGraph in the shape the extractor produces
         (label = generic type, name in properties), a fake inner writer,
         and no driver (Neo4j lookups are skipped when driver is None).

Run from the project root:
    python newpipeline\\test_post_extraction_gate.py
"""

import asyncio
import os

# Stage 2 reads these at import time; tests never connect to Neo4j.
for _key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
    os.environ.setdefault(_key, "offline-test")

from neo4j_graphrag.components.kg_writer import KGWriterModel
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship, LexicalGraphConfig

from memory_gates import post_extraction_filter, fact_key
from cluster_memory import ClusterMemory, RepresentativeMemory
from post_extraction_writer import GatedKGWriter


# =============================================================================
# Helpers
# =============================================================================
def E(i, name, t="Person"):
    return {"id": i, "name": name, "type": t}


def F(a, rel, b):
    return {"start": a, "type": rel, "end": b}


PORTMAN_ENTITIES = [
    E("e1", "Natalie Portman"),
    E("e2", "Black Swan", "Work"),
    E("e3", "Academy Award for Best Actress", "Event"),
    E("e4", "Jackie", "Work"),
    E("e5", "Pablo Larrain"),
    E("e6", "Cannes Film Festival", "Event"),      # island
    E("e7", "Palme d'Or", "Concept"),              # island
    E("e8", "Chile", "Location"),                  # 3 links from the anchor
]
PORTMAN_FACTS = [
    F("e1", "ASSOCIATED_WITH", "e2"),   # anchor - anchor
    F("e1", "RELATED_TO", "e3"),        # 1 link
    F("e1", "ASSOCIATED_WITH", "e4"),   # 1 link
    F("e4", "CREATED_BY", "e5"),        # 2 links
    F("e5", "LOCATED_IN", "e8"),        # 3 links -> e8 too far
    F("e6", "RELATED_TO", "e7"),        # island, no anchor
]


# =============================================================================
# Level 1: post_extraction_filter
# =============================================================================
def test_off_topic_page_rejected():
    r = post_extraction_filter("Reliance Industries", ["Mukesh Ambani"],
                               [E("a", "Black Swan", "Work"), E("b", "Darren Aronofsky")],
                               [F("b", "CREATED_BY", "a")])
    assert r.decision == "reject" and r.signals["reason"] == "no_anchor_in_extraction", r.signals
    assert r.kept_fact_indices == [] and r.kept_entity_ids == set()
    print("test_off_topic_page_rejected: PASS")


def test_keeps_connected_chain_drops_island_and_far_nodes():
    r = post_extraction_filter("Natalie Portman", ["Black Swan", "Darren Aronofsky"],
                               PORTMAN_ENTITIES, PORTMAN_FACTS)
    assert r.decision == "accept", r.signals
    assert r.kept_entity_ids == {"e1", "e2", "e3", "e4", "e5"}, r.kept_entity_ids
    statuses = [f["status"] for f in r.fact_log]
    assert statuses == ["kept", "kept", "kept", "kept", "dropped_unconnected", "dropped_unconnected"], statuses
    assert r.kept_fact_indices == [0, 1, 2, 3]
    assert r.signals["facts_kept"] == 4 and r.signals["facts_dropped_unconnected"] == 2
    print("test_keeps_connected_chain_drops_island_and_far_nodes: PASS")


def test_duplicates_skipped_and_all_duplicates_rejected():
    existing = {fact_key("Natalie Portman", "ASSOCIATED_WITH", "Black Swan")}
    r = post_extraction_filter("Natalie Portman", ["Black Swan"],
                               [E("e1", "Natalie Portman"), E("e2", "Black Swan", "Work")],
                               [F("e2", "ASSOCIATED_WITH", "e1")],   # reversed direction still a duplicate
                               existing_fact_keys=existing)
    assert r.fact_log[0]["status"] == "duplicate"
    assert r.decision == "reject" and r.signals["reason"] == "nothing_new_only_duplicates", r.signals
    print("test_duplicates_skipped_and_all_duplicates_rejected: PASS")


def test_conflict_logged_but_fact_kept():
    r = post_extraction_filter("Black Swan", [],
                               [E("e1", "Black Swan", "Work"), E("e2", "2011", "Date")],
                               [F("e1", "OCCURRED_ON", "e2")],
                               existing_functional={("black swan", "OCCURRED_ON"): {"2010"}})
    assert r.decision == "accept"
    assert r.fact_log[0]["status"] == "kept" and r.fact_log[0]["conflict_with"] == ["2010"]
    assert r.signals["fact_conflicts"] == 1
    print("test_conflict_logged_but_fact_kept: PASS")


def test_surname_matches_anchor():
    r = post_extraction_filter("Natalie Portman", [],
                               [E("e1", "Portman"), E("e2", "Thor", "Work")],
                               [F("e1", "ASSOCIATED_WITH", "e2")])
    assert r.decision == "accept" and "e1" in r.kept_entity_ids
    print("test_surname_matches_anchor: PASS")


def test_relevance_is_logged_not_used():
    r = post_extraction_filter("Natalie Portman", [], PORTMAN_ENTITIES, PORTMAN_FACTS,
                               relevance_fn=lambda text: 0.0)   # zero relevance everywhere
    assert r.decision == "accept" and len(r.kept_fact_indices) == 4   # nothing dropped because of it
    assert all(f["relevance"] == 0.0 for f in r.fact_log)
    print("test_relevance_is_logged_not_used: PASS")


# =============================================================================
# Level 2: GatedKGWriter.run
# =============================================================================
class FakeInnerWriter:
    def __init__(self):
        self.called = False
        self.received_graph = None

    async def run(self, graph, lexical_graph_config=LexicalGraphConfig()):
        self.called = True
        self.received_graph = graph
        return KGWriterModel(status="SUCCESS", metadata={})


class FakeEmbedder:
    def embed_query(self, text):
        return [1.0, 0.0] if "portman" in text.lower() else [0.0, 1.0]


def _writer(cluster_memory_obj, node_label, question=None):
    w = GatedKGWriter.__new__(GatedKGWriter)       # skip the real Neo4jWriter
    w._inner_writer = FakeInnerWriter()
    w._driver = None                               # Neo4j lookups skipped
    w._embedder = FakeEmbedder()
    w.set_gate_context(node_label=node_label, cluster_memory=cluster_memory_obj,
                       cluster_member_node_ids=["m1"], question=question)
    return w


def _cluster(entities):
    return ClusterMemory(cluster_id=0, representative=RepresentativeMemory(
        topic_summary="t", representative_entities=entities,
        representative_relations=[], representative_keywords=[]))


def _realistic_graph():
    """Two chunks + one document; entities as the extractor shapes them
    (label = generic type, real name in properties). e6/e7 (island) come
    only from chunk c2, so c2 should be dropped entirely."""
    nodes = [
        Neo4jNode(id="doc", label="Document", properties={"path": "document.txt"}),
        Neo4jNode(id="c1", label="Chunk", properties={"text": "Portman ...", "index": 0}),
        Neo4jNode(id="c2", label="Chunk", properties={"text": "Cannes ...", "index": 1}),
        Neo4jNode(id="e1", label="Person", properties={"name": "Natalie Portman"}),
        Neo4jNode(id="e2", label="Work", properties={"title": "Black Swan"}),   # title only -> F1
        Neo4jNode(id="e4", label="Work", properties={"name": "Jackie"}),
        Neo4jNode(id="e6", label="Event", properties={"name": "Cannes Film Festival"}),
        Neo4jNode(id="e7", label="Concept", properties={"name": "Palme d'Or"}),
    ]
    rels = [
        Neo4jRelationship(start_node_id="e1", end_node_id="e2", type="ASSOCIATED_WITH"),
        Neo4jRelationship(start_node_id="e1", end_node_id="e4", type="ASSOCIATED_WITH"),
        Neo4jRelationship(start_node_id="e6", end_node_id="e7", type="RELATED_TO"),
        Neo4jRelationship(start_node_id="c1", end_node_id="doc", type="FROM_DOCUMENT"),
        Neo4jRelationship(start_node_id="c2", end_node_id="doc", type="FROM_DOCUMENT"),
        Neo4jRelationship(start_node_id="c1", end_node_id="c2", type="NEXT_CHUNK"),
        Neo4jRelationship(start_node_id="e1", end_node_id="c1", type="FROM_CHUNK"),
        Neo4jRelationship(start_node_id="e2", end_node_id="c1", type="FROM_CHUNK"),
        Neo4jRelationship(start_node_id="e4", end_node_id="c1", type="FROM_CHUNK"),
        Neo4jRelationship(start_node_id="e6", end_node_id="c2", type="FROM_CHUNK"),
        Neo4jRelationship(start_node_id="e7", end_node_id="c2", type="FROM_CHUNK"),
    ]
    return Neo4jGraph(nodes=nodes, relationships=rels)


def test_writer_fail_open_writes_everything():
    w = _writer(None, None)
    g = _realistic_graph()
    asyncio.run(w.run(g))
    assert w._inner_writer.called and len(w._inner_writer.received_graph.nodes) == len(g.nodes)
    assert w.last_result is None
    print("test_writer_fail_open_writes_everything: PASS")


def test_writer_rejects_off_topic_without_writing():
    w = _writer(_cluster(["Mukesh Ambani"]), "Reliance Industries")
    result = asyncio.run(w.run(_realistic_graph()))
    assert w._inner_writer.called is False
    assert w.last_result.decision == "reject"
    assert result.metadata["post_extraction_gate"] == "rejected"
    print("test_writer_rejects_off_topic_without_writing: PASS")


def test_writer_writes_only_the_filtered_graph():
    w = _writer(_cluster(["Black Swan"]), "Natalie Portman",
                question="Which films has Natalie Portman starred in?")
    asyncio.run(w.run(_realistic_graph()))
    assert w._inner_writer.called
    g = w._inner_writer.received_graph
    node_ids = {n.id for n in g.nodes}
    rels = {(r.start_node_id, r.type, r.end_node_id) for r in g.relationships}
    assert node_ids == {"doc", "c1", "e1", "e2", "e4"}, node_ids           # island + its chunk gone
    assert ("e6", "RELATED_TO", "e7") not in rels
    assert ("c1", "c2", "NEXT_CHUNK") not in rels and ("c1", "NEXT_CHUNK", "c2") not in rels
    assert ("c2", "FROM_DOCUMENT", "doc") not in rels and ("c1", "FROM_DOCUMENT", "doc") in rels
    assert ("e1", "FROM_CHUNK", "c1") in rels and ("e6", "FROM_CHUNK", "c2") not in rels
    for r in g.relationships:                                               # no dangling links
        assert r.start_node_id in node_ids and r.end_node_id in node_ids, r
    assert "Black Swan" in w.last_result.kept_entity_labels                 # [F1] title used as name
    assert all("relevance" in f for f in w.last_result.fact_log)
    print("test_writer_writes_only_the_filtered_graph: PASS")


if __name__ == "__main__":
    test_off_topic_page_rejected()
    test_keeps_connected_chain_drops_island_and_far_nodes()
    test_duplicates_skipped_and_all_duplicates_rejected()
    test_conflict_logged_but_fact_kept()
    test_surname_matches_anchor()
    test_relevance_is_logged_not_used()
    test_writer_fail_open_writes_everything()
    test_writer_rejects_off_topic_without_writing()
    test_writer_writes_only_the_filtered_graph()
    print("\nALL POST-EXTRACTION GATE v2 TESTS PASSED (9/9)")