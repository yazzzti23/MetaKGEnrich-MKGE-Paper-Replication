"""
Standalone test for Phase 3a (memory_gates.py) -- NOT part of the pipeline.

Loads the current graph from Neo4j, builds the node->cluster mapping and
Cluster Memory Store exactly like test_cluster_memory.py already does,
then picks a couple of real (node_id, cluster) targets and runs
question_memory_gate() against a handful of hand-written questions --
some clearly on-topic for that node's cluster, some clearly off-topic --
using your REAL Neo4j graph, REAL all-MiniLM-L6-v2 embedder, and REAL
Groq-backed LLM client (for the tiebreak path only, if any question
happens to land in the T_REJECT < score < T_ACCEPT band).

Does NOT modify Stage 3 or Stage 4, does NOT write anything to
cluster_memory (question_memory_gate is a pure evaluator, per spec) --
this script itself also never calls add_accepted/add_rejected, since
that's Stage 4's responsibility, not the gate's.

Run with:  python test_memory_gate.py
(same .env as the rest of the pipeline -- needs GROQ_API_KEY since this
uses a real llm_client for build_cluster_memory_store's summaries AND
as the potential tiebreak client)
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
import networkx as nx

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

from stage2_sparsity_metrics import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    load_entity_graph,
    get_node_cluster_mapping,
)
from cluster_memory import build_cluster_memory_store, compute_cluster_embedding
from memory_gates import question_memory_gate, _cosine_similarity

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# Direction-aware phrasing per relation type, based on the actual
# relation vocabulary this pipeline extracts (see GENERIC_RELATIONS in
# stage1_seed_kg.py / stage4_retrieval_ingestion.py) and the real
# subject->object direction each relation type actually encodes in the
# graph (e.g. confirmed: person --RESIDES_IN--> city, not the reverse).
#
# Each builder takes (target_label, other_label, direction) where
# direction is "outgoing" (target_label --REL--> other_label, i.e. the
# target node IS the subject/source of the real edge) or "incoming"
# (other_label --REL--> target_label, i.e. the target node IS the
# object/destination of the real edge). This lets the generated question
# always keep the grammatically/semantically correct entity as the
# subject, regardless of which side of the edge the test's arbitrarily-
# picked target node happens to sit on.
def _question_owns(target_label, other_label, direction):
    if direction == "outgoing":
        # target_label --OWNS--> other_label  (target is the owner)
        return f"What company does {target_label} own, and how is {other_label} connected to it?"
    else:
        # other_label --OWNS--> target_label  (target is the owned entity)
        return f"Who owns {target_label}, and how is {other_label} connected to it?"


def _question_resides_in(target_label, other_label, direction):
    if direction == "outgoing":
        # target_label --RESIDES_IN--> other_label  (target is the person)
        return f"What city does {target_label} reside in?"
    else:
        # other_label --RESIDES_IN--> target_label  (target is the city)
        return f"Which person resides in {target_label}, and how is {other_label} connected to it?"


def _question_has_sector(target_label, other_label, direction):
    if direction == "outgoing":
        # target_label --HAS_SECTOR--> other_label  (target is the company/person)
        return f"What sector or industry is {target_label} associated with?"
    else:
        # other_label --HAS_SECTOR--> target_label  (target is the sector)
        return f"Which company or person is associated with the {target_label} sector, via {other_label}?"


def _question_located_in(target_label, other_label, direction):
    if direction == "outgoing":
        return f"Where is {target_label} located?"
    else:
        return f"What is located in {target_label}, and how is {other_label} connected to it?"


def _question_part_of(target_label, other_label, direction):
    if direction == "outgoing":
        return f"What is {target_label} part of?"
    else:
        return f"What is part of {target_label}, and how is {other_label} connected to it?"


def _question_associated_with(target_label, other_label, direction):
    # Symmetric-feeling relation type; direction still respected so the
    # subject named first always matches the real edge's source.
    if direction == "outgoing":
        return f"How is {target_label} associated with {other_label}?"
    else:
        return f"How is {other_label} associated with {target_label}?"


def _question_generic_fallback(target_label, other_label, direction, rel_type):
    # Unknown relation type -- stays direction-agnostic ("relationship
    # between") since we don't know the real subject/object semantics
    # of an arbitrary relation type, rather than guessing a direction.
    return f"What is the relationship between {target_label} and {other_label} ({rel_type})?"


RELATION_QUESTION_BUILDERS = {
    "OWNS": _question_owns,
    "RESIDES_IN": _question_resides_in,
    "HAS_SECTOR": _question_has_sector,
    "LOCATED_IN": _question_located_in,
    "PART_OF": _question_part_of,
    "ASSOCIATED_WITH": _question_associated_with,
}


def _get_target_node_relation_facts(driver, node_id: str) -> list:
    """
    Pulls the ACTUAL relation triples connecting the target node to its
    direct 1-hop neighbors in Neo4j, WITH DIRECTION -- so the test's
    "relevant" questions can respect the real subject->object direction
    of each relation (e.g. person --RESIDES_IN--> city) instead of
    treating every edge as undirected and risking a backwards/nonsense
    question like "Where does Ahmedabad reside?".

    Two directed queries (outgoing edges from the target node, incoming
    edges into it) are UNIONed rather than using an undirected pattern,
    specifically to recover which side of each real edge the target
    node sits on.

    Returns
    -------
    list[tuple[str, str, str]]
        (rel_type, other_label, direction) triples, one per direct
        relationship. direction is "outgoing" if
        target_node --rel_type--> other, or "incoming" if
        other --rel_type--> target_node. Order matches whatever Neo4j
        returns -- callers pick preferred rel_types out of this list
        themselves.
    """
    facts = []
    with driver.session() as session:
        result = session.run(
            "MATCH (n:__Entity__)-[r]->(m:__Entity__) "
            "WHERE elementId(n) = $node_id "
            "RETURN DISTINCT type(r) AS rel_type, "
            "coalesce(m.name, m.id, 'unnamed') AS other_label, "
            "'outgoing' AS direction "
            "UNION "
            "MATCH (n:__Entity__)<-[r]-(m:__Entity__) "
            "WHERE elementId(n) = $node_id "
            "RETURN DISTINCT type(r) AS rel_type, "
            "coalesce(m.name, m.id, 'unnamed') AS other_label, "
            "'incoming' AS direction",
            node_id=node_id,
        )
        for record in result:
            facts.append((record["rel_type"], record["other_label"], record["direction"]))
    return facts


def build_relevant_questions_from_relations(target_label: str, relation_facts: list) -> list:
    """
    Builds up to 2 "RELEVANT" test questions directly from the target
    node's REAL, DIRECTED relation facts (rel_type, other_label,
    direction triples pulled from Neo4j), preferring the relation types
    the manifesto specifically called out as known graph facts worth
    testing against: OWNS, RESIDES_IN, and HAS_SECTOR (in that priority
    order), before falling back to whatever other relation types are
    actually present.

    Each question is phrased via RELATION_QUESTION_BUILDERS, which picks
    the grammatically/semantically correct subject based on `direction`
    -- e.g. for RESIDES_IN, the PERSON is always the subject ("What city
    does X reside in?"), never the city, regardless of whether the
    target node happens to be the person or the city side of the edge.

    Falls back to a single generic (still real-neighbor-grounded)
    question only if the target node has no direct relations at all --
    which would itself be worth noticing (a genuinely isolated node).
    """
    if not relation_facts:
        return [("RELEVANT", f"What is {target_label} known for?")]

    facts_by_type = {}
    for rel_type, other_label, direction in relation_facts:
        facts_by_type.setdefault(rel_type, (other_label, direction))  # first match per type

    preferred_order = ["OWNS", "RESIDES_IN", "HAS_SECTOR"]
    ordered_types = [t for t in preferred_order if t in facts_by_type]
    ordered_types += [t for t in facts_by_type if t not in ordered_types]

    questions = []
    for rel_type in ordered_types[:2]:
        other_label, direction = facts_by_type[rel_type]
        builder = RELATION_QUESTION_BUILDERS.get(rel_type)
        if builder is not None:
            question_text = builder(target_label, other_label, direction)
        else:
            question_text = _question_generic_fallback(target_label, other_label, direction, rel_type)
        questions.append(("RELEVANT", question_text))

    return questions



def _find_cluster_id_by_entity_substring(clusters_to_nodes: dict, labels: dict, substrings: list) -> int:
    """
    Diagnostic-only helper: scans member node labels within each cluster
    for a case-insensitive substring match (e.g. "Ambani" or "Adani"),
    and returns the first cluster_id whose members include a match.
    Read-only -- does not touch Neo4j, just inspects the already-loaded
    NetworkX label mapping and the cluster_id -> [member_node_ids] dict
    already built in main().

    Returns None if no cluster's members match any of the substrings.
    """
    for cluster_id, member_node_ids in clusters_to_nodes.items():
        for node_id in member_node_ids:
            label = str(labels.get(node_id, "")).lower()
            if any(s.lower() in label for s in substrings):
                return cluster_id
    return None


def run_chunk_embedding_diagnostic(store, clusters_to_nodes: dict, labels: dict, driver) -> None:
    """
    DIAGNOSTIC ONLY -- does not touch memory_gates.py or cluster_memory.py,
    calls only the existing, unmodified compute_cluster_embedding() and
    the existing, unmodified _cosine_similarity() from memory_gates.py.
    Read-only Neo4j queries only (MATCH ... RETURN, no writes).

    Verifies (or refutes) the hypothesis raised during signal inspection:
    that, at this stage of the pipeline (single-snippet Stage 1 seed KG,
    no Stage 4 enrichment yet run), different clusters' member nodes may
    all be FROM_CHUNK-linked to the SAME underlying Chunk node -- which
    would mean compute_cluster_embedding() returns an identical (or
    near-identical) vector for every cluster right now, regardless of
    the cluster's actual topic.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Chunk-embedding sharing between clusters")
    print("=" * 70)

    reliance_cid = _find_cluster_id_by_entity_substring(
        clusters_to_nodes, labels, ["Ambani", "Reliance"]
    )
    adani_cid = _find_cluster_id_by_entity_substring(
        clusters_to_nodes, labels, ["Adani"]
    )

    if reliance_cid is None or adani_cid is None:
        print(f"[Diagnostic] Could not locate both target clusters "
              f"(Reliance/Ambani cluster_id={reliance_cid}, "
              f"Adani cluster_id={adani_cid}). Skipping diagnostic.")
        return

    cluster_specs = [
        ("Reliance/Ambani", reliance_cid, clusters_to_nodes[reliance_cid]),
        ("Adani Group", adani_cid, clusters_to_nodes[adani_cid]),
    ]

    embeddings = {}
    chunk_id_sets = {}

    for tag, cluster_id, member_node_ids in cluster_specs:
        member_labels = [labels.get(n, n) for n in member_node_ids]

        # ---- Step 2: call the EXISTING, unmodified compute_cluster_embedding
        embedding = compute_cluster_embedding(driver, member_node_ids)
        embeddings[cluster_id] = embedding

        has_embedding = embedding is not None and len(embedding) > 0
        dim = len(embedding) if has_embedding else 0
        first5 = list(embedding[:5]) if has_embedding else []

        print(f"\n--- [{tag}] cluster_id={cluster_id} ---")
        print(f"  member node labels : {member_labels}")
        print(f"  embedding exists   : {has_embedding}")
        print(f"  embedding dim      : {dim}")
        print(f"  first 5 values     : {first5}")

        # ---- Step 5: READ-ONLY query -- distinct Chunk nodes connected
        # ---- via FROM_CHUNK to this cluster's member nodes.
        with driver.session() as session:
            result = session.run(
                "MATCH (n:__Entity__)-[:FROM_CHUNK]-(c:Chunk) "
                "WHERE elementId(n) IN $node_ids "
                "RETURN DISTINCT elementId(c) AS chunk_id",
                node_ids=member_node_ids,
            )
            chunk_ids = [record["chunk_id"] for record in result]
        chunk_id_sets[cluster_id] = set(chunk_ids)

        print(f"  distinct Chunk count: {len(chunk_ids)}")
        print(f"  Chunk elementIds    : {chunk_ids}")

    # ---- Step 4: cosine similarity between the two cluster embeddings,
    # ---- using the EXISTING, unmodified _cosine_similarity from
    # ---- memory_gates.py (no reimplementation here).
    emb_a = embeddings[reliance_cid]
    emb_b = embeddings[adani_cid]
    if emb_a is not None and emb_b is not None and len(emb_a) > 0 and len(emb_b) > 0:
        similarity = _cosine_similarity(emb_a, emb_b)
        print(f"\n[Diagnostic] Cosine similarity between the two cluster "
              f"embeddings: {similarity:.6f}")
    else:
        similarity = None
        print("\n[Diagnostic] Cannot compute cosine similarity -- "
              "one or both cluster embeddings are missing/empty.")

    # ---- Step 6: clear verdict on shared vs. distinct Chunk usage.
    chunks_a = chunk_id_sets[reliance_cid]
    chunks_b = chunk_id_sets[adani_cid]

    print(f"\n[Diagnostic] Reliance/Ambani Chunk set: {chunks_a}")
    print(f"[Diagnostic] Adani Group Chunk set     : {chunks_b}")

    if not chunks_a or not chunks_b:
        verdict = "Unable to determine"
    elif chunks_a == chunks_b:
        verdict = "The two clusters use the same Chunk embedding"
    elif chunks_a.isdisjoint(chunks_b):
        verdict = "The two clusters use different Chunk embeddings"
    else:
        verdict = "Unable to determine"  # partial overlap -- ambiguous

    print(f"\n[Diagnostic] VERDICT: {verdict}")
    print("=" * 70 + "\n")


def print_result(question: str, result) -> None:
    """_log_gate_decision (inside question_memory_gate) already prints
    the raw signals + combined score + decision + path to stdout. This
    just adds a compact, structured summary line afterward so results
    are easy to scan/compare across questions at a glance.
    """
    print(f"    >>> SUMMARY: decision={result.decision!r}  "
          f"confidence={result.confidence_score:.4f}  "
          f"path={result.decision_path!r}  "
          f"attempt={result.attempt_number}")
    print(f"    >>> signals: {result.signals}")
    print()


def run_questions_against_node(label, questions, node_id, node_label, cluster_memory,
                                cluster_member_node_ids, driver, embedder, client):
    print(f"\n===== Target node: {node_label!r} (cluster has "
          f"{len(cluster_member_node_ids)} member node(s)) =====")
    print(f"Cluster topic_summary: {cluster_memory.representative.topic_summary}")
    print(f"Cluster entities: {cluster_memory.representative.representative_entities}\n")

    for tag, question in questions:
        print(f"--- [{tag}] question: {question}")
        result = question_memory_gate(
            question=question,
            node_id=node_id,
            node_label=node_label,
            cluster_memory=cluster_memory,
            cluster_member_node_ids=cluster_member_node_ids,
            driver=driver,
            embedder=embedder,
            llm_client=client,
            attempt_number=1,
        )
        print_result(question, result)


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    G = load_entity_graph(driver)
    print(f"[Test] Loaded entity graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges.\n")

    if G.number_of_nodes() == 0:
        print("No __Entity__ nodes found -- did Stage 1 actually run first?")
        driver.close()
        return

    node_cluster_map = get_node_cluster_mapping(G)

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

    # Real llm_client passed here -> build_cluster_memory_store makes one
    # Groq call PER CLUSTER for topic_summary/keywords (cheap, one-time,
    # matches how Stage 4's planned integration will actually call it).
    store = build_cluster_memory_store(driver, node_cluster_map, G, llm_client=client)
    print(f"[Test] Built Cluster Memory Store with {len(store)} cluster(s).\n")

    labels = nx.get_node_attributes(G, "label")

    # Build cluster_id -> [member_node_ids], same dict shape Stage 4's
    # planned integration builds as its own first step (see manifesto).
    clusters_to_nodes = {}
    for node_id, cluster_id in node_cluster_map.items():
        clusters_to_nodes.setdefault(cluster_id, []).append(node_id)

    cluster_ids = sorted(store.all_cluster_ids())
    if not cluster_ids:
        print("[Test] No clusters found -- nothing to test the gate against.")
        driver.close()
        return

    # ---- Diagnostic (read-only): check whether the Reliance/Ambani and
    # ---- Adani Group clusters currently resolve to the same underlying
    # ---- Chunk embedding. Runs before the existing gate tests below;
    # ---- does not affect them.
    run_chunk_embedding_diagnostic(store, clusters_to_nodes, labels, driver)

    # ---- Pick up to 2 real clusters to test against, so results aren't
    # ---- a fluke of one particular cluster's shape. Each cluster picks
    # ---- ONE representative member node as the "target sparse node".
    num_targets = min(2, len(cluster_ids))
    targets = []
    for cid in cluster_ids[:num_targets]:
        member_node_ids = clusters_to_nodes[cid]
        target_node_id = member_node_ids[0]
        target_label = labels.get(target_node_id, target_node_id)
        targets.append((cid, target_node_id, target_label, member_node_ids))

    for cid, target_node_id, target_label, member_node_ids in targets:
        cluster_memory = store.get(cid)

        # ---- Clearly RELEVANT questions: built directly from the
        # ---- target node's REAL relation facts (OWNS / RESIDES_IN /
        # ---- HAS_SECTOR, when present), not a guessed same-cluster
        # ---- pairing -- see build_relevant_questions_from_relations().
        relation_facts = _get_target_node_relation_facts(driver, target_node_id)
        print(f"[Test] Real relation facts (rel_type, other_label, direction) "
              f"for {target_label!r}: {relation_facts}")
        relevant_questions = build_relevant_questions_from_relations(target_label, relation_facts)

        # ---- Clearly IRRELEVANT questions: generic, off-topic content
        # ---- that shares no entities/keywords with this cluster at all
        # ---- (deliberately mundane, unrelated domains).
        irrelevant_questions = [
            ("IRRELEVANT", "What is the boiling point of liquid nitrogen?"),
            ("IRRELEVANT", "Who won the most recent Eurovision Song Contest?"),
        ]

        run_questions_against_node(
            label=cid,
            questions=relevant_questions + irrelevant_questions,
            node_id=target_node_id,
            node_label=target_label,
            cluster_memory=cluster_memory,
            cluster_member_node_ids=member_node_ids,
            driver=driver,
            embedder=embedder,
            client=client,
        )

    print("\n[Test] Done. Inspect the SUMMARY lines above: RELEVANT questions "
          "should generally score closer to/above T_ACCEPT (0.70) and "
          "IRRELEVANT ones closer to/below T_REJECT (0.35), keeping in mind "
          "every cluster here is cold-start on accepted/rejected exemplars "
          "(no gate has ever run before), so only the semantic + entity "
          "signals will carry weight this first time.")

    driver.close()


if __name__ == "__main__":
    main()