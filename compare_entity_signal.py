"""
MKGE Cluster-Memory Extension -- Entity Consistency Signal: Option 1 vs Option 2
============================================================
Standalone, READ-ONLY comparison script. Does NOT modify memory_gates.py
or cluster_memory.py -- same pattern as compare_clustering.py (the
Louvain vs SLPA decision): build the competing implementations here, run
them side by side against real data, decide, THEN (in a later, separate
step) fold the winner back into memory_gates.py's actual
_entity_consistency_signal.

WHAT THIS TESTS
----------------
Question_Memory_Gate.pdf section 6-7 identified that the CURRENT entity-
consistency signal (memory_gates.py's _entity_consistency_signal) divides
by the size of union(cluster's representative_entities, target node's
direct neighbours) -- meaning a perfectly relevant question gets
penalized purely because its cluster happens to be large. Two fixes are
compared here:

  Option 1 -- Tiered Match
      target node mentioned      -> 1.0
      direct neighbour mentioned -> 0.5
      neither                    -> 0.0

  Option 2 -- Weighted Target + Neighbour Coverage
      score = w_target * target_match + w_neighbour * neighbour_coverage
      Tested at both weightings the PDF proposed:
        2a: 0.5 / 0.5  (equal weight)
        2b: 0.7 / 0.3  (target-prioritized)

Both are compared against the CURRENT (baseline) implementation, on the
SAME real Neo4j graph, SAME real questions, SAME real semantic/accepted/
rejected signals -- only the entity-consistency term is swapped, so any
difference in the final combined score and gate decision is attributable
ONLY to the entity signal choice.

CLUSTER SELECTION -- deliberate, not arbitrary
------------------------------------------------
test_memory_gate.py picks the first 2 clusters in dict order. This
script instead picks the SMALLEST and LARGEST cluster (by member count):
the bug under test is specifically a cluster-SIZE penalty, so comparing
behaviour at the two ends of that spectrum is the whole point -- two
arbitrary same-ish-size clusters wouldn't exercise the thing being
tested.

NO LLM TIEBREAK CALLS
----------------------
This script labels every question x variant ACCEPT / REJECT / TIEBREAK
purely from where the deterministic combined_score falls relative to
T_ACCEPT/T_REJECT -- it never calls _llm_tiebreak. Running the real LLM
tiebreak for every question x every variant would burn Groq free-tier
quota for comparison purposes only, and the LLM's verdict on a given
question would be identical regardless of which entity-signal variant
produced the tiebreak-band score (the LLM only ever sees question +
cluster topic + entities, never the raw signal that triggered the
tiebreak) -- so it adds no comparative information here. TIEBREAK is
reported as its own outcome rather than guessed either way.

EXEMPLARS -- deliberately NOT seeded
--------------------------------------
Unlike the seeded version described in the manifesto, this script does
not seed synthetic AcceptedExemplar/RejectedExemplar records (matching
the current, real test_memory_gate.py, which also doesn't). Real
exemplar memory is genuinely empty at this stage (no gate has run for
real yet), so accepted/rejected signals are None for every question
here, renormalized out of _combine_signals entirely. That's actually
useful FOR THIS comparison: with accepted/rejected excluded, the
combined score is driven only by semantic + entity, which isolates
exactly how much the entity-signal choice moves the needle.

Run with:  python compare_entity_signal.py
(same .env as the rest of the pipeline)
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
from cluster_memory import build_cluster_memory_store
from memory_gates import (
    _semantic_similarity_signal,
    _entity_consistency_signal,
    _accepted_exemplar_signal,
    _rejected_exemplar_signal,
    _combine_signals,
    _get_direct_neighbors,
    T_ACCEPT,
    T_REJECT,
)
# Reused, not reimplemented -- same real-relation-grounded question
# builder test_memory_gate.py already wrote and validated.
from test_memory_gate import (
    _get_target_node_relation_facts,
    build_relevant_questions_from_relations,
)

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# =============================================================================
# Option 1 -- Tiered Match
# =============================================================================
def _entity_option1_tiered(question: str, node_id: str, node_label: str, driver) -> float:
    """
    target node mentioned -> 1.0
    direct neighbour mentioned (target not mentioned) -> 0.5
    neither -> 0.0

    Reuses the existing, unmodified _get_direct_neighbors from
    memory_gates.py -- same neighbour lookup the current implementation
    already uses; only the SCORING changes.
    """
    question_lower = question.lower()

    if node_label and node_label.lower() in question_lower:
        return 1.0

    neighbours = _get_direct_neighbors(driver, node_id)
    if any(n and n.lower() in question_lower for n in neighbours):
        return 0.5

    return 0.0


# =============================================================================
# Option 2 -- Weighted Target + Neighbour Coverage
# =============================================================================
def _entity_option2_weighted(question: str, node_id: str, node_label: str, driver,
                              w_target: float, w_neighbour: float) -> float:
    """
    score = w_target * target_match + w_neighbour * neighbour_coverage

    target_match is 0/1 (target label mentioned or not). neighbour_coverage
    is (# direct neighbours mentioned) / (# direct neighbours) -- same
    coverage IDEA as the current implementation, but the denominator is
    now just the target's own neighbour count, never the whole cluster --
    so it no longer shrinks as unrelated parts of the cluster grow.
    """
    question_lower = question.lower()
    target_match = 1.0 if node_label and node_label.lower() in question_lower else 0.0

    neighbours = _get_direct_neighbors(driver, node_id)
    if neighbours:
        matched = [n for n in neighbours if n and n.lower() in question_lower]
        neighbour_coverage = len(matched) / len(neighbours)
    else:
        neighbour_coverage = 0.0

    return w_target * target_match + w_neighbour * neighbour_coverage


ENTITY_VARIANT_FNS = {
    "current":  None,   # special-cased below -- calls the real, unmodified _entity_consistency_signal
    "option1":  _entity_option1_tiered,
    "option2a": lambda q, nid, nlabel, driver: _entity_option2_weighted(q, nid, nlabel, driver, 0.5, 0.5),
    "option2b": lambda q, nid, nlabel, driver: _entity_option2_weighted(q, nid, nlabel, driver, 0.7, 0.3),
}


def _decision_code(score: float) -> str:
    if score >= T_ACCEPT:
        return "ACC"
    if score <= T_REJECT:
        return "REJ"
    return "TIE"


# =============================================================================
# Test question set per target node
# =============================================================================
def build_test_questions(driver, node_id: str, node_label: str) -> list:
    """
    Returns [(tag, question)]:
      - RELEVANT: reuses test_memory_gate.py's real-relation-grounded
        builder unchanged (OWNS / RESIDES_IN / HAS_SECTOR when present).
      - NEIGHBOUR_ONLY: mentions a real direct neighbour but deliberately
        NOT the target label -- the exact motivating case from PDF
        section 6 (a legitimately-connected fact that shouldn't be
        penalized just for not repeating the target's own name).
      - IRRELEVANT: same two generic off-topic questions used elsewhere.
    """
    relation_facts = _get_target_node_relation_facts(driver, node_id)
    relevant = build_relevant_questions_from_relations(node_label, relation_facts)

    neighbours = _get_direct_neighbors(driver, node_id)
    other_neighbours = [n for n in neighbours if n and n.lower() != node_label.lower()]
    if other_neighbours:
        neighbour_label = other_neighbours[0]
        neighbour_only = [("NEIGHBOUR_ONLY", f"What is {neighbour_label} known for?")]
    else:
        neighbour_only = []

    irrelevant = [
        ("IRRELEVANT", "What is the boiling point of liquid nitrogen?"),
        ("IRRELEVANT", "Who won the most recent Eurovision Song Contest?"),
    ]

    return relevant + neighbour_only + irrelevant


# =============================================================================
# Per-question evaluation across all 4 entity-signal variants
# =============================================================================
def evaluate_question(question, node_id, node_label, cluster_memory,
                       cluster_member_node_ids, driver, embedder):
    question_embedding = embedder.embed_query(question)

    # Computed ONCE per question, reused for every variant below -- only
    # the entity term changes between variants.
    semantic = _semantic_similarity_signal(question_embedding, cluster_member_node_ids, driver)
    accepted = _accepted_exemplar_signal(question_embedding, cluster_memory, embedder)
    rejected = _rejected_exemplar_signal(question_embedding, cluster_memory, embedder)

    results = {}
    for variant_name, fn in ENTITY_VARIANT_FNS.items():
        if variant_name == "current":
            entity_score = _entity_consistency_signal(question, node_id, cluster_memory, driver)
        else:
            entity_score = fn(question, node_id, node_label, driver)

        combined = _combine_signals({
            "semantic": semantic,
            "entity": entity_score,
            "accepted": accepted,
            "rejected": rejected,
        })
        # entity_score can be None only for "current" on a degenerate
        # empty pool -- guard the printed row, don't crash the comparison.
        results[variant_name] = (entity_score, combined, _decision_code(combined))

    return results


# =============================================================================
# Reporting
# =============================================================================
def print_comparison_table(target_label, cluster_size, question_results):
    print(f"\n===== Target node: {target_label!r}  (cluster size = {cluster_size} member nodes) =====\n")
    print("Each cell: entity_signal_value / combined_score / decision (ACC=accept, REJ=reject, TIE=tiebreak-band)\n")

    header = f"{'tag':<14} {'question':<52} " + " ".join(f"{v:<18}" for v in ENTITY_VARIANT_FNS)
    print(header)
    print("-" * len(header))
    for tag, question, results in question_results:
        q_short = (question[:49] + "...") if len(question) > 52 else question
        row = f"{tag:<14} {q_short:<52} "
        for variant_name in ENTITY_VARIANT_FNS:
            entity_score, combined, code = results[variant_name]
            entity_str = f"{entity_score:.2f}" if entity_score is not None else "None"
            cell = f"{entity_str}/{combined:.2f}/{code}"
            row += f"{cell:<18} "
        print(row)
    print()


def print_tally(all_question_results):
    """
    For each variant: does it push RELEVANT questions toward ACC/TIE
    (good) or REJ (bad, false negative)? Does it push IRRELEVANT
    questions toward REJ (good) or ACC/TIE (bad, false positive)? Where
    does NEIGHBOUR_ONLY land -- the specific case Option 1/2 exist to fix.
    """
    print("=" * 70)
    print("SUMMARY TALLY")
    print("=" * 70)

    for variant_name in ENTITY_VARIANT_FNS:
        relevant_rejected = 0
        relevant_total = 0
        irrelevant_not_rejected = 0
        irrelevant_total = 0
        neighbour_only_codes = []

        for tag, question, results in all_question_results:
            _, _, code = results[variant_name]
            if tag == "RELEVANT":
                relevant_total += 1
                if code == "REJ":
                    relevant_rejected += 1
            elif tag == "IRRELEVANT":
                irrelevant_total += 1
                if code in ("ACC", "TIE"):
                    irrelevant_not_rejected += 1
            elif tag == "NEIGHBOUR_ONLY":
                neighbour_only_codes.append(code)

        print(f"\n[{variant_name}]")
        print(f"  RELEVANT questions incorrectly REJECTed      : {relevant_rejected}/{relevant_total}")
        print(f"  IRRELEVANT questions NOT rejected (ACC/TIE)  : {irrelevant_not_rejected}/{irrelevant_total}")
        print(f"  NEIGHBOUR_ONLY question decision(s)          : {neighbour_only_codes or '(none found)'}")

    print()


# =============================================================================
# Main
# =============================================================================
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

    # Real llm_client -> one Groq call PER CLUSTER for topic_summary
    # (cheap, one-time), same as test_memory_gate.py.
    store = build_cluster_memory_store(driver, node_cluster_map, G, llm_client=client)
    print(f"[Test] Built Cluster Memory Store with {len(store)} cluster(s).\n")

    labels = nx.get_node_attributes(G, "label")

    clusters_to_nodes = {}
    for node_id, cluster_id in node_cluster_map.items():
        clusters_to_nodes.setdefault(cluster_id, []).append(node_id)

    cluster_ids = sorted(store.all_cluster_ids())
    if not cluster_ids:
        print("[Test] No clusters found -- nothing to compare against.")
        driver.close()
        return

    # ---- Deliberately pick SMALLEST and LARGEST cluster by member
    # ---- count. See module docstring for why (cluster-size penalty is
    # ---- exactly what's under test here).
    sizes = {cid: len(clusters_to_nodes[cid]) for cid in cluster_ids}
    smallest_cid = min(sizes, key=sizes.get)
    largest_cid = max(sizes, key=sizes.get)
    target_cids = [smallest_cid] if smallest_cid == largest_cid else [smallest_cid, largest_cid]

    print(f"[Test] Cluster sizes: {sizes}")
    print(f"[Test] Comparing on smallest cluster (id={smallest_cid}, "
          f"size={sizes[smallest_cid]}) and largest cluster "
          f"(id={largest_cid}, size={sizes[largest_cid]}).\n")

    all_question_results = []

    for cid in target_cids:
        member_node_ids = clusters_to_nodes[cid]
        target_node_id = member_node_ids[0]
        target_label = labels.get(target_node_id, target_node_id)
        cluster_memory = store.get(cid)

        questions = build_test_questions(driver, target_node_id, target_label)

        question_results = []
        for tag, question in questions:
            results = evaluate_question(
                question, target_node_id, target_label, cluster_memory,
                member_node_ids, driver, embedder,
            )
            question_results.append((tag, question, results))
            all_question_results.append((tag, question, results))

        print_comparison_table(target_label, len(member_node_ids), question_results)

    print_tally(all_question_results)

    print("[Test] Done. Read SUMMARY TALLY first: the variant with the fewest "
          "incorrectly-REJECTed RELEVANT questions and fewest wrongly-accepted "
          "IRRELEVANT ones, on BOTH the smallest and largest cluster, is the "
          "stronger choice. Pay special attention to NEIGHBOUR_ONLY -- that's "
          "the exact case (a real, connected fact that doesn't repeat the "
          "target's own name) the current baseline is known to under-score as "
          "clusters grow; whichever option lands it closer to ACC/TIE rather "
          "than REJ is doing its job.\n"
          "Reminder: entity is only W_ENTITY=0.30 of the combined score in "
          "production, so a big swing in the raw entity value may still only "
          "shift combined_score modestly -- that gap itself is useful evidence "
          "for whether W_ENTITY also needs retuning later, separately from "
          "which option wins.")

    driver.close()


if __name__ == "__main__":
    main()