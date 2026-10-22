"""
MKGE -- OUR PIPELINE -- Stage 4: Retrieval + Ingestion (memory-gated)
============================================================
Our improved version of Stage 4. The original MKGE baseline (run A) is
stage4_retrieval_inges.py in the original pipeline, untouched.

Loop structure (the Algorithm 1 fix, unchanged): for each metric,
reload the graph -> recompute sparsity -> generate THAT metric's
questions -> gate -> search -> extract -> gate -> write, before moving to
the next metric, so later metrics see the graph enriched by earlier ones.

Changes vs the previous gated Stage 4:
  [imports] Uses THIS folder's files: stage2_sparsity_metrics1,
            stage3_question_generation2, stage1_seed_kg1.
  [F2] Clusters come from the seed (get_or_create_seed_clusters), not a
       fresh Louvain run, so every run judges against the same clusters.
       Clusters are printed and logged by member names.
  [R1] Retrieval picks the best of NUM_CANDIDATES Tavily results by
       similarity to the question (select_best_snippet from Stage 1).
       SELECTION_MODE = "top1" reproduces the old behaviour for the
       "ours minus R1" ablation run.
  [F3] One QuestionHistory for the whole run is passed to question
       generation and regeneration, so duplicates are dropped and
       already-asked nodes are deprioritised across ALL metrics.
       Stage 3's sparse_previews_for_metric (with the [F4] node-centred
       preview) replaces Stage 4's old local copy.
  [F5 part 1] Counters: how each target node got its cluster (seed /
       assigned by embedding / fail-open) and how many writes skipped
       both gates. Printed at the end and saved in the run log.
  [run log] One JSON record per question (question, both gates' scores
       and who decided, search candidates, post-gate result, outcome),
       saved after EVERY question to runs/<timestamp>_<RUN_TAG>.json so a
       crash mid-run (e.g. Groq quota) still keeps everything so far.
  [S5] The Groq client retries automatically on rate-limit errors
       (max_retries). The 40 s wait between extractions is unchanged.
  [S6] Cluster membership comes from cluster_store.members(), the one
       register, instead of a second hand-maintained clusters_to_nodes.
  [PG2] Post-Extraction Gate v2 (per entity / per fact, see
       post_extraction_writer.py): Stage 4 now passes the question to the
       writer, logs every extracted fact with its status (kept / duplicate
       / dropped_unconnected, plus relevance and conflicts) in the run log,
       and adds fact-level totals to the counters.

Run with (from the project root):
    python newpipeline\\stage4_retrieval_ingestion2.py
"""

import asyncio
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from tavily import TavilyClient

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

from stage1_seed_kg1 import select_best_snippet                        # [R1]
from stage2_sparsity_metrics1 import (                                  # [imports]
    METRIC_NAMES,
    load_entity_graph,
    get_or_create_seed_clusters,                                        # [F2]
    describe_clusters,                                                  # [F2]
)
from stage3_question_generation2 import (
    generate_questions_for_metric,
    regenerate_question_for_node,
    sparse_previews_for_metric,                                         # [F3/F4]
    QuestionHistory,                                                    # [F3]
)
from cluster_memory import (
    build_cluster_memory_store,
    compute_cluster_embedding,
    AcceptedExemplar,
    RejectedExemplar,
)
from memory_gates import question_memory_gate
from post_extraction_writer import GatedKGWriter

load_dotenv()

GENERIC_ENTITIES = ["Person", "Organization", "Location", "Work", "Event", "Concept", "Date"]
GENERIC_RELATIONS = ["RELATED_TO", "LOCATED_IN", "PART_OF", "CREATED_BY",
                     "OWNS", "CAUSED_BY", "OCCURRED_ON", "ASSOCIATED_WITH"]

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# [R1] Retrieval selection.
NUM_CANDIDATES = 3
SELECTION_MODE = "best"   # "top1" = ablation run "ours minus R1"

# [run log] Name of this run in the log file name ("ours", "ours_minus_r1", ...).
RUN_TAG = "ours"
RUNS_DIR = "runs"

# [S5] Automatic retries (with backoff) on Groq rate-limit / transient errors.
GROQ_MAX_RETRIES = 5


def get_top_snippet(tavily: TavilyClient, question: str, embedder):
    """[R1] Search Tavily, then pick ONE snippet with select_best_snippet.
    Returns (snippet or None, retrieval_log)."""
    try:
        response = tavily.search(question, max_results=NUM_CANDIDATES)
        results = response.get("results", [])
    except (IndexError, KeyError):
        results = []
    return select_best_snippet(question, results, embedder, mode=SELECTION_MODE)


def _gate_record(result) -> dict:
    """GateResult -> plain dict for the run log."""
    if result is None:
        return None
    return {
        "decision": result.decision,
        "score": round(result.confidence_score, 4),
        "path": result.decision_path,
        "signals": result.signals,
    }


def _save_run_log(path: str, run_log: dict) -> None:
    """[run log] Rewrites the whole log file. Called after every question."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False, default=str)


async def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    with driver.session() as session:
        before_total = session.run("MATCH (n:__Entity__) RETURN count(n) AS c").single()["c"]
    print(f"[Stage 4] Entity nodes BEFORE enrichment: {before_total}\n")

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
                    max_retries=GROQ_MAX_RETRIES)                       # [S5]
    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    llm = OpenAILLM(
        model_name="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model_params={"temperature": 0, "response_format": {"type": "json_object"}},
    )
    llm.supports_structured_output = False
    embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
    splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)

    # ---- Post-Extraction Memory Gate wiring (unchanged) ----
    # gated_writer intercepts SimpleKGPipeline's extraction result BEFORE
    # it is written to Neo4j (see post_extraction_writer.py).
    gated_writer = GatedKGWriter(driver=driver, embedder=embedder)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        text_splitter=splitter,
        entities=GENERIC_ENTITIES,
        relations=GENERIC_RELATIONS,
        from_pdf=False,
        kg_writer=gated_writer,
    )

    # ---- [F2] Cluster Memory Store: built ONCE from the SAVED seed clusters.
    cluster_G = load_entity_graph(driver)
    node_cluster_map = dict(get_or_create_seed_clusters(driver, cluster_G))
    cluster_store = build_cluster_memory_store(driver, node_cluster_map, cluster_G, llm_client=client)

    # [S6] Cluster membership is read from the store (cluster_store.members),
    # the ONE register -- no separate clusters_to_nodes copy any more.

    clusters_by_name = describe_clusters(cluster_G, node_cluster_map)
    print(f"[Stage 4] Cluster Memory Store built once: {len(cluster_store)} cluster(s), "
          f"{len(node_cluster_map)} node(s) clustered.")
    for cluster_id, members in clusters_by_name.items():
        print(f"    cluster {cluster_id}: {', '.join(members)}")
    print()

    # ---- [F3] One question history for the whole run.
    history = QuestionHistory()

    # ---- [F5 part 1] Counters.
    counts = {
        "questions": 0,
        "cluster_from_seed": 0,
        "cluster_assigned_by_embedding": 0,
        "fail_open": 0,
        "question_gate_accept_first_try": 0,
        "question_gate_accept_after_regen": 0,
        "question_gate_reject": 0,
        "no_search_result": 0,
        "post_gate_accept": 0,
        "post_gate_reject": 0,
        "written_without_any_gate": 0,
        "ingested": 0,
        # [PG2] fact-level totals across all gated extractions
        "facts_extracted": 0,
        "facts_kept": 0,
        "facts_duplicate": 0,
        "facts_dropped_unconnected": 0,
        "fact_conflicts": 0,
    }

    # ---- [run log] Set up the log file.
    os.makedirs(RUNS_DIR, exist_ok=True)
    started = datetime.now()
    log_path = os.path.join(RUNS_DIR, f"{started.strftime('%Y-%m-%d_%H%M%S')}_{RUN_TAG}.json")
    run_log = {
        "run_tag": RUN_TAG,
        "started_at": started.isoformat(timespec="seconds"),
        "settings": {
            "selection_mode": SELECTION_MODE,
            "num_candidates": NUM_CANDIDATES,
            "metrics": METRIC_NAMES,
        },
        "entity_nodes_before": before_total,
        "clusters": clusters_by_name,
        "questions": [],
        "counts": counts,
    }
    _save_run_log(log_path, run_log)
    print(f"[Stage 4] Run log: {log_path}\n")

    # ---- One metric at a time: detect -> question -> ingest (Algorithm 1).
    for metric in METRIC_NAMES:
        print(f"[Stage 4] === Metric cycle: {metric} ===")

        sparse_previews = sparse_previews_for_metric(driver, metric)               # [F3/F4]
        print(f"[Stage 4] ({metric}) {len(sparse_previews)} sparse node(s) on current graph.")

        questions = generate_questions_for_metric(client, metric, sparse_previews,
                                                  embedder=embedder, history=history)  # [F3]
        for node_id, node_label, question in questions:
            print(f"    - [{node_label}] {question}")

        for node_id, node_label, question in questions:
            print(f"[Stage 4] Target node: {node_label}")
            print(f"[Stage 4] Question: {question}")
            counts["questions"] += 1

            record = {
                "metric": metric,
                "target_node": node_label,
                "target_node_id": node_id,
                "question": question,
                "cluster_resolution": None,
                "cluster_id": None,
                "question_gate_1": None,
                "regenerated_question": None,
                "question_gate_2": None,
                "final_question": None,
                "retrieval": None,
                "post_gate": None,
                "outcome": None,
            }
            run_log["questions"].append(record)

            # ---- Which cluster? (seed -> assign by embedding -> fail-open)
            cluster_id = node_cluster_map.get(node_id)
            if cluster_id is not None:
                record["cluster_resolution"] = "seed"
                counts["cluster_from_seed"] += 1
            else:
                # Node added earlier in THIS run: try nearest-cluster assignment.
                node_embedding = compute_cluster_embedding(driver, [node_id])
                assigned_cluster_id = (
                    cluster_store.assign_new_node(node_id, node_embedding, driver)
                    if node_embedding is not None else None
                )
                if assigned_cluster_id is not None:
                    print(f"    [Gate] Assigned previously-unclustered node to cluster {assigned_cluster_id}.")
                    node_cluster_map[node_id] = assigned_cluster_id
                    # [S6] assign_new_node() already added it to the store's register.
                    cluster_id = assigned_cluster_id
                    record["cluster_resolution"] = "assigned_by_embedding"
                    counts["cluster_assigned_by_embedding"] += 1
            record["cluster_id"] = cluster_id

            if cluster_id is None:
                # True fail-open: no cluster could be found -- no gates.
                print("    [Gate] No cluster assignment for this node -- fail-open, skipping gate.")
                record["cluster_resolution"] = "fail_open"
                counts["fail_open"] += 1
                final_question = question
                gate_cluster_memory = None
                member_node_ids = None
                accept_result = None
            else:
                cluster_memory = cluster_store.get(cluster_id)
                member_node_ids = cluster_store.members(cluster_id)        # [S6]

                result = question_memory_gate(
                    question=question,
                    node_id=node_id,
                    node_label=node_label,
                    cluster_memory=cluster_memory,
                    cluster_member_node_ids=member_node_ids,
                    driver=driver,
                    embedder=embedder,
                    llm_client=client,
                    attempt_number=1,
                )
                record["question_gate_1"] = _gate_record(result)

                if result.decision == "accept":
                    counts["question_gate_accept_first_try"] += 1
                    final_question = question
                    gate_cluster_memory = cluster_memory
                    accept_result = result
                else:
                    print(f"    [Gate] REJECTED (attempt 1, {result.decision_path}, "
                          f"score={result.confidence_score:.4f}) -- regenerating once.")

                    regen_node_id, regen_node_label, regen_question = regenerate_question_for_node(
                        client, node_id, node_label, metric, driver, rejected_question=question,
                        embedder=embedder, history=history,                     # [F3]
                    )
                    print(f"    [Gate] Regenerated question: {regen_question}")
                    record["regenerated_question"] = regen_question

                    result2 = question_memory_gate(
                        question=regen_question,
                        node_id=regen_node_id,
                        node_label=regen_node_label,
                        cluster_memory=cluster_memory,
                        cluster_member_node_ids=member_node_ids,
                        driver=driver,
                        embedder=embedder,
                        llm_client=client,
                        attempt_number=2,
                    )
                    record["question_gate_2"] = _gate_record(result2)

                    if result2.decision == "reject":
                        # Maximum 2 TOTAL attempts -- skip retrieval for this
                        # node this cycle and record a RejectedExemplar.
                        print(f"    [Gate] REJECTED again (attempt 2, {result2.decision_path}, "
                              f"score={result2.confidence_score:.4f}) -- skipping retrieval.")
                        cluster_memory.exemplars.add_rejected(RejectedExemplar(
                            sparse_node=regen_node_label,
                            generated_question=regen_question,
                            failure_type=f"question_gate_reject_{result2.decision_path}",
                            rejection_reason=(
                                f"combined_score={result2.confidence_score:.4f} did not clear "
                                f"T_ACCEPT after 1 regeneration; signals={result2.signals}"
                            ),
                            confidence_score=result2.confidence_score,
                        ))
                        counts["question_gate_reject"] += 1
                        record["outcome"] = "skipped_by_question_gate"
                        _save_run_log(log_path, run_log)
                        continue  # next question -- no retrieval for this node this cycle

                    counts["question_gate_accept_after_regen"] += 1
                    final_question = regen_question
                    gate_cluster_memory = cluster_memory
                    accept_result = result2

            record["final_question"] = final_question

            # ---- [R1] Retrieval: best of NUM_CANDIDATES results.
            snippet, retrieval_log = get_top_snippet(tavily, final_question, embedder)
            record["retrieval"] = retrieval_log
            if not snippet:
                print("    -> no result found, skipping")
                counts["no_search_result"] += 1
                record["outcome"] = "skipped_no_search_result"
                _save_run_log(log_path, run_log)
                continue
            chosen = retrieval_log["chosen_index"]
            print(f"    [R1] Chose result #{chosen} of {len(retrieval_log['candidates'])} "
                  f"({retrieval_log['candidates'][chosen].get('url')})")

            # ---- Post-Extraction Memory Gate context (unchanged).
            gated_writer.set_gate_context(
                node_label=node_label,
                cluster_memory=gate_cluster_memory,
                cluster_member_node_ids=member_node_ids if gate_cluster_memory is not None else None,
                question=final_question,                                    # [PG2]
            )

            await kg_builder.run_async(text=snippet)
            post_result = gated_writer.last_result

            if post_result is not None:
                record["post_gate"] = {
                    "decision": post_result.decision,
                    "reason": post_result.signals.get("reason"),
                    "share_of_facts_kept": round(post_result.confidence_score, 4),
                    "signals": post_result.signals,
                    "kept_entities": post_result.kept_entity_labels,
                    "facts": post_result.fact_log,                           # [PG2]
                }
                sig = post_result.signals
                counts["facts_extracted"] += sig.get("facts_total", 0)
                counts["facts_kept"] += sig.get("facts_kept", 0)
                counts["facts_duplicate"] += sig.get("facts_duplicate", 0)
                counts["facts_dropped_unconnected"] += sig.get("facts_dropped_unconnected", 0)
                counts["fact_conflicts"] += sig.get("fact_conflicts", 0)

            if post_result is not None and post_result.decision == "reject":
                # Post-Extraction Gate rejected -- nothing was written to Neo4j.
                print(f"    -> REJECTED by Post-Extraction Gate "
                      f"({post_result.signals.get('reason')}, signals={post_result.signals}) "
                      f"-- not written to KG.")
                counts["post_gate_reject"] += 1
                record["outcome"] = "rejected_by_post_gate"
                gate_cluster_memory.exemplars.add_rejected(RejectedExemplar(
                    sparse_node=node_label,
                    generated_question=final_question,
                    failure_type="post_extraction_gate_reject",
                    rejection_reason=(
                        f"post_extraction_gate_v2: {post_result.signals.get('reason')}; "
                        f"signals={post_result.signals}"
                    ),
                    confidence_score=post_result.confidence_score,
                ))
                _save_run_log(log_path, run_log)
                # Still wait before the next extraction (Groq rate limit).
                await asyncio.sleep(40)
                continue

            counts["ingested"] += 1
            if post_result is not None:
                counts["post_gate_accept"] += 1
                record["outcome"] = "ingested"
            else:
                counts["written_without_any_gate"] += 1
                record["outcome"] = "ingested_without_any_gate"
            print(f"    -> ingested ({len(snippet)} chars)")

            # ---- Exemplar write-back (unchanged) -- only when a real cluster gated this.
            if gate_cluster_memory is not None:
                accepted_triple = (
                    ", ".join(post_result.kept_entity_labels) if post_result is not None else None
                )
                acceptance_reason = (
                    f"question_gate_accept_{accept_result.decision_path} "
                    f"(combined_score={accept_result.confidence_score:.4f})"
                )
                if post_result is not None:
                    acceptance_reason += (
                        f"; post_extraction_gate_v2_accept "
                        f"({post_result.signals.get('facts_kept')} new fact(s) kept)"
                    )
                gate_cluster_memory.exemplars.add_accepted(AcceptedExemplar(
                    sparse_node=node_label,
                    generated_question=final_question,
                    retrieved_evidence=snippet,
                    acceptance_reason=acceptance_reason,
                    accepted_triple=accepted_triple,
                ))

            _save_run_log(log_path, run_log)

            # Wait before the next extraction to avoid hitting Groq's token rate limit
            await asyncio.sleep(40)

        print()  # blank line between metric cycles

    with driver.session() as session:
        after_total = session.run("MATCH (n:__Entity__) RETURN count(n) AS c").single()["c"]

    run_log["entity_nodes_after"] = after_total
    run_log["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _save_run_log(log_path, run_log)

    total = counts["questions"] or 1
    print(f"[Stage 4] Done. {counts['questions']} question(s): "
          f"{counts['ingested']} ingested, "
          f"{counts['question_gate_reject']} skipped by Question Gate, "
          f"{counts['post_gate_reject']} rejected by Post-Extraction Gate, "
          f"{counts['no_search_result']} with no search result.")
    print(f"[Stage 4] [F5] Cluster for target node: seed={counts['cluster_from_seed']}, "
          f"assigned by embedding={counts['cluster_assigned_by_embedding']}, "
          f"fail-open={counts['fail_open']} ({100 * counts['fail_open'] / total:.0f}% of questions).")
    print(f"[Stage 4] [F5] Written without any gate: {counts['written_without_any_gate']} "
          f"of {counts['ingested']} ingested.")
    print(f"[Stage 4] [PG2] Facts extracted (gated pages): {counts['facts_extracted']}  ->  "
          f"kept {counts['facts_kept']}, duplicates skipped {counts['facts_duplicate']}, "
          f"dropped as unconnected {counts['facts_dropped_unconnected']}, "
          f"conflicts logged {counts['fact_conflicts']}.")
    print(f"[Stage 4] Entity nodes BEFORE: {before_total}  ->  AFTER: {after_total}")
    print(f"[Stage 4] Run log saved: {log_path}")

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())