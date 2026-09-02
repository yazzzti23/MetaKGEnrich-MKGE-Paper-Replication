"""
MKGE Paper Replication -- Stage 4: Retrieval Ingestion (FIXED)
============================================================
FIX vs. the original version: the paper's Algorithm 1 recomputes sparsity
INSIDE the per-metric loop --

    foreach m in M do
        sparse <- DETECTSPARSE(KG, m)
        Qm <- LLMQUESTIONGEN(sparse, m)
        docs <- TAVILYSEARCH(Qm)
        KG <- INGEST(docs)

-- meaning each metric's sparsity detection runs on a graph that has
ALREADY been enriched by every metric before it in the loop. The
original version generated all 35 questions upfront from the same seed
graph, then ingested everything after. This version instead loops
metric-by-metric: reload the graph -> recompute sparsity -> generate
THAT metric's 5 questions -> ingest THOSE answers -> move to next metric.

This also explains something worth mentioning to your professor: the
paper reports ~5 Tavily calls per question on average despite a 35-
question ceiling. That only makes sense if later metrics increasingly
find fewer sparse nodes left to ask about, which can only happen if
sparsity detection re-runs on the progressively-enriched graph -- i.e.
exactly this interleaved structure, not the flat/batched version.

Run with:  python stage4_retrieval_ingestion.py
(same .env, same folder as stage2/stage3 files -- this imports both)
"""

import asyncio
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from tavily import TavilyClient

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

from stage2_sparsity_metrics import (
    METRIC_NAMES,
    compute_sparsity_metrics,
    flag_sparse_nodes,
    load_entity_graph,
    get_node_cluster_mapping,
)
from stage3_question_generation2 import (
    generate_questions_for_metric,
    regenerate_question_for_node,
    get_node_preview,
)
from cluster_memory import (
    build_cluster_memory_store,
    compute_cluster_embedding,
    AcceptedExemplar,
    RejectedExemplar,
)
from memory_gates import question_memory_gate

load_dotenv()

GENERIC_ENTITIES = ["Person", "Organization", "Location", "Work", "Event", "Concept", "Date"]
GENERIC_RELATIONS = ["RELATED_TO", "LOCATED_IN", "PART_OF", "CREATED_BY",
                      "OWNS", "CAUSED_BY", "OCCURRED_ON", "ASSOCIATED_WITH"]

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


def get_top_snippet(tavily: TavilyClient, question: str):
    """Same rule as Stage 1: keep ONLY the first web result."""
    try:
        response = tavily.search(question, max_results=3)
        return response["results"][0]["content"]
    except (IndexError, KeyError):
        return None


def sparse_previews_for_metric(driver, metric: str) -> dict:
    """Reloads the CURRENT graph state (post any prior metric's ingestion
    in this same run) and recomputes sparsity fresh, then returns previews
    only for nodes sparse on THIS metric. This is the piece that was
    missing before -- calling this once per metric, mid-loop, is what
    makes detection see the progressively-enriched graph.
    """
    G = load_entity_graph(driver)
    scores = compute_sparsity_metrics(G)
    sparse_flags, _ = flag_sparse_nodes(scores)
    import networkx as nx
    labels = nx.get_node_attributes(G, "label")

    sparse_node_ids = [n for n in sparse_flags if sparse_flags[n][metric]][:50]
    return {
        node_id: (
            labels.get(node_id, node_id),
            get_node_preview(driver, node_id, labels.get(node_id, node_id)),
        )
        for node_id in sparse_node_ids
    }


async def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    with driver.session() as session:
        before_total = session.run("MATCH (n:__Entity__) RETURN count(n) AS c").single()["c"]
    print(f"[Stage 4] Entity nodes BEFORE enrichment: {before_total}\n")

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
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

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        text_splitter=splitter,
        entities=GENERIC_ENTITIES,
        relations=GENERIC_RELATIONS,
        from_pdf=False,
    )

    # ---- Cluster Memory Store: built ONCE here, held for the entire run.
    # ---- Per cluster_memory.py's own docstring, this (like
    # ---- get_node_cluster_mapping) is meant to run once on the verified
    # ---- seed KG, NOT rebuilt every metric cycle the way sparsity
    # ---- detection is -- re-running Louvain on a progressively-enriched
    # ---- graph would renumber cluster IDs mid-run and corrupt the
    # ---- exemplar memory this integration is accumulating.
    cluster_G = load_entity_graph(driver)
    node_cluster_map = get_node_cluster_mapping(cluster_G)
    cluster_store = build_cluster_memory_store(driver, node_cluster_map, cluster_G, llm_client=client)

    clusters_to_nodes = {}
    for member_node_id, cluster_id in node_cluster_map.items():
        clusters_to_nodes.setdefault(cluster_id, []).append(member_node_id)

    print(f"[Stage 4] Cluster Memory Store built once: {len(cluster_store)} cluster(s), "
          f"{len(node_cluster_map)} node(s) clustered.\n")

    total_ingested, total_skipped, total_gate_rejected = 0, 0, 0

    # ---- THE FIX: one metric at a time, detect -> question -> ingest,
    # ---- before moving to the next metric. Matches Algorithm 1 exactly.
    for metric in METRIC_NAMES:
        print(f"[Stage 4] === Metric cycle: {metric} ===")

        sparse_previews = sparse_previews_for_metric(driver, metric)
        print(f"[Stage 4] ({metric}) {len(sparse_previews)} sparse node(s) on current graph.")

        questions = generate_questions_for_metric(client, metric, sparse_previews)
        for node_id, node_label, question in questions:
            print(f"    - [{node_label}] {question}")

        for node_id, node_label, question in questions:
            print(f"[Stage 4] Target node: {node_label}")
            print(f"[Stage 4] Question: {question}")

            # ---- Question Memory Gate ----
            cluster_id = node_cluster_map.get(node_id)

            if cluster_id is None:
                # This node has no cluster assignment, most likely
                # because it was ingested by an earlier metric cycle
                # THIS run (i.e. after the one-time clustering pass
                # above). Try nearest-cluster assignment before falling
                # back to fail-open: reuse compute_cluster_embedding on
                # a single-element list to get this node's OWN chunk
                # embedding (no new Cypher query), then ask the store
                # for its nearest existing cluster.
                node_embedding = compute_cluster_embedding(driver, [node_id])
                assigned_cluster_id = (
                    cluster_store.assign_new_node(node_id, node_embedding, driver)
                    if node_embedding is not None else None
                )

                if assigned_cluster_id is not None:
                    print(f"    [Gate] Assigned previously-unclustered node to cluster {assigned_cluster_id}.")
                    node_cluster_map[node_id] = assigned_cluster_id
                    clusters_to_nodes.setdefault(assigned_cluster_id, []).append(node_id)
                    cluster_id = assigned_cluster_id
                    # falls through to the gated branch below, now that
                    # cluster_id is a real assignment instead of None

            if cluster_id is None:
                # True fail-open: no embedding yet for this node, or no
                # existing cluster had a usable embedding to compare
                # against. Unchanged from the original behavior.
                print("    [Gate] No cluster assignment for this node -- fail-open, skipping gate.")
                final_question = question
                gate_cluster_memory = None
                accept_result = None
            else:
                cluster_memory = cluster_store.get(cluster_id)
                member_node_ids = clusters_to_nodes[cluster_id]

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

                if result.decision == "accept":
                    final_question = question
                    gate_cluster_memory = cluster_memory
                    accept_result = result
                else:
                    print(f"    [Gate] REJECTED (attempt 1, {result.decision_path}, "
                          f"score={result.confidence_score:.4f}) -- regenerating once.")

                    regen_node_id, regen_node_label, regen_question = regenerate_question_for_node(
                        client, node_id, node_label, metric, driver, rejected_question=question,
                    )
                    print(f"    [Gate] Regenerated question: {regen_question}")

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

                    if result2.decision == "reject":
                        # Maximum 2 TOTAL attempts, per the locked retry
                        # policy -- no further regeneration. Skip
                        # retrieval for this node this cycle, and record
                        # a real RejectedExemplar so future gate calls
                        # in this cluster have actual exemplar signal
                        # instead of permanent cold-start.
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
                        total_skipped += 1
                        total_gate_rejected += 1
                        continue  # next question -- no retrieval for this node this cycle

                    final_question = regen_question
                    gate_cluster_memory = cluster_memory
                    accept_result = result2

            # ---- Retrieval + ingestion (existing logic, unchanged) ----
            snippet = get_top_snippet(tavily, final_question)
            if not snippet:
                print("    -> no result found, skipping")
                total_skipped += 1
                continue
            
            await kg_builder.run_async(text=snippet)
            total_ingested += 1
            print(f"    -> ingested ({len(snippet)} chars)")

            # ---- Exemplar write-back -- only when a real cluster gated
            # ---- this question (fail-open nodes have no cluster_memory
            # ---- to write to). accepted_triple is left None: this
            # ---- pipeline's ingestion is one opaque SimpleKGPipeline
            # ---- call, with no clean access to the specific triple it
            # ---- wrote -- that's the future Post-Extraction Gate's job,
            # ---- not fabricated here.
            if gate_cluster_memory is not None:
                gate_cluster_memory.exemplars.add_accepted(AcceptedExemplar(
                    sparse_node=node_label,
                    generated_question=final_question,
                    retrieved_evidence=snippet,
                    acceptance_reason=(
                        f"question_gate_accept_{accept_result.decision_path} "
                        f"(combined_score={accept_result.confidence_score:.4f})"
                    ),
                ))

            # Wait before the next extraction to avoid hitting Groq's token rate limit
            await asyncio.sleep(40)

        print()  # blank line between metric cycles

    with driver.session() as session:
        after_total = session.run("MATCH (n:__Entity__) RETURN count(n) AS c").single()["c"]

    print(f"[Stage 4] Done. Ingested {total_ingested} answers, skipped {total_skipped} "
          f"({total_gate_rejected} of those due to Question Memory Gate rejection after regeneration).")
    print(f"[Stage 4] Entity nodes BEFORE: {before_total}  ->  AFTER: {after_total}")

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())