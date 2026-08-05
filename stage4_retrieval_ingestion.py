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
)
from stage3_question_generation import (
    generate_questions_for_metric,
    get_node_preview,
)

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

    total_ingested, total_skipped = 0, 0

    # ---- THE FIX: one metric at a time, detect -> question -> ingest,
    # ---- before moving to the next metric. Matches Algorithm 1 exactly.
    for metric in METRIC_NAMES:
        print(f"[Stage 4] === Metric cycle: {metric} ===")

        sparse_previews = sparse_previews_for_metric(driver, metric)
        print(f"[Stage 4] ({metric}) {len(sparse_previews)} sparse node(s) on current graph.")

        questions = generate_questions_for_metric(client, metric, sparse_previews)
        for q in questions:
            print(f"    - {q}")

        for question in questions:
            print(f"[Stage 4] ({metric}) Searching: {question}")
            snippet = get_top_snippet(tavily, question)
            if not snippet:
                print("    -> no result found, skipping")
                total_skipped += 1
                continue
            
            await kg_builder.run_async(text=snippet)
            total_ingested += 1
            print(f"    -> ingested ({len(snippet)} chars)")

            # Wait before the next extraction to avoid hitting Groq's token rate limit
            await asyncio.sleep(40)

        print()  # blank line between metric cycles

    with driver.session() as session:
        after_total = session.run("MATCH (n:__Entity__) RETURN count(n) AS c").single()["c"]

    print(f"[Stage 4] Done. Ingested {total_ingested} answers, skipped {total_skipped}.")
    print(f"[Stage 4] Entity nodes BEFORE: {before_total}  ->  AFTER: {after_total}")

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())