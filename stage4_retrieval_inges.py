"""
MKGE Paper Replication -- Stage 4: Retrieval Ingestion BASELINE
===============================================================

Baseline enrichment pipeline.

For each metric:
    1. Reload current KG
    2. Recompute sparsity
    3. Generate questions
    4. Search Tavily
    5. Ingest result into Neo4j

IMPORTANT:
This version intentionally contains NO memory gate.
It is the baseline condition against which the memory-gated
pipeline will later be compared.

Token optimization:
- Stage 3 generates 2 questions per metric.
- 40 second wait between extractions to avoid Groq rate limits.
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
from neo4j_graphrag.experimental.pipeline.kg_builder import (
    SimpleKGPipeline
)
from neo4j_graphrag.llm import OpenAILLM

from stage2_sparsity_metrics import (
    METRIC_NAMES,
    compute_sparsity_metrics,
    flag_sparse_nodes,
    load_entity_graph,
)

from stage3_question_gen import (
    generate_questions_for_metric,
    get_node_preview,
)

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


def get_top_snippet(
    tavily: TavilyClient,
    question: str
):
    """
    Same rule as Stage 1:
    keep ONLY the first web result.
    """

    try:
        response = tavily.search(
            question,
            max_results=3
        )

        return response["results"][0]["content"]

    except (IndexError, KeyError):
        return None


def sparse_previews_for_metric(
    driver,
    metric: str
) -> dict:

    """
    Reload the CURRENT graph after previous metric cycles,
    recompute sparsity, and return nodes sparse on this metric.
    """

    G = load_entity_graph(driver)

    scores = compute_sparsity_metrics(G)

    sparse_flags, _ = flag_sparse_nodes(scores)

    import networkx as nx

    labels = nx.get_node_attributes(
        G,
        "label"
    )

    sparse_node_ids = [
        n
        for n in sparse_flags
        if sparse_flags[n][metric]
    ][:50]

    return {
        node_id: (
            labels.get(
                node_id,
                node_id
            ),
            get_node_preview(
                driver,
                node_id,
                labels.get(
                    node_id,
                    node_id
                )
            ),
        )
        for node_id in sparse_node_ids
    }


async def main():

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USERNAME,
            NEO4J_PASSWORD
        )
    )

    driver.verify_connectivity()

    print(
        "[Setup] Connected to Neo4j.\n"
    )

    with driver.session() as session:

        before_total = session.run(
            "MATCH (n:__Entity__) "
            "RETURN count(n) AS c"
        ).single()["c"]

    print(
        f"[Stage 4] Entity nodes BEFORE enrichment: "
        f"{before_total}\n"
    )

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    tavily = TavilyClient(
        api_key=TAVILY_API_KEY
    )

    llm = OpenAILLM(
        model_name="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model_params={
            "temperature": 0,
            "response_format": {
                "type": "json_object"
            }
        },
    )

    # Groq compatibility workaround used in the project.
    llm.supports_structured_output = False

    embedder = SentenceTransformerEmbeddings(
        model="all-MiniLM-L6-v2"
    )

    splitter = FixedSizeSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        text_splitter=splitter,
        entities=[
            "Person",
            "Organization",
            "Location",
            "Work",
            "Event",
            "Concept",
            "Date",
        ],
        relations=[
            "RELATED_TO",
            "LOCATED_IN",
            "PART_OF",
            "CREATED_BY",
            "OWNS",
            "CAUSED_BY",
            "OCCURRED_ON",
            "ASSOCIATED_WITH",
        ],
        from_pdf=False,
    )

    total_ingested = 0
    total_skipped = 0

    # ---------------------------------------------------------
    # BASELINE LOOP
    # ---------------------------------------------------------

    for metric in METRIC_NAMES:

        print(
            f"[Stage 4] === Metric cycle: "
            f"{metric} ==="
        )

        # Recompute sparsity on the CURRENT graph.
        sparse_previews = sparse_previews_for_metric(
            driver,
            metric
        )

        print(
            f"[Stage 4] ({metric}) "
            f"{len(sparse_previews)} sparse node(s) "
            f"on current graph."
        )

        # Generate token-optimized questions.
        questions = generate_questions_for_metric(
            client,
            metric,
            sparse_previews
        )

        for node_id, node_label, question in questions:

            print(
                f"    - [{node_label}] "
                f"{question}"
            )

        # Retrieve + ingest each question.
        for node_id, node_label, question in questions:

            print(
                f"[Stage 4] Target node: "
                f"{node_label}"
            )

            print(
                f"[Stage 4] Question: "
                f"{question}"
            )

            snippet = get_top_snippet(
                tavily,
                question
            )

            if not snippet:

                print(
                    "    -> no result found, skipping"
                )

                total_skipped += 1

                continue

            await kg_builder.run_async(
                text=snippet
            )

            total_ingested += 1

            print(
                f"    -> ingested "
                f"({len(snippet)} chars)"
            )

            # -------------------------------------------------
            # RATE LIMIT PROTECTION
            # -------------------------------------------------

            await asyncio.sleep(40)

        print()

    # ---------------------------------------------------------
    # FINAL KG SIZE
    # ---------------------------------------------------------

    with driver.session() as session:

        after_total = session.run(
            "MATCH (n:__Entity__) "
            "RETURN count(n) AS c"
        ).single()["c"]

    print(
        f"[Stage 4] Done. "
        f"Ingested {total_ingested} answers, "
        f"skipped {total_skipped}."
    )

    print(
        f"[Stage 4] Entity nodes BEFORE: "
        f"{before_total} -> AFTER: {after_total}"
    )

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())