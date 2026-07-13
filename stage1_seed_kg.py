"""
MKGE Paper Replication -- Stage 1: Knowledge Graph Seeding
============================================================
This is STAGE 1 ONLY of the MetaKGEnrich (MKGE) pipeline from the paper.
Nothing else -- no sparsity metrics, no question generation, no enrichment.
Get this working and understood before moving to Stage 2.

What this script does:
  1. Takes ONE question (set in QUESTION below).
  2. Sends it to Tavily, keeps ONLY the top-ranked result's text snippet.
     -> Paper quote: "We retain only the top-ranked snippet and ingest it
        as one 500-token document into Neo4j using the GraphRAG pipeline."
  3. Feeds that snippet through Neo4j's GraphRAG knowledge-graph builder
     (SimpleKGPipeline). Internally: an LLM reads the text, extracts
     entities + relationships, and writes them into Neo4j as nodes/edges.
  4. Prints a summary + gives you a Cypher query to visualize the result
     in Neo4j Browser / Aura's Query tab.

Run with:  python stage1_seed_kg.py
"""

import asyncio
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tavily import TavilyClient

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

# import logging
# logging.basicConfig(level=logging.DEBUG)

load_dotenv()  # reads the .env file sitting next to this script

# -----------------------------------------------------------------------
# CONFIG -- this is the ONLY thing you should change to test a new question
# -----------------------------------------------------------------------
QUESTION = "Who owns the rights to Alice in Wonderland?"
# ^ This is literally one of the worked examples in Table 3 of the paper
#   (their "GraphRAG before/after" table). Using it first means you can
#   sanity-check your output against what they report.

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
# We're using Groq as a FREE, OpenAI-compatible stand-in for GPT-4o.
# This works because Groq's API speaks the same protocol as OpenAI's --
# we just redirect the OpenAILLM class to Groq's URL instead of OpenAI's.


def get_seed_snippet(question: str) -> str:
    """Stage 1a -- Tavily search, keep ONLY the first result's content.

    The paper deliberately keeps just one snippet so they can later test
    whether a question was answerable from minimal info, or genuinely
    needed enrichment. Don't be tempted to grab all 3 results here --
    that would change what you're measuring later.
    """
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily.search(question, max_results=3)
    top_snippet = response["results"][0]["content"]
    print(f"[Stage 1a] Tavily top result ({len(top_snippet)} chars):")
    print(top_snippet[:300] + ("..." if len(top_snippet) > 300 else ""))
    print()
    return top_snippet


async def build_seed_kg(snippet: str, driver) -> None:
    """Stage 1b -- feed the snippet through GraphRAG's KG builder.

    Paper quote: "ingest it as one 500-token document into Neo4j using
    the GraphRAG pipeline." This is exactly what SimpleKGPipeline does:
    LLM extracts entities/relations from the text -> writes them to Neo4j.
    """
    llm = OpenAILLM(
        model_name="openai/gpt-oss-120b",  # a free model hosted on Groq
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",  # <-- this is the whole trick
        model_params={"temperature": 0, "response_format": {"type": "json_object"}},
    )
    llm.supports_structured_output = False
    # Local, free, runs on your own machine -- same model the paper itself
    # uses (all-MiniLM-L6-v2), no API call needed for this part at all.
    embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        from_pdf=False,  # we're passing raw text directly, not a file
    )

    print("[Stage 1b] Building seed KG in Neo4j (this makes one Groq call)...")
    result = await kg_builder.run_async(text=snippet)
    print(f"[Stage 1b] Done. Pipeline result: {result.result}")
    print()


def report_graph_contents(driver) -> None:
    """Sanity check -- confirm something actually landed in Neo4j."""
    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    print(f"[Check] Seed KG now has {node_count} nodes and {rel_count} relationships.")
    print()
    print("To SEE the graph, open Neo4j Browser (or Aura's Query tab) and run:")
    print("    MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")


async def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    snippet = get_seed_snippet(QUESTION)
    await build_seed_kg(snippet, driver)
    report_graph_contents(driver)

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())
