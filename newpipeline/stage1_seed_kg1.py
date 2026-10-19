"""
MKGE -- OUR PIPELINE -- Stage 1: Knowledge Graph Seeding
============================================================
Our improved version of Stage 1. The original, untouched version stays
in the original pipeline folder and is what run A (original MKGE) uses.

What this script does:
  1. Takes ONE question (set in QUESTION below).
  2. Sends it to Tavily and gets the top 3 results.
  3. [R1] Picks the BEST of the 3 instead of blindly taking the first:
     the one whose text is closest in meaning to the question
     (cosine similarity, using the same local MiniLM embedder the
     pipeline already loads -- no extra API call). Every candidate and
     its score is printed and saved to seed_retrieval_log.json.
  4. Feeds the chosen snippet through Neo4j's GraphRAG knowledge-graph
     builder (SimpleKGPipeline), now with:
       [F1] the explicit entity/relation vocabulary passed in, so every
            entity is stored with a `name` property (no more "unnamed"
            movies) and the seed uses the same vocabulary as Stage 4.
       [F4] a text splitter (500 chars, 100 overlap, same as Stage 4),
            so the seed becomes several chunks instead of one -- each
            entity links to the chunk it actually came from.
  5. Prints a summary + a Cypher query to visualize the result.

Changes vs the original Stage 1 are marked [R1], [F1], [F4] below.

Run with:  python newpipeline\\stage1_seed_kg1.py   (from the project root)
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tavily import TavilyClient

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (  # [F4]
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

load_dotenv()  # finds the .env file in this folder or any parent folder

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
GENERIC_ENTITIES = ["Person", "Organization", "Location", "Work", "Event", "Concept", "Date"]
GENERIC_RELATIONS = ["RELATED_TO", "LOCATED_IN", "PART_OF", "CREATED_BY",
                     "OWNS", "CAUSED_BY", "OCCURRED_ON", "ASSOCIATED_WITH"]
QUESTION = "What are some of the best psychological thriller movies of the modern era?"
# ^ One of the worked examples in Table 3 of the paper.

# [R1] How many Tavily results to compare, and how to choose among them.
#   "best" = pick the result most similar in meaning to the question (ours)
#   "top1" = take Tavily's first result, like the original (for the D - R1 run)
NUM_CANDIDATES = 3
SELECTION_MODE = "best"
RETRIEVAL_LOG_FILE = "seed_retrieval_log.json"

# [F4] Same chunking as Stage 4.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
# Groq is a FREE, OpenAI-compatible stand-in for GPT-4o: we point the
# OpenAILLM class at Groq's URL instead of OpenAI's.


# -----------------------------------------------------------------------
# [R1] Best-of-N snippet selection
# -----------------------------------------------------------------------
def _cosine_similarity(vec_a, vec_b) -> float:
    """How alike two meaning-vectors are: 1 = same meaning, 0 = unrelated.
    Plain Python, no numpy (same convention as the rest of the project)."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def select_best_snippet(question: str, results: list, embedder, mode: str = "best"):
    """[R1] Choose ONE snippet out of Tavily's results.

    mode="best": embed the question once, embed each result's text, keep
                 the result with the highest cosine similarity. Ties go to
                 the earlier result (i.e. Tavily's own ranking).
    mode="top1": return the first result with non-empty text, exactly like
                 the original pipeline (used for the "ours minus R1" run).

    Returns (chosen_text, log). chosen_text is None if every result is empty.
    log records every candidate's url and similarity plus the chosen index,
    so we can report how often R1 picked something other than Tavily's #1.
    """
    log = {"question": question, "mode": mode, "chosen_index": None, "candidates": []}

    if mode == "top1":
        for i, result in enumerate(results):
            text = result.get("content") or ""
            log["candidates"].append({"index": i, "url": result.get("url"),
                                      "title": result.get("title"), "chars": len(text)})
            if text and log["chosen_index"] is None:
                log["chosen_index"] = i
        chosen = results[log["chosen_index"]]["content"] if log["chosen_index"] is not None else None
        return chosen, log

    question_vec = embedder.embed_query(question)
    best_index, best_sim = None, -1.0

    for i, result in enumerate(results):
        text = result.get("content") or ""
        candidate = {"index": i, "url": result.get("url"), "title": result.get("title"),
                     "chars": len(text), "tavily_score": result.get("score"), "similarity": None}
        if text:
            sim = _cosine_similarity(question_vec, embedder.embed_query(text))
            candidate["similarity"] = round(sim, 4)
            if sim > best_sim:  # strict ">" so ties keep Tavily's earlier rank
                best_index, best_sim = i, sim
        log["candidates"].append(candidate)

    log["chosen_index"] = best_index
    chosen = results[best_index]["content"] if best_index is not None else None
    return chosen, log


def get_seed_snippet(question: str, embedder) -> str:
    """Stage 1a -- Tavily search, then [R1] pick the best of the results."""
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily.search(question, max_results=NUM_CANDIDATES)
    results = response.get("results", [])

    snippet, log = select_best_snippet(question, results, embedder, mode=SELECTION_MODE)

    print(f"[Stage 1a] Tavily returned {len(results)} result(s); selection mode = {SELECTION_MODE!r}")
    for c in log["candidates"]:
        marker = "  <-- chosen" if c["index"] == log["chosen_index"] else ""
        print(f"    #{c['index']}  sim={c.get('similarity')}  chars={c['chars']}  "
              f"{c.get('title')}  {c.get('url')}{marker}")

    with open(RETRIEVAL_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"[Stage 1a] Candidate log saved to {RETRIEVAL_LOG_FILE}")

    if snippet is None:
        raise RuntimeError("Tavily returned no usable result for the seed question.")

    print(f"[Stage 1a] Chosen snippet ({len(snippet)} chars):")
    print(snippet[:300] + ("..." if len(snippet) > 300 else ""))
    print()
    return snippet


async def build_seed_kg(snippet: str, driver, embedder) -> None:
    """Stage 1b -- feed the snippet through GraphRAG's KG builder."""
    llm = OpenAILLM(
        model_name="openai/gpt-oss-120b",  # a free model hosted on Groq
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model_params={"temperature": 0, "response_format": {"type": "json_object"}},
    )
    llm.supports_structured_output = False

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        text_splitter=FixedSizeSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),  # [F4]
        entities=GENERIC_ENTITIES,    # [F1]
        relations=GENERIC_RELATIONS,  # [F1]
        on_error="RAISE",
        from_pdf=False,  # we're passing raw text directly, not a file
    )

    print("[Stage 1b] Building seed KG in Neo4j (one Groq call per chunk)...")
    result = await kg_builder.run_async(text=snippet)
    print(f"[Stage 1b] Done. Pipeline result: {result.result}")
    print()


def report_graph_contents(driver) -> None:
    """Sanity check -- confirm something landed in Neo4j, and check F1/F4."""
    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
        no_name = session.run(
            "MATCH (n:__Entity__) WHERE n.name IS NULL RETURN count(n) AS c"
        ).single()["c"]

    print(f"[Check] Seed KG now has {node_count} nodes and {rel_count} relationships.")
    print(f"[Check] [F4] Chunk nodes: {chunk_count} (should be more than 1)")
    print(f"[Check] [F1] Entities without a `name`: {no_name} (should be 0)")
    print()
    print("To SEE the graph, open Neo4j Browser (or Aura's Query tab) and run:")
    print("    MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")


async def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    # Loaded ONCE and shared: used by [R1] to compare snippets and by
    # SimpleKGPipeline to embed chunks. Local, free, no API call.
    embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

    snippet = get_seed_snippet(QUESTION, embedder)
    await build_seed_kg(snippet, driver, embedder)
    report_graph_contents(driver)

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())
