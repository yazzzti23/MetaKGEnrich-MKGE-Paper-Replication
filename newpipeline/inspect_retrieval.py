"""
MKGE experiment helper -- inspect what Stage 5 actually retrieves
==================================================================
READ-ONLY. No LLM / Groq calls, no writes to Neo4j.

Stage 5 answers from the top-5 Chunk nodes closest to the question.
This script shows WHICH 5 chunks those are on the CURRENT graph, and
whether each one came from the frozen seed or was added by enrichment
(Stage 4). Use it right after a "stage5 ... after" run, BEFORE you
restore the seed, to see whether enrichment actually reached the answer.

Usage (from the repo root):
    python newpipeline\\inspect_retrieval.py q2
    python newpipeline\\inspect_retrieval.py q2 | Tee-Object runs\\q2_original_after_chunks.txt

The first argument picks the question and its frozen seed (seeds\\seed_<qid>.json).
"""

import json
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

INDEX_NAME = "chunk_embeddings"   # same index Stage 5 uses
TOP_K = 5                         # same top_k Stage 5 uses
PREVIEW_CHARS = 300

QUESTIONS = {
    "q1": "Where was High School Musical 2 filmed?",
    "q2": "Who owns the rights to Alice in Wonderland?",
    "q3": "Where did the first African-American Air Force unit train?",
    "q4": "What causes monsoon winds to change direction seasonally?",
    "q5": "What are the requirements to pass the South African National Senior Certificate?",
}


def load_seed_chunk_texts(qid: str) -> set:
    """Texts of the Chunk nodes in the frozen seed, so each retrieved
    chunk can be labelled SEED or ADDED BY ENRICHMENT."""
    path = os.path.join("seeds", f"seed_{qid}.json")
    if not os.path.exists(path):
        print(f"[warn] {path} not found -- cannot tell seed chunks from added ones.\n")
        return set()
    with open(path, encoding="utf-8") as f:
        snapshot = json.load(f)
    return {
        node["properties"].get("text", "")
        for node in snapshot["nodes"]
        if "Chunk" in node["labels"]
    }


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in QUESTIONS:
        print(f"Usage: python newpipeline\\inspect_retrieval.py <{'|'.join(QUESTIONS)}>")
        sys.exit(1)
    qid = sys.argv[1]
    question = QUESTIONS[qid]

    seed_texts = load_seed_chunk_texts(qid)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()

    embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
    query_vector = embedder.embed_query(question)

    with driver.session() as session:
        total_chunks = session.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        rows = session.run(
            """
            CALL db.index.vector.queryNodes($index, $k, $vec)
            YIELD node AS c, score
            OPTIONAL MATCH (c)-[:FROM_DOCUMENT]->(d:Document)
            RETURN c.text AS text, c.index AS idx, score, d.path AS doc
            ORDER BY score DESC
            """,
            index=INDEX_NAME, k=TOP_K, vec=query_vector,
        ).data()
    driver.close()

    print(f"Question : {question}")
    print(f"Chunks on graph now: {total_chunks}  (seed had {len(seed_texts)})")
    print(f"Top-{TOP_K} chunks Stage 5 retrieves:\n")

    added = 0
    for rank, row in enumerate(rows, start=1):
        text = row["text"] or ""
        origin = "SEED" if text in seed_texts else "ADDED BY ENRICHMENT"
        added += origin != "SEED"
        preview = " ".join(text.split())[:PREVIEW_CHARS]
        print(f"#{rank}  score={row['score']:.4f}  [{origin}]  chunk_index={row['idx']}  doc={row['doc']}")
        print(f"    {preview}{'...' if len(text) > PREVIEW_CHARS else ''}\n")

    print(f"Summary: {added} of {len(rows)} retrieved chunks came from enrichment.")


if __name__ == "__main__":
    main()