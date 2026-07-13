"""
MKGE Paper Replication -- Stage 5: Evaluation
============================================================
The piece that was missing: generate an answer to the ORIGINAL seed
question from the current graph state, using GraphRAG (vector retrieval
over Chunk embeddings + LLM answer generation), and judge whether the
AFTER answer improved on the BEFORE answer.

Paper quote: "GraphRAG retrieves the top five semantically closest KG
chunks before and after enrichment. These are fed to GPT-4 for answer
generation. A separate GPT-4 judge labels each answer pair as Improved
or Not Improved."

IMPORTANT GOTCHA (confirmed against the neo4j-graphrag-python docs):
SimpleKGPipeline automatically EMBEDS each Chunk node if you give it an
embedder, but it does NOT create the vector index used for similarity
search -- that's a separate one-time step (`create_vector_index`).
Skipping this is the most likely reason a GraphRAG search would come
back empty. `ensure_vector_index()` below handles it, and is safe to
call every run (no-ops if it already exists).

USAGE -- run this TWICE around your demo:
    python stage5_evaluation.py before   <- run right after stage1_seed_kg.py
    python stage5_evaluation.py after    <- run right after stage4_retrieval_ingestion_FIXED.py
The "before" run saves its answer to answers_cache.json; the "after" run
reads it back, generates its own answer, and calls the judge on both.
"""

import json
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

INDEX_NAME = "chunk_embeddings"   # pick one name, reuse it everywhere
EMBED_DIM = 384                    # all-MiniLM-L6-v2 output size
CACHE_FILE = "answers_cache.json"

# Must match whatever QUESTION you set in stage1_seed_kg.py for this run
QUESTION = "Who owns the rights to Alice in Wonderland?"


def ensure_vector_index(driver) -> None:
    try:
        create_vector_index(
            driver,
            INDEX_NAME,
            label="Chunk",
            embedding_property="embedding",
            dimensions=EMBED_DIM,
            similarity_fn="cosine",
        )
        print(f"[Stage 5] Vector index '{INDEX_NAME}' ready.\n")
    except Exception as e:
        # Most common case: index already exists from a previous run -- fine.
        print(f"[Stage 5] Vector index setup note: {e}\n")


def answer_question(driver, embedder, llm, question: str, tag: str) -> str:
    """GraphRAG: retrieve top-5 closest Chunk nodes by embedding
    similarity, feed them to the LLM, return the generated answer.
    """
    retriever = VectorRetriever(driver, INDEX_NAME, embedder=embedder)
    rag = GraphRAG(retriever=retriever, llm=llm)

    response = rag.search(
        query_text=question,
        retriever_config={"top_k": 5},
        return_context=True,
    )

    print(f"[Stage 5 - {tag}] Answer:")
    print(response.answer)
    print()
    return response.answer


def judge_answers(client, question: str, ans_before: str, ans_after: str) -> str:
    """Separate LLM call acting as an independent judge -- mirrors the
    paper's 'a separate GPT-4 judge labels each answer pair as Improved
    or Not Improved.'
    """
    prompt = f"""You are an impartial judge comparing two answers to the same question.

Question: {question}

Answer BEFORE enrichment:
{ans_before}

Answer AFTER enrichment:
{ans_after}

Did the AFTER answer improve on the BEFORE answer in completeness, correctness,
or specificity? Respond with exactly one word on the first line: "Improved" or
"Not Improved". Then give a one-sentence justification on the next line.
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    verdict_text = response.choices[0].message.content.strip()
    print(f"[Stage 5 - JUDGE VERDICT]\n{verdict_text}")
    return verdict_text


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("before", "after"):
        print("Usage: python stage5_evaluation.py [before|after]")
        sys.exit(1)
    mode = sys.argv[1]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    ensure_vector_index(driver)

    embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
    llm = OpenAILLM(
        model_name="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model_params={"temperature": 0},
    )

    if mode == "before":
        ans_before = answer_question(driver, embedder, llm, QUESTION, "BEFORE")
        with open(CACHE_FILE, "w") as f:
            json.dump({"question": QUESTION, "ans_before": ans_before}, f)
        print(f"[Stage 5] Saved BEFORE answer to {CACHE_FILE}. "
              f"Now run stage4_retrieval_ingestion_FIXED.py, then re-run this script with 'after'.")

    else:  # mode == "after"
        if not os.path.exists(CACHE_FILE):
            print(f"[Stage 5] No {CACHE_FILE} found -- run this script with 'before' first.")
            driver.close()
            sys.exit(1)
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        ans_before = cache["ans_before"]

        ans_after = answer_question(driver, embedder, llm, QUESTION, "AFTER")

        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        judge_answers(client, QUESTION, ans_before, ans_after)

    driver.close()


if __name__ == "__main__":
    main()
