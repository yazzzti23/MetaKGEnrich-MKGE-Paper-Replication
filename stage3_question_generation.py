"""
MKGE Paper Replication -- Stage 3: Question Generation
============================================================
Takes the sparse-node flags from Stage 2 and, for EACH of the 7 metrics,
asks the LLM to generate exactly 5 fact-seeking questions targeting the
nodes flagged sparse on that metric. 7 metrics x 5 questions = up to 35
enrichment questions per run (fewer if a metric has zero sparse nodes).

This stage does NOT write to Neo4j -- it only reads the graph (via Stage
2's functions) and talks to the LLM. Stage 4 (next) is what goes back out
to Tavily and writes the answers into Neo4j.

Run with:  python stage3_question_generation.py
(reuses the same .env -- no new API keys needed. Must be in the same
folder as stage2_sparsity_metrics.py, since this imports from it.)
"""

import os

import networkx as nx
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

from stage2_sparsity_metrics import (
    METRIC_NAMES,
    compute_sparsity_metrics,
    flag_sparse_nodes,
    load_entity_graph,
)

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# Condensed from Table 1 of the paper -- these are the "cognitive labels"
# the paper says get sent to the LLM alongside each sparse node
# (their own example: "bridge missing" for low betweenness).
COGNITIVE_LABELS = {
    "clique": "deduction gap -- known facts haven't been linked into a larger conclusion",
    "non_clique": "unknown unknown -- an isolated concept with no connections at all",
    "clustering": "over-specialization -- missing shared-neighbor context",
    "degree": "fluency illusion -- under-connected, possibly obscure or under-explored",
    "betweenness": "bridge missing -- not connecting otherwise distant topics",
    "diameter": "fragmented global view -- part of a small, disconnected pocket of the graph",
    "louvain": "poor module transfer -- isolated within a tiny topic cluster",
}

QUESTIONS_PER_METRIC = 5  # matches the paper exactly


def get_node_preview(driver, node_id: str, fallback_label: str) -> str:
    """160-character preview of a node's content (paper's exact spec),
    pulled from the text chunk it was extracted from. Falls back to the
    node's own label if no chunk text is found.
    """
    with driver.session() as session:
        result = session.run(
            "MATCH (n)-[:FROM_CHUNK]-(c:Chunk) "
            "WHERE elementId(n) = $node_id "
            "RETURN c.text AS text LIMIT 1",
            node_id=node_id,
        )
        record = result.single()
        if record and record["text"]:
            return record["text"][:160]
    return fallback_label


def generate_questions_for_metric(client, metric: str, sparse_previews: dict) -> list:
    """Stage 3 core step: ONE LLM call per metric, asking for exactly 5
    fact-seeking questions targeting the sparse nodes on that metric.
    Plain-text parsing (not structured output) -- mirrors what the paper
    itself describes: 'end each one with "?" so downstream code can parse
    them deterministically.' This also sidesteps the Groq structured-
    output issues we hit in Stage 1.
    """
    if not sparse_previews:
        return []

    node_list = "\n".join(
        f'- Node "{label}" (id={node_id}): "{preview}"'
        for node_id, (label, preview) in sparse_previews.items()
    )

    prompt = f"""You are diagnosing gaps in a knowledge graph.

The following nodes were flagged SPARSE on the "{metric}" metric, which signals:
{COGNITIVE_LABELS[metric]}

Sparse nodes:
{node_list}

Generate EXACTLY {QUESTIONS_PER_METRIC} concise, fact-seeking questions that, if
answered and added to the graph, would directly address this specific gap for
these nodes. Each question MUST end with a "?".
Output ONLY the questions, one per line, nothing else -- no numbering, no preamble.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content
    questions = [line.strip() for line in text.splitlines() if line.strip().endswith("?")]
    return questions[:QUESTIONS_PER_METRIC]


def generate_all_questions(driver, client) -> dict:
    """Runs Stage 2's sparsity detection + Stage 3's question generation
    for all 7 metrics, returns {metric: [questions]}. Pulled out into its
    own function so Stage 4 can import and reuse this directly.
    """
    G = load_entity_graph(driver)
    scores = compute_sparsity_metrics(G)
    sparse_flags, medians = flag_sparse_nodes(scores)
    labels = nx.get_node_attributes(G, "label")

    all_questions = {}

    for metric in METRIC_NAMES:
        sparse_node_ids = [n for n in sparse_flags if sparse_flags[n][metric]][:50]
        sparse_previews = {
            node_id: (
                labels.get(node_id, node_id),
                get_node_preview(driver, node_id, labels.get(node_id, node_id)),
            )
            for node_id in sparse_node_ids
        }

        print(f"[Stage 3] Metric '{metric}': {len(sparse_node_ids)} sparse node(s) -- generating questions...")
        questions = generate_questions_for_metric(client, metric, sparse_previews)
        all_questions[metric] = questions

        for q in questions:
            print(f"    - {q}")
        print()

    return all_questions


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    all_questions = generate_all_questions(driver, client)

    total = sum(len(v) for v in all_questions.values())
    print(f"[Stage 3] Done. Generated {total} enrichment questions across {len(METRIC_NAMES)} metrics.")

    driver.close()


if __name__ == "__main__":
    main()
