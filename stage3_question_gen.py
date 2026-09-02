"""
MKGE Paper Replication -- Stage 3: Question Generation
============================================================
Token-optimized version.

Generates 2 fact-seeking questions per metric instead of the paper's
5, to stay within the Groq free-tier token budget.

This stage does NOT write to Neo4j.
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

COGNITIVE_LABELS = {
    "clique": "deduction gap -- known facts haven't been linked into a larger conclusion",
    "non_clique": "unknown unknown -- an isolated concept with no connections at all",
    "clustering": "over-specialization -- missing shared-neighbor context",
    "degree": "fluency illusion -- under-connected, possibly obscure or under-explored",
    "betweenness": "bridge missing -- not connecting otherwise distant topics",
    "diameter": "fragmented global view -- part of a small, disconnected pocket of the graph",
    "louvain": "poor module transfer -- isolated within a tiny topic cluster",
}

# Token-optimized demo value
QUESTIONS_PER_METRIC = 2


def get_node_preview(driver, node_id: str, fallback_label: str) -> str:
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


def _parse_attributed_line(line: str, indexed_nodes: list):
    if not line.endswith("?"):
        return None

    cleaned = line

    for prefix in ("Node ", "node ", "#"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    for sep in (":", "."):
        if sep in cleaned:
            index_part, _, question_part = cleaned.partition(sep)

            index_part = index_part.strip()
            question_part = question_part.strip()

            if index_part.isdigit() and question_part:
                index = int(index_part) - 1

                if 0 <= index < len(indexed_nodes):
                    node_id, (node_label, _preview) = indexed_nodes[index]
                    return (node_id, node_label, question_part)

            break

    return None


def generate_questions_for_metric(
    client,
    metric: str,
    sparse_previews: dict
) -> list:

    if not sparse_previews:
        return []

    indexed_nodes = list(sparse_previews.items())

    node_list = "\n".join(
        f'{i + 1}. Node "{label}" (id={node_id}): "{preview}"'
        for i, (node_id, (label, preview))
        in enumerate(indexed_nodes)
    )

    prompt = f"""You are diagnosing gaps in a knowledge graph.

The following nodes were flagged SPARSE on the "{metric}" metric, which signals:
{COGNITIVE_LABELS[metric]}

Sparse nodes:
{node_list}

Generate EXACTLY {QUESTIONS_PER_METRIC} concise, fact-seeking questions that, if
answered and added to the graph, would directly address this specific gap.

Each question MUST target exactly ONE of the numbered nodes above.

Output ONLY, one per line:
<node number>: <question ending in "?">

Example:
2: What city does this entity operate in?
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    text = response.choices[0].message.content

    attributed_questions = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        parsed = _parse_attributed_line(line, indexed_nodes)

        if parsed is not None:
            attributed_questions.append(parsed)

    return attributed_questions[:QUESTIONS_PER_METRIC]


def regenerate_question_for_node(
    client,
    node_id: str,
    node_label: str,
    metric: str,
    driver,
    rejected_question: str = None
) -> tuple:

    preview = get_node_preview(driver, node_id, node_label)

    avoid_clause = (
        f'\nThe previous question "{rejected_question}" was rejected as '
        f"off-topic for this node -- generate a DIFFERENT question, not a "
        f"rephrasing of it."
        if rejected_question else ""
    )

    prompt = f"""You are diagnosing gaps in a knowledge graph.

Node "{node_label}" (id={node_id}): "{preview}"

This node was flagged SPARSE on the "{metric}" metric, which signals:
{COGNITIVE_LABELS[metric]}

Generate EXACTLY 1 concise, fact-seeking question that, if answered and
added to the graph, would directly address this specific gap for THIS node.

The question must end with "?".{avoid_clause}

Output ONLY the question, nothing else -- no numbering, no preamble.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    text = response.choices[0].message.content.strip()

    question = next(
        (
            line.strip()
            for line in text.splitlines()
            if line.strip().endswith("?")
        ),
        None,
    )

    if question is None:
        question = f"What is {node_label} known for?"

    return (node_id, node_label, question)


def generate_all_questions(driver, client) -> dict:

    G = load_entity_graph(driver)

    scores = compute_sparsity_metrics(G)

    sparse_flags, medians = flag_sparse_nodes(scores)

    labels = nx.get_node_attributes(G, "label")

    all_questions = {}

    for metric in METRIC_NAMES:

        sparse_node_ids = [
            n
            for n in sparse_flags
            if sparse_flags[n][metric]
        ][:50]

        sparse_previews = {
            node_id: (
                labels.get(node_id, node_id),
                get_node_preview(
                    driver,
                    node_id,
                    labels.get(node_id, node_id)
                ),
            )
            for node_id in sparse_node_ids
        }

        print(
            f"[Stage 3] Metric '{metric}': "
            f"{len(sparse_node_ids)} sparse node(s) "
            f"-- generating questions..."
        )

        attributed_questions = generate_questions_for_metric(
            client,
            metric,
            sparse_previews
        )

        all_questions[metric] = attributed_questions

        for node_id, node_label, question in attributed_questions:
            print(f"    - [{node_label}] {question}")

        print()

    return all_questions


def main():

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    driver.verify_connectivity()

    print("[Setup] Connected to Neo4j.\n")

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    all_questions = generate_all_questions(
        driver,
        client
    )

    total = sum(
        len(v)
        for v in all_questions.values()
    )

    print(
        f"[Stage 3] Done. Generated {total} "
        f"enrichment questions across "
        f"{len(METRIC_NAMES)} metrics."
    )

    driver.close()


if __name__ == "__main__":
    main()