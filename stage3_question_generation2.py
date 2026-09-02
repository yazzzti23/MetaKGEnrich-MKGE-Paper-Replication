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

# Paper value = 5
# Demo value = 2 (to fit Groq free-tier quota)
QUESTIONS_PER_METRIC = 2


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


def _parse_attributed_line(line: str, indexed_nodes: list):
    """
    Parses one LLM output line, expected in "<node number>: <question>?"
    form, into (node_id, node_label, question).

    Tolerant of minor format drift Groq's plain-text output sometimes
    produces (a leading "Node "/"#", or "." instead of ":" after the
    number) -- but strict about requiring SOME leading integer that
    resolves to a valid index into indexed_nodes, and about the line
    actually ending in "?" (same parse-safety convention already used
    elsewhere in this file: "end each one with '?' so downstream code
    can parse them deterministically").

    Returns None (silently dropped, not crashed) if the line doesn't
    parse cleanly -- one malformed line here should never take down the
    whole batch of otherwise-valid questions.
    """
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
                index = int(index_part) - 1  # prompt uses 1-based numbering
                if 0 <= index < len(indexed_nodes):
                    node_id, (node_label, _preview) = indexed_nodes[index]
                    return (node_id, node_label, question_part)
            break  # only try the first separator that's actually present

    return None


def generate_questions_for_metric(client, metric: str, sparse_previews: dict) -> list:
    """Stage 3 core step: ONE LLM call per metric, asking for exactly
    QUESTIONS_PER_METRIC fact-seeking questions targeting the sparse
    nodes on that metric. Plain-text parsing (not structured output) --
    mirrors what the paper itself describes: 'end each one with "?" so
    downstream code can parse them deterministically.' This also
    sidesteps the Groq structured-output issues hit in Stage 1.

    Returns
    -------
    list[tuple[str, str, str]]
        (node_id, node_label, question) triples -- one per generated
        question, attributed to the specific sparse node it targets.

    NODE ATTRIBUTION -- design note
    ---------------------------------
    Earlier versions of this function returned bare question strings
    with no link back to which sparse node a question was actually
    about -- fine for Stage 4's old batched ingest-everything loop, but
    not enough for the Question Memory Gate (Phase 3a), which needs a
    specific node_id/node_label per question to evaluate against.

    Rather than asking the LLM to echo back each node's real Neo4j
    elementId verbatim (long, opaque strings like
    "4:b4a4a831-...:115" -- exactly the kind of token an LLM can
    silently typo or truncate, which would then fail node_id lookups
    downstream with no obvious error at the point of failure), this
    function assigns each sparse node a short LOCAL index (1, 2, 3...)
    purely for this one prompt, has the LLM reference that index
    instead of the real id, then resolves index -> real node_id back in
    Python using the exact same order the prompt was built in. The LLM
    never has to reproduce an id string at all, so there's nothing for
    it to get wrong there.
    """
    if not sparse_previews:
        return []

    # Fixed order, used for BOTH building the prompt and parsing the
    # response -- this is what makes "2: ..." in the LLM's output
    # unambiguously resolve back to a real (node_id, label) pair below.
    indexed_nodes = list(sparse_previews.items())  # [(node_id, (label, preview)), ...]

    node_list = "\n".join(
        f'{i + 1}. Node "{label}" (id={node_id}): "{preview}"'
        for i, (node_id, (label, preview)) in enumerate(indexed_nodes)
    )

    prompt = f"""You are diagnosing gaps in a knowledge graph.

The following nodes were flagged SPARSE on the "{metric}" metric, which signals:
{COGNITIVE_LABELS[metric]}

Sparse nodes:
{node_list}

Generate EXACTLY {QUESTIONS_PER_METRIC} concise, fact-seeking questions that, if
answered and added to the graph, would directly address this specific gap.
Each question MUST target exactly ONE of the numbered nodes above.

Output ONLY, one per line, nothing else -- no numbering commentary, no preamble:
<node number>: <question ending in "?">

Example line: 2: What city does this entity operate in?
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


def regenerate_question_for_node(client, node_id: str, node_label: str, metric: str, driver,
                                  rejected_question: str = None) -> tuple:
    """
    Generates ONE replacement fact-seeking question for a SINGLE sparse
    node -- reuses this file's existing conventions (COGNITIVE_LABELS,
    get_node_preview, same Groq-hosted plain-text LLM call pattern as
    generate_questions_for_metric) but scoped to exactly one node
    instead of a batch.

    Used by Stage 4's retry path: when question_memory_gate() rejects
    the original attributed question for a node, Stage 4 calls this
    ONCE to get a single replacement question for that SAME node, then
    re-evaluates through the gate. This is a separate, single-node LLM
    call made only on the reject path (never unconditionally), so the
    added call volume scales with how often the gate actually rejects,
    not with every metric cycle by default.

    Parameters
    ----------
    client : openai.OpenAI
        Same Groq-backed client already used throughout Stage 3/4.
    node_id : str
        The SAME node_id the rejected question targeted -- preserved
        through to the return value, never re-derived or guessed.
    node_label : str
        The SAME node_label -- also preserved through unchanged.
    metric : str
        Which sparsity metric this node was originally flagged under
        (Stage 4 already has this in scope from its per-metric loop),
        used to pull the same COGNITIVE_LABELS framing the original
        batch question used.
    driver : neo4j.Driver
        Used only for get_node_preview's chunk-text lookup.
    rejected_question : str, optional
        The question that was just rejected. If given, it's included
        in the prompt so the LLM generates something genuinely
        different rather than a near-rephrasing of the same question
        that already failed the gate.

    Returns
    -------
    tuple[str, str, str]
        (node_id, node_label, question) -- same shape as one entry from
        generate_questions_for_metric's return list, so Stage 4 can
        handle a regenerated question the same way it handles an
        original one.
    """
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

    # Same parse-safety convention as the rest of this file: take the
    # first line that actually ends in "?", ignore any stray preamble.
    question = next(
        (line.strip() for line in text.splitlines() if line.strip().endswith("?")),
        None,
    )

    if question is None:
        # Nothing parseable came back -- fall back to a generic,
        # node-grounded question (same fallback convention used
        # elsewhere in this project, e.g. test_memory_gate.py's
        # "What is {target_label} known for?" for a node with no
        # relation facts) rather than returning an invalid/empty
        # question that would just fail the gate again for an
        # unrelated reason.
        question = f"What is {node_label} known for?"

    return (node_id, node_label, question)


def generate_all_questions(driver, client) -> dict:
    """Runs Stage 2's sparsity detection + Stage 3's question generation
    for all 7 metrics, returns {metric: [(node_id, node_label, question), ...]}.
    Pulled out into its own function so Stage 4 can import and reuse this
    directly.

    NOTE: as of the node-attribution change (see generate_questions_for_metric's
    docstring), each metric's value is now a list of (node_id, node_label,
    question) triples, not bare question strings -- any caller (Stage 4)
    iterating this needs to unpack accordingly.
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
        attributed_questions = generate_questions_for_metric(client, metric, sparse_previews)
        all_questions[metric] = attributed_questions

        for node_id, node_label, question in attributed_questions:
            print(f"    - [{node_label}] {question}")
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