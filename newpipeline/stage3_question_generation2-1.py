"""
MKGE -- OUR PIPELINE -- Stage 3: Question Generation
============================================================
Our improved version of Stage 3. The original, untouched version is what
run A (original MKGE) uses.

For each of the 7 metrics, asks the LLM for fact-seeking questions about
the nodes Stage 2 flagged sparse on that metric. Does NOT write to Neo4j.

Changes vs the original Stage 3:
  [F3] Robust parsing + "ask for 4, keep the best 2":
       - Tolerant parser: accepts "3:", "3.", "3)", "3 -", "Node 3:", "#3:",
         "Q3:", with bullets / **bold** / `code` marks stripped first.
         Every dropped line is printed, and a warning is printed whenever
         fewer than QUESTIONS_PER_METRIC questions survive.
       - The LLM is asked for CANDIDATES_PER_METRIC (4) questions in the
         SAME single call, then select_questions() keeps the best 2:
           1. drop near-duplicates of any question already used this run
              (or already picked in this batch): cosine similarity >=
              DUPLICATE_THRESHOLD with the local MiniLM embedder;
           2. at most one question per node per metric;
           3. prefer nodes NOT yet targeted this run ("cooldown");
           4. prefer questions that name their target node;
           5. otherwise keep the LLM's own order.
         If fewer than 2 survive, keep what survived -- no extra LLM call.
       - A run-level QuestionHistory remembers used questions and
         targeted nodes across metrics. Nodes already covered are listed
         in the prompt so the LLM avoids them up front.
       - Token savings: the long Neo4j elementIds are no longer sent in
         the prompt (the LLM answers with a local number anyway).
  [F4] Node-centred preview: instead of the first 160 characters of the
       node's chunk (identical for every node when they share a chunk),
       take the 160 characters AROUND where the node's name appears.
       Falls back to the chunk start only if the name isn't found.
  [G2] (Oct 22) Goal-aware questions. Stage 4 can pass the USER'S question
       (goal_question). Then:
         - both prompts (batch + regeneration) say the graph is being
           enriched to answer that question and ask the LLM to PREFER gap
           questions whose answers also help answer it -- still one
           question per sparse node, still naming the node (MKGE's core
           idea is unchanged: questions come from sparse nodes);
         - select_questions() ranks candidates by similarity to the user's
           question (rounded to 0.1, after the fresh-node rule), so ties
           still follow the existing rules.
       Without goal_question everything behaves exactly as before.

Imports Stage 2 from THIS folder (stage2_sparsity_metrics1), never the
original's.

Run with (from the project root):
    python newpipeline\\stage3_question_generation2.py
"""

import os
import re

import networkx as nx
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

from stage2_sparsity_metrics1 import (
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

LLM_MODEL = "openai/gpt-oss-120b"

# Condensed from Table 1 of the paper -- the "cognitive labels" sent to
# the LLM alongside each sparse node.
COGNITIVE_LABELS = {
    "clique": "deduction gap -- known facts haven't been linked into a larger conclusion",
    "non_clique": "unknown unknown -- an isolated concept with no connections at all",
    "clustering": "over-specialization -- missing shared-neighbor context",
    "degree": "fluency illusion -- under-connected, possibly obscure or under-explored",
    "betweenness": "bridge missing -- not connecting otherwise distant topics",
    "diameter": "fragmented global view -- part of a small, disconnected pocket of the graph",
    "louvain": "poor module transfer -- isolated within a tiny topic cluster",
}

# Paper value = 5; we keep 2 to fit the Groq free tier.
QUESTIONS_PER_METRIC = 2
# [F3] How many candidates to ask for in the single call, before selecting.
CANDIDATES_PER_METRIC = 4
# [F3] Two questions at least this similar (cosine, MiniLM) count as
# duplicates. Starting value -- check it against real logged questions.
DUPLICATE_THRESHOLD = 0.85
# [F4] Preview length (paper's spec).
PREVIEW_CHARS = 160
# Max sparse nodes listed in one prompt (unchanged from the original).
MAX_NODES_IN_PROMPT = 50


# =============================================================================
# Small helpers
# =============================================================================
def _cosine_similarity(vec_a, vec_b) -> float:
    """1 = same meaning, 0 = unrelated. Plain Python, no numpy."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _goal_clause(goal_question: str = None) -> str:
    """[G2] Extra prompt text telling the LLM what the enrichment is FOR."""
    if not goal_question:
        return ""
    return (
        f'\nThe graph is being enriched so it can answer this user\'s question: "{goal_question}"\n'
        "Among questions that address the gap, PREFER ones whose answer would also help answer "
        "the user's question -- for example by linking the node to the people, works, "
        "organisations, places or facts that question is about. Stay factual: do not assume "
        "connections the node cannot plausibly have.\n"
    )


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace -- used for exact-duplicate checks
    when no embedder is available."""
    return " ".join(text.lower().split())


# =============================================================================
# [F3] Run-level memory of what has already been asked
# =============================================================================
class QuestionHistory:
    """Remembers, for ONE Stage 4 run, every question already used and every
    node already targeted -- across all metrics. Create one at the start of
    the run and pass the same object to every Stage 3 call.
    """

    def __init__(self):
        self.questions = []        # [str]
        self.embeddings = []       # [list[float] | None], parallel to questions
        self.targeted_node_ids = set()
        self.targeted_labels = []  # in order, for the prompt

    def add(self, node_id: str, node_label: str, question: str, embedding=None) -> None:
        self.questions.append(question)
        self.embeddings.append(embedding)
        if node_id not in self.targeted_node_ids:
            self.targeted_node_ids.add(node_id)
            self.targeted_labels.append(node_label)

    def is_duplicate(self, question: str, embedding=None) -> bool:
        """True if `question` is a near-duplicate of anything already used.
        Uses embeddings when available, otherwise exact (normalised) text."""
        if embedding is not None:
            for previous in self.embeddings:
                if previous is not None and _cosine_similarity(embedding, previous) >= DUPLICATE_THRESHOLD:
                    return True
        normalised = _normalise(question)
        return any(_normalise(previous) == normalised for previous in self.questions)


# =============================================================================
# [F4] Node-centred preview
# =============================================================================
def _centred_window(text: str, name: str, width: int = PREVIEW_CHARS) -> str:
    """`width` characters of `text` centred on the first occurrence of `name`
    (case-insensitive). Falls back to the start of the text if `name` isn't
    in it. Whitespace is collapsed first so the preview reads cleanly."""
    text = " ".join(text.split())
    if not text:
        return ""
    position = text.lower().find(name.lower()) if name else -1
    if position < 0:
        return text[:width]
    start = max(0, position + len(name) // 2 - width // 2)
    start = min(start, max(0, len(text) - width))
    return text[start:start + width]


def get_node_preview(driver, node_id: str, fallback_label: str) -> str:
    """[F4] 160-character preview of a node, taken from the chunk text it was
    extracted from, CENTRED on where the node's name appears. If the node
    came from several chunks, the first chunk that actually mentions the
    name is used. Falls back to the node's own label if it has no chunk.
    """
    with driver.session() as session:
        records = session.run(
            "MATCH (n)-[:FROM_CHUNK]-(c:Chunk) "
            "WHERE elementId(n) = $node_id AND c.text IS NOT NULL "
            "RETURN c.text AS text LIMIT 5",
            node_id=node_id,
        ).data()

    texts = [record["text"] for record in records if record["text"]]
    if not texts:
        return fallback_label

    name = fallback_label or ""
    for text in texts:
        if name and name.lower() in " ".join(text.split()).lower():
            return _centred_window(text, name)
    return _centred_window(texts[0], name)


# =============================================================================
# [F3] Tolerant parsing
# =============================================================================
# After cleaning, a valid line looks like "<number><separator> <question>?",
# optionally prefixed with "Node", "Q" or "Question" and/or "#".
_LINE_PATTERN = re.compile(
    r"^(?:node|question|q)?\s*#?\s*(\d+)\s*[:.)\]\-–—]\s*(.+\?)$",
    re.IGNORECASE,
)


def _clean_line(line: str) -> str:
    """Strip markdown noise the LLM sometimes adds: **bold**, `code`,
    leading bullets ("-", "*", "•") and surrounding whitespace."""
    line = line.replace("**", "").replace("`", "").strip()
    line = re.sub(r"^[-*•]+\s*", "", line)
    return line.strip()


def _parse_attributed_line(line: str, indexed_nodes: list):
    """[F3] Parses one LLM output line into (node_id, node_label, question),
    or returns None if it can't be parsed. Accepts "3:", "3.", "3)", "3 -",
    "Node 3:", "#3:", "Q3:", with markdown noise stripped first. Still
    strict that the number points at a real node and that the line is a
    question (ends with "?").
    """
    match = _LINE_PATTERN.match(_clean_line(line))
    if not match:
        return None
    index = int(match.group(1)) - 1  # prompt uses 1-based numbering
    question = match.group(2).strip()
    if not question or not (0 <= index < len(indexed_nodes)):
        return None
    node_id, (node_label, _preview) = indexed_nodes[index]
    return (node_id, node_label, question)


def parse_candidates(text: str, indexed_nodes: list):
    """Parses the whole LLM reply. Returns (candidates, dropped_lines)."""
    candidates, dropped = [], []
    for raw_line in (text or "").splitlines():
        if not raw_line.strip():
            continue
        parsed = _parse_attributed_line(raw_line, indexed_nodes)
        if parsed is None:
            dropped.append(raw_line.strip())
        else:
            candidates.append(parsed)
    return candidates, dropped


# =============================================================================
# [F3] Selecting the best questions
# =============================================================================
def select_questions(candidates: list, k: int = QUESTIONS_PER_METRIC,
                     embedder=None, history: QuestionHistory = None,
                     goal_embedding=None) -> list:
    """[F3] Picks up to `k` questions out of the parsed candidates.

    Rules, in order:
      1. drop near-duplicates of questions already used this run (history)
         or already picked in this batch;
      2. at most one question per node;
      3. prefer nodes not yet targeted this run (cooldown);
      3b. [G2] prefer questions closer to the user's question (similarity
          rounded to 0.1, so small differences don't override rules 4-5);
          only when goal_embedding is given;
      4. prefer questions that name their target node;
      5. otherwise keep the LLM's own order.
    Picked questions are added to `history` (if given), so later metrics
    see them. Returns [(node_id, node_label, question), ...].
    """
    history = history if history is not None else QuestionHistory()

    # Embed each candidate once (if an embedder is available).
    scored = []
    for order, (node_id, node_label, question) in enumerate(candidates):
        embedding = embedder.embed_query(question) if embedder is not None else None
        fresh_node = node_id not in history.targeted_node_ids
        names_target = bool(node_label) and node_label.lower() in question.lower()
        goal = (_cosine_similarity(embedding, goal_embedding)
                if embedding is not None and goal_embedding is not None else None)   # [G2]
        scored.append({
            "node_id": node_id, "node_label": node_label, "question": question,
            "embedding": embedding, "order": order,
            "fresh_node": fresh_node, "names_target": names_target, "goal": goal,
        })

    if goal_embedding is not None:
        for c in scored:
            g = "  -  " if c["goal"] is None else f"{c['goal']:.2f}"
            print(f"    [Select] goal={g} [{c['node_label']}] {c['question']}")

    # Rules 3-5: fresh nodes first, [G2] then closer to the user's question,
    # then target-naming questions, then LLM order.
    scored.sort(key=lambda c: (not c["fresh_node"],
                               -round(c["goal"], 1) if c["goal"] is not None else 0.0,
                               not c["names_target"], c["order"]))

    picked, picked_nodes = [], set()
    batch = QuestionHistory()  # duplicates within this batch
    for c in scored:
        if len(picked) >= k:
            break
        if c["node_id"] in picked_nodes:                                   # rule 2
            print(f"    [Select] skip (node already has a question this metric): {c['question']}")
            continue
        if history.is_duplicate(c["question"], c["embedding"]) or \
                batch.is_duplicate(c["question"], c["embedding"]):        # rule 1
            print(f"    [Select] skip (near-duplicate of an earlier question): {c['question']}")
            continue
        picked.append((c["node_id"], c["node_label"], c["question"]))
        picked_nodes.add(c["node_id"])
        batch.add(c["node_id"], c["node_label"], c["question"], c["embedding"])

    for (node_id, node_label, question), embedding in zip(picked, batch.embeddings):
        history.add(node_id, node_label, question, embedding)

    if len(picked) < k:
        print(f"    [Select] WARNING: only {len(picked)} of {k} question(s) survived selection.")
    return picked


# =============================================================================
# Question generation
# =============================================================================
def generate_questions_for_metric(client, metric: str, sparse_previews: dict,
                                  embedder=None, history: QuestionHistory = None,
                                  goal_question: str = None, goal_embedding=None) -> list:
    """ONE LLM call per metric. [F3] Asks for CANDIDATES_PER_METRIC
    candidates, parses them tolerantly, then select_questions() keeps the
    best QUESTIONS_PER_METRIC.

    Node attribution (unchanged idea): each sparse node gets a short local
    number for this prompt; the LLM answers "<number>: <question>?" and the
    number is mapped back to the real node in Python, so the LLM never has
    to copy a long elementId. [F3] The elementIds are no longer even shown.

    embedder / history are optional so this still works standalone; Stage 4
    should pass both (the same history object for the whole run).

    [G2] goal_question / goal_embedding (optional): the user's question and
    its embedding; adds the goal clause to the prompt and goal ranking to
    select_questions().

    Returns [(node_id, node_label, question), ...].
    """
    if not sparse_previews:
        return []

    indexed_nodes = list(sparse_previews.items())[:MAX_NODES_IN_PROMPT]

    node_list = "\n".join(
        f'{i + 1}. Node "{label}": "{preview}"'
        for i, (node_id, (label, preview)) in enumerate(indexed_nodes)
    )

    covered_clause = ""
    if history is not None and history.targeted_labels:
        covered_clause = (
            "\nAlready covered earlier in this run (prefer OTHER nodes if possible): "
            + ", ".join(history.targeted_labels) + "\n"
        )

    prompt = f"""You are diagnosing gaps in a knowledge graph.

The following nodes were flagged SPARSE on the "{metric}" metric, which signals:
{COGNITIVE_LABELS[metric]}

Sparse nodes:
{node_list}
{covered_clause}{_goal_clause(goal_question)}
Generate EXACTLY {CANDIDATES_PER_METRIC} concise, fact-seeking questions that, if
answered and added to the graph, would directly address this specific gap.
Each question MUST target exactly ONE of the numbered nodes above, name that
node explicitly, and the questions should target DIFFERENT nodes where possible.

Output ONLY, one per line, nothing else -- no preamble:
<node number>: <question ending in "?">

Example line: 2: What city does this entity operate in?
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content

    candidates, dropped = parse_candidates(text, indexed_nodes)
    for line in dropped:
        print(f"    [Parse] dropped unparseable line: {line!r}")
    print(f"    [Parse] {len(candidates)} candidate question(s) parsed for '{metric}'.")

    if goal_question and goal_embedding is None and embedder is not None:
        goal_embedding = embedder.embed_query(goal_question)
    return select_questions(candidates, k=QUESTIONS_PER_METRIC, embedder=embedder, history=history,
                            goal_embedding=goal_embedding if goal_question else None)   # [G2]


def regenerate_question_for_node(client, node_id: str, node_label: str, metric: str, driver,
                                 rejected_question: str = None, embedder=None,
                                 history: QuestionHistory = None,
                                 goal_question: str = None) -> tuple:
    """Generates ONE replacement question for a SINGLE node. Used by Stage 4
    when the Question Memory Gate rejects the original question.

    [F4] uses the node-centred preview; [F3] no elementId in the prompt, and
    the regenerated question is recorded in `history` (if given). If it is a
    near-duplicate of an earlier question, a warning is printed -- no second
    regeneration (keeps cost bounded).

    Returns (node_id, node_label, question).
    """
    preview = get_node_preview(driver, node_id, node_label)

    avoid_clause = (
        f'\nThe previous question "{rejected_question}" was rejected as '
        f"off-topic for this node -- generate a DIFFERENT question, not a "
        f"rephrasing of it."
        if rejected_question else ""
    )

    prompt = f"""You are diagnosing gaps in a knowledge graph.

Node "{node_label}": "{preview}"

This node was flagged SPARSE on the "{metric}" metric, which signals:
{COGNITIVE_LABELS[metric]}
{_goal_clause(goal_question)}
Generate EXACTLY 1 concise, fact-seeking question that, if answered and
added to the graph, would directly address this specific gap for THIS node.
The question must name "{node_label}" and end with "?".{avoid_clause}

Output ONLY the question, nothing else -- no numbering, no preamble.
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content or ""

    question = next(
        (_clean_line(line) for line in text.splitlines() if _clean_line(line).endswith("?")),
        None,
    )
    if question is None:
        question = f"What is {node_label} known for?"

    if history is not None:
        embedding = embedder.embed_query(question) if embedder is not None else None
        if history.is_duplicate(question, embedding):
            print(f"    [Select] WARNING: regenerated question is a near-duplicate of an earlier one: {question}")
        history.add(node_id, node_label, question, embedding)

    return (node_id, node_label, question)


def sparse_previews_for_metric(driver, metric: str, G=None) -> dict:
    """{node_id: (label, preview)} for nodes sparse on `metric` in the
    CURRENT graph (pass G to reuse an already-loaded graph)."""
    G = G if G is not None else load_entity_graph(driver)
    scores = compute_sparsity_metrics(G)
    sparse_flags, _ = flag_sparse_nodes(scores)
    labels = nx.get_node_attributes(G, "label")
    sparse_node_ids = [n for n in sparse_flags if sparse_flags[n][metric]][:MAX_NODES_IN_PROMPT]
    return {
        node_id: (labels.get(node_id, node_id),
                  get_node_preview(driver, node_id, labels.get(node_id, node_id)))
        for node_id in sparse_node_ids
    }


def generate_all_questions(driver, client, embedder=None) -> dict:
    """Standalone run: sparsity + questions for all 7 metrics on the CURRENT
    graph (no ingestion in between). Stage 4 does NOT use this -- it calls
    generate_questions_for_metric itself, once per metric, after re-checking
    the graph. Returns {metric: [(node_id, node_label, question), ...]}.
    """
    G = load_entity_graph(driver)
    history = QuestionHistory()
    all_questions = {}

    for metric in METRIC_NAMES:
        sparse_previews = sparse_previews_for_metric(driver, metric, G=G)
        print(f"[Stage 3] Metric '{metric}': {len(sparse_previews)} sparse node(s) -- generating questions...")
        questions = generate_questions_for_metric(client, metric, sparse_previews,
                                                  embedder=embedder, history=history)
        all_questions[metric] = questions
        for node_id, node_label, question in questions:
            print(f"    - [{node_label}] {question}")
        print()

    return all_questions


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[Setup] Connected to Neo4j.\n")

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    # Local MiniLM embedder for the duplicate check (free, no API call).
    from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
    embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

    all_questions = generate_all_questions(driver, client, embedder=embedder)

    total = sum(len(v) for v in all_questions.values())
    unique_nodes = len({node_id for qs in all_questions.values() for node_id, _, _ in qs})
    print(f"[Stage 3] Done. {total} question(s) across {len(METRIC_NAMES)} metrics, "
          f"targeting {unique_nodes} different node(s).")

    driver.close()


if __name__ == "__main__":
    main()
