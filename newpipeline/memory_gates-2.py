"""
MKGE -- OUR PIPELINE -- Memory Gates (Question Gate + Post-Extraction Gate)
============================================================
Changes vs the previous version:
  [F1] _get_direct_neighbors names neighbours with the shared naming rule
       (cypher_label from stage2_sparsity_metrics1) -- no more "unnamed".
  [S1] LLM tiebreak reads the FIRST WORD of the reply exactly. Before, any
       first line merely containing "accept" counted as accept, so
       "do not accept" was read as accept.
  [S2] Name matching in the entity signal is whole-word, and for
       multi-word names the last word alone also counts (so "Portman"
       matches "Natalie Portman"). Before: raw substring, so "Portman"
       alone scored 0 and short names could match inside other words.
  [S9] Removed dead code: assign_question_to_sparse_node (replaced by
       Stage 3's numbered attribution) and MAX_REGENERATION_ATTEMPTS
       (never used; Stage 4 allows exactly ONE regeneration).
  [PG2] Post-Extraction Gate v2 (post_extraction_filter, at the end of this
       file): judges each extracted ENTITY and FACT instead of the whole
       page. Keeps entities that are an anchor (the target node or a
       cluster member) or connected to one within POST_V2_MAX_HOPS links
       in the extraction; keeps facts whose two ends were kept; skips facts
       that already exist in the graph; logs (does not act on) conflicts on
       one-value relations; rejects the page only if nothing new survives.
       Per-fact relevance (meaning similarity to the question) is computed
       and LOGGED only -- not used to drop anything yet.
       The old post_extraction_memory_gate() is kept below for reference
       (legacy, no longer called by the pipeline).
  [G3] Goal anchoring of the Question Gate (Oct 22). q2 showed both gates
       only ask "does this fit the SEED GRAPH?", never "does this help
       answer the USER'S question?", so an off-target seed got reinforced.
       question_memory_gate() now takes the user's question (optional):
         - 5th signal "goal" = cosine(candidate, user's question);
         - goal < GOAL_FLOOR  -> reject, no LLM call (path "goal_floor");
         - in the tiebreak band and goal >= GOAL_STRONG -> accept, no LLM
           call (path "goal_strong");
         - the LLM tiebreak prompt includes the user's question.
       Without a user's question everything behaves exactly as before.
  [G4] PG2 no longer uses bare dates / numbers ("1871", "3.3 million
       viewers", "August 17, 2007") as anchors: in q2 a Chicago Fire 1871
       page got through only because it shared the year 1871. Such entities
       can still be KEPT when connected to a real (named) anchor.
       post_extraction_filter() also logs each fact's similarity to the
       user's question ("goal_relevance"), log only.

Original notes follow.

Implements Phase 3a of the memory-augmented MKGE design: the Question
Memory Gate. Evaluates a candidate enrichment question against the
target sparse node's Cluster Memory BEFORE that question is allowed to
trigger a Tavily search + SimpleKGPipeline ingestion in Stage 4 -- the
goal is to catch semantic drift before it happens, not after.

SCOPE OF THIS FILE
-------------------
This file implements BOTH gates (Question Memory Gate, and the
Post-Extraction Gate further down). It does not implement:
  - rejected-exemplar pruning (that lives in cluster_memory.py)
  - Stage 3 / Stage 4 wiring -- question_memory_gate() below is called
    BY Stage 4, but Stage 4 itself is not modified by this file.
Nothing in this file mutates a ClusterMemory object. question_memory_gate()
is a pure evaluator -- writing an AcceptedExemplar or RejectedExemplar
based on its verdict is the caller's (Stage 4's) responsibility, exactly
as scoped in the project manifesto.

EMBEDDER REUSE
--------------
This file deliberately does NOT construct a SentenceTransformerEmbeddings
instance anywhere. Every function that needs an embedding takes the
already-loaded embedder in as a parameter and calls its embed_query(text)
method (confirmed against neo4j_graphrag.embeddings.Embedder's interface
-- embed_query, not embed() or encode()). This matches the existing
codebase's pattern of loading all-MiniLM-L6-v2 exactly once and reusing
that instance across SimpleKGPipeline ingestion and Stage 5's
VectorRetriever.

DESIGN NOTE -- cluster_member_node_ids
---------------------------------------
ClusterMemory (cluster_memory.py) stores each cluster's entities as
human-readable LABELS (RepresentativeMemory.representative_entities),
never as raw Neo4j elementIds. compute_cluster_embedding(), however,
needs the raw member_node_ids to look up each member's Chunk.embedding.
Since ClusterMemory has no node-id list to fall back on,
question_memory_gate() below takes cluster_member_node_ids as its own
parameter. Stage 4 already builds a cluster_id -> [member_node_ids] dict
as the first step of its planned integration (see project manifesto,
"Immediate Next Steps"), so this is passing through data Stage 4 already
has -- not new plumbing.

NO NUMPY -- cosine similarity below is plain Python, matching this
project's existing convention (cluster_memory.py's embedding average is
also plain Python, no numpy).
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Constants (placeholder values -- explicitly flagged as needing tuning
# against real gate-call data once Stage 4 integration produces some;
# see project manifesto, Source of Truth section)
# =============================================================================
# [G3] Five signals. The four old weights keep their old 3:3:2:2 ratio
# (0.30/0.30/0.20/0.20 scaled by 0.70), so without a user's question the
# "goal" signal is None, is dropped by the cold-start renormalisation in
# _combine_signals, and the score is EXACTLY the old 4-signal score.
W_GOAL = 0.30
W_SEMANTIC = 0.21
W_ENTITY = 0.21
W_ACCEPTED = 0.14
W_REJECTED = 0.14

T_ACCEPT = 0.70
T_REJECT = 0.35

# [G3] goal = cosine(candidate question, user's question), MiniLM.
# Calibrated on the q1/q2 runs (runs/goal_relevance_analysis.txt):
#   - clearly off-goal questions all sit below 0.20 (coastline paradox 0.00,
#     Great Fire of 1871 0.00, Watters Primetime 0.08, Eastern Time Zone
#     0.05/0.09, PS3 0.07, "1871" 0.18)                -> GOAL_FLOOR;
#   - on-goal questions sit at 0.60-0.85 ("premiere of HSM2" 0.82, "Carroll's
#     work vs the game's IP" 0.61), and in q1 the cluster-only LLM tiebreak
#     rejected many of them                            -> GOAL_STRONG.
# Known cost of the floor: "What novel precedes Through the Looking-Glass?"
# (0.19) would also be rejected.
GOAL_FLOOR = 0.20
GOAL_STRONG = 0.60

# Entity-consistency signal's INTERNAL target/neighbour blend (Option 2a,
# chosen after comparing Option 1 -- tiered match -- against Option 2 at
# two weightings, via compare_entity_signal.py; see that script and
# Question_Memory_Gate.pdf sections 6-8 for the full comparison and
# rationale). Separate from W_ENTITY above: W_ENTITY weights this
# signal's OUTPUT against the other three signals in _combine_signals;
# these two weight what goes INTO that output in the first place.
ENTITY_W_TARGET = 0.5
ENTITY_W_NEIGHBOUR = 0.5

# [S9] Retry policy lives in Stage 4: ONE regeneration, i.e. at most 2 gate
# evaluations per question. (The old MAX_REGENERATION_ATTEMPTS = 2 constant
# was never used and did not match that policy, so it was removed.)


# =============================================================================
# Result type
# =============================================================================
@dataclass
class GateResult:
    """
    Everything question_memory_gate() reports back to its caller (Stage 4).

    signals holds the RAW, unweighted value of each of the four signals
    -- a value of None means that signal had no data and was excluded
    from the weighted average entirely (cold-start handling), never
    silently defaulted to 0.5. confidence_score is always the
    deterministic combined score, even when decision_path is
    "llm_tiebreak" -- the LLM never self-reports its own confidence
    (per the finalized design, this was found unreliable), it only
    supplies the accept/reject verdict when the deterministic score
    alone is inconclusive.
    """
    decision: str                 # "accept" or "reject"
    confidence_score: float       # combined weighted score, deterministic-derived, always present
    signals: dict                 # {"semantic", "entity", "goal" [G3], "accepted", "rejected"}: float|None
    decision_path: str            # "deterministic", "llm_tiebreak", [G3] "goal_floor", "goal_strong"
    attempt_number: int           # 1, 2, or 3 -- which gate call this was for this node


# =============================================================================
# Small helpers
# =============================================================================
def _cosine_similarity(vec_a, vec_b) -> float:
    """Plain Python cosine similarity -- no numpy, matching this
    project's existing convention (see cluster_memory.py's
    compute_cluster_embedding, which averages vectors the same way).
    Returns 0.0 for degenerate inputs (None, empty, or zero-norm vectors)
    rather than raising, since callers treat "no meaningful similarity"
    the same as "couldn't compute one" for this gate's purposes.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity = dot / (norm_a * norm_b)
    # Clip to [0, 1] for the weighted-average combination downstream --
    # cosine similarity can technically go negative, but a negative
    # value and a zero value both mean "not supportive," and the
    # weighted-average combiner expects every signal on the same
    # [0, 1] support scale.
    return max(0.0, min(1.0, similarity))


def _get_direct_neighbors(driver, node_id: str) -> list:
    """
    1-hop Neo4j neighbor labels for one __Entity__ node. Same
    __Entity__-to-__Entity__ pattern and same coalesce(name, id,
    'unnamed') convention already used by stage2_sparsity_metrics.py's
    load_entity_graph(), so this returns labels consistent with what
    the rest of the pipeline already calls each node.

    Used by _entity_consistency_signal below to score the neighbour-
    coverage half of its target/neighbour blend -- originally added to
    handle cases like "Metals & Mining" being genuinely connected to two
    different people's clusters but only "owned" by one Representative
    Memory after Louvain partitioning; still relevant under the current
    target+neighbour-coverage formula even though that formula no longer
    reads Representative Memory directly (see _entity_consistency_signal's
    docstring for the full history).
    """
    # [F1] Shared naming rule (imported here, not at the top, so this file
    # stays importable without Stage 2's Neo4j settings -- same pattern as
    # compute_cluster_embedding below).
    from stage2_sparsity_metrics1 import cypher_label

    with driver.session() as session:
        result = session.run(
            "MATCH (n:__Entity__)-[r]-(m:__Entity__) "
            "WHERE elementId(n) = $node_id "
            f"RETURN DISTINCT {cypher_label('m')} AS label",
            node_id=node_id,
        )
        return [record["label"] for record in result]


# [S2] Whole-word name matching.
_NAME_STOPWORDS = {"the", "a", "an", "of", "and"}


def _mentions(name: str, text: str) -> bool:
    """[S2] Does `text` mention `name`? Case-insensitive and WHOLE-WORD, so a
    short name can't match inside another word ("Pi" no longer matches
    "pilot"). For multi-word names, the last word alone also counts if it
    has at least 4 letters -- so "Portman" matches "Natalie Portman" and
    "Swan" matches "Black Swan".
    """
    if not name or not text:
        return False
    text_l = text.lower()
    name_l = " ".join(name.lower().split())
    if re.search(r"(?<!\w)" + re.escape(name_l) + r"(?!\w)", text_l):
        return True
    words = [w for w in re.findall(r"\w+", name_l) if w not in _NAME_STOPWORDS]
    if len(name_l.split()) >= 2 and words and len(words[-1]) >= 4:
        return re.search(r"(?<!\w)" + re.escape(words[-1]) + r"(?!\w)", text_l) is not None
    return False


# =============================================================================
# The four deterministic signals
# =============================================================================
def _semantic_similarity_signal(question_embedding, cluster_member_node_ids: list, driver) -> Optional[float]:
    """
    How semantically close is the candidate question to this cluster's
    overall topic? Computed by comparing the question's embedding
    against the cluster's on-demand embedding (compute_cluster_embedding,
    cluster_memory.py -- averages the Chunk.embedding values already
    written by SimpleKGPipeline for this cluster's member entities).

    Returns None (cold start, excluded from the average) if the cluster
    has no member node with an embedded chunk yet -- e.g. immediately
    after seeding, before any ingestion has run.
    """
    # Imported here rather than at module top level to keep this file's
    # only hard dependency on cluster_memory.py explicit and localized.
    from cluster_memory import compute_cluster_embedding

    cluster_embedding = compute_cluster_embedding(driver, cluster_member_node_ids)
    if cluster_embedding is None:
        return None

    return _cosine_similarity(question_embedding, cluster_embedding)


def _entity_consistency_signal(question: str, node_id: str, node_label: str, driver) -> Optional[float]:
    """
    Does the candidate question actually mention the target node itself,
    or its immediate graph neighborhood?

    score = ENTITY_W_TARGET * target_match + ENTITY_W_NEIGHBOUR * neighbour_coverage
      target_match       = 1.0 if node_label appears in the question, else 0.0
      neighbour_coverage  = (# direct 1-hop neighbors mentioned) / (# direct
                             1-hop neighbors), 0.0 if the node has none

    CHANGE OF DESIGN (superseded the original pool-size formula)
    --------------------------------------------------------------
    The original implementation divided by
    len(union(cluster's Representative Memory entities, node's direct
    neighbors)) -- meaning a perfectly relevant question was penalized
    purely because its cluster happened to be large (4 cluster entities,
    1 match -> 0.25; 20 cluster entities, same 1 match -> 0.05), even
    though relevance to the TARGET node hadn't changed at all.

    This version drops cluster size from the denominator entirely: it
    only ever divides by the target node's OWN direct-neighbor count,
    so it no longer shrinks as unrelated parts of the cluster grow.
    compute_entity_signal was compared at this weighting (Option 2a,
    0.5/0.5) against a 0.7/0.3 target-prioritized weighting (Option 2b)
    and a simpler tiered scheme (Option 1: target=1.0, any neighbor=0.5,
    neither=0.0) in compare_entity_signal.py. Option 2b was rejected
    because a genuinely relevant question that names only a neighbor,
    never the target (0 target_match, full neighbour_coverage), tops out
    at 0.7*0 + 0.3*1.0 = 0.30 -- still below T_REJECT (0.35), recreating
    the exact under-scoring failure this change exists to fix. Option 1
    and Option 2a scored identically on the real seed graph tested (every
    node there had exactly one direct neighbor, so neighbour_coverage
    could only be 0 or 1 -- collapsing Option 2a onto Option 1's same
    three values); Option 2a was chosen anyway since it's strictly more
    expressive the moment a node has 2+ neighbors (partial coverage,
    e.g. 2 of 3 mentioned -> 0.5*(2/3), where Option 1 would still
    flatten that to 0.5), which will start mattering once Stage 4
    enrichment grows clusters past their current seed-stage size.

    This also means cluster_memory / representative_entities are no
    longer read by this signal at all -- dropped from the signature
    (not passed-but-ignored) so the interface stays honest about what
    data this signal actually needs now, matching this file's existing
    convention (see the module docstring's cluster_member_node_ids note).

    [S2] Matching is case-insensitive and whole-word via _mentions(); for
    multi-word names the last word alone also counts.

    Returns None (cold start, excluded from the average) only in the
    degenerate case where node_label itself is empty/falsy AND the node
    has no direct neighbors -- i.e. there is no information at all to
    score against. In ordinary operation node_label is always a real
    string, so this branch is a defensive guard rather than an expected
    path.
    """
    neighbours = _get_direct_neighbors(driver, node_id)

    if not node_label and not neighbours:
        return None

    # [S2] whole-word matching, surname alone counts (see _mentions).
    target_match = 1.0 if _mentions(node_label, question) else 0.0

    if neighbours:
        matched_neighbours = [n for n in neighbours if _mentions(n, question)]
        neighbour_coverage = len(matched_neighbours) / len(neighbours)
    else:
        neighbour_coverage = 0.0

    return ENTITY_W_TARGET * target_match + ENTITY_W_NEIGHBOUR * neighbour_coverage


def _accepted_exemplar_signal(question_embedding, cluster_memory, embedder) -> Optional[float]:
    """
    How similar is the candidate question to this cluster's past
    SUCCESSFUL enrichment questions? High similarity to a question that
    previously led to a good, accepted extraction is supportive
    evidence for accepting this one too.

    Aggregation: takes the MAX similarity across all accepted exemplars
    in this cluster (closest past success), not an average -- one
    strong match to a known-good precedent is meaningful support even
    if the cluster's other accepted exemplars look quite different.

    Returns None (cold start, excluded from the average) if this
    cluster has no accepted exemplars yet -- true for every cluster
    before the very first successful enrichment cycle.
    """
    accepted = cluster_memory.exemplars.accepted
    if not accepted:
        return None

    best = 0.0
    for exemplar in accepted:
        exemplar_embedding = embedder.embed_query(exemplar.generated_question)
        similarity = _cosine_similarity(question_embedding, exemplar_embedding)
        best = max(best, similarity)

    return best


def _rejected_exemplar_signal(question_embedding, cluster_memory, embedder) -> Optional[float]:
    """
    How similar is the candidate question to this cluster's past FAILED
    enrichment questions? Returns the RAW similarity to the closest
    rejected exemplar (max across the cluster's rejected list) -- NOT
    yet converted to a "support" value. The 1 - similarity conversion
    (rejected-exemplar similarity is a weighted NEGATIVE signal) happens
    inside _combine_signals, so the raw value logged here stays directly
    interpretable on its own ("how close is this to something that
    already failed"), separate from how it gets folded into the combined
    score.

    Same MAX aggregation rationale as _accepted_exemplar_signal: one
    close match to a known failure is meaningful evidence against
    proceeding, even if the cluster's other rejections look different.

    Returns None (cold start, excluded from the average) if this
    cluster has no rejected exemplars yet.
    """
    rejected = cluster_memory.exemplars.rejected
    if not rejected:
        return None

    best = 0.0
    for exemplar in rejected:
        exemplar_embedding = embedder.embed_query(exemplar.generated_question)
        similarity = _cosine_similarity(question_embedding, exemplar_embedding)
        best = max(best, similarity)

    return best


# =============================================================================
# Combination
# =============================================================================
def _combine_signals(raw_signals: dict) -> float:
    """
    Weighted combination of the four signals, with cold-start handling:
    a signal with no data (value is None) is excluded ENTIRELY from the
    average, and the remaining weights are renormalized -- never
    defaulted to a neutral 0.5, per the finalized design.

    The rejected-exemplar signal is converted from raw similarity to a
    "support" value (support = 1 - similarity) here, right before
    weighting -- it's the one signal where HIGH raw value means LOW
    support for accepting.

    Edge case not covered by the finalized design: if ALL FOUR signals
    are cold-start (e.g. a brand-new cluster with no chunk embeddings
    yet AND no exemplars of either kind -- possible right after Stage 4
    starts, before this cluster has been touched at all), there is
    nothing to average. Rather than inventing a default accept/reject,
    this returns the exact midpoint of the tiebreak band
    ((T_ACCEPT + T_REJECT) / 2), which guarantees the caller routes to
    the LLM tiebreak -- i.e. "we have no deterministic evidence either
    way, let the one-time LLM call decide" -- rather than silently
    defaulting to accept or reject. Flagged here as a deliberate design
    choice, not something pinned down in the original spec.

    Parameters
    ----------
    raw_signals : dict
        {"semantic": float|None, "entity": float|None,
         "accepted": float|None, "rejected": float|None} -- exactly the
        shape produced by question_memory_gate() below.

    Returns
    -------
    float
        Combined score in [0, 1].
    """
    weights = {
        "semantic": W_SEMANTIC,
        "entity": W_ENTITY,
        "goal": W_GOAL,            # [G3]
        "accepted": W_ACCEPTED,
        "rejected": W_REJECTED,
    }

    support = {}
    if raw_signals.get("semantic") is not None:
        support["semantic"] = raw_signals["semantic"]
    if raw_signals.get("entity") is not None:
        support["entity"] = raw_signals["entity"]
    if raw_signals.get("goal") is not None:                     # [G3]
        support["goal"] = raw_signals["goal"]
    if raw_signals.get("accepted") is not None:
        support["accepted"] = raw_signals["accepted"]
    if raw_signals.get("rejected") is not None:
        support["rejected"] = 1.0 - raw_signals["rejected"]

    if not support:
        return (T_ACCEPT + T_REJECT) / 2.0

    weight_sum = sum(weights[k] for k in support)
    combined = sum(support[k] * weights[k] for k in support) / weight_sum

    return combined


# =============================================================================
# LLM tiebreak (only place an LLM is called in this gate)
# =============================================================================
def _llm_tiebreak(llm_client, question: str, node_label: str, cluster_memory,
                  goal_question: Optional[str] = None) -> str:
    """
    Called ONLY when the deterministic combined score falls strictly
    between T_REJECT and T_ACCEPT -- i.e. the four signals genuinely
    disagree or are too sparse to decide alone. Asks the LLM a single
    yes/no question: given this cluster's topic, does this candidate
    enrichment question belong?

    Plain-text output, not structured output -- matches this codebase's
    existing Groq convention (see stage1_seed_kg.py's
    llm.supports_structured_output = False comment, and stage3/stage5's
    "end each line with a specific token so downstream code can parse
    deterministically" pattern).

    The LLM's answer is used ONLY for the accept/reject verdict -- it
    does NOT supply a confidence score. GateResult.confidence_score is
    always the deterministic combined score computed before this
    function is even called, per the finalized design (LLM self-
    reported confidence was found unreliable).

    Returns
    -------
    str
        "accept" or "reject", read from the first word of the reply [S1].
        Defaults to "reject" if the LLM's response can't be parsed as either -- the safer failure direction, since
        an unparseable tiebreak response is itself a signal something
        is off, and this design never terminates the pipeline on a
        single node's rejection (Stage 4 just skips that node and
        continues).
    """
    rep = cluster_memory.representative
    entities = ", ".join(rep.representative_entities) if rep.representative_entities else "(none)"

    if goal_question:
        # [G3] The tiebreak also knows WHY the graph is being enriched.
        prompt = f"""You are validating a candidate enrichment question before it triggers a web search and graph write. The knowledge graph is being enriched so that it can answer the user's question below.

User's question: {goal_question}

Cluster topic: {rep.topic_summary}
Cluster's known entities: {entities}
Target node: {node_label}

Candidate question: {question}

Accept if answering the candidate question would plausibly help answer the user's question, directly or by adding context about entities the user's question involves, AND it is about the target node. Reject if it leads away from the user's question (an unrelated topic, or a detail that cannot matter for it). Respond with exactly one word on the first line: "accept" or "reject". Then give a one-sentence reason on the next line.
"""
    else:
        prompt = f"""You are validating whether a candidate enrichment question belongs to a specific knowledge-graph cluster before it triggers a web search and graph write.

Cluster topic: {rep.topic_summary}
Cluster's known entities: {entities}
Target node: {node_label}

Candidate question: {question}

Does this question genuinely belong to this cluster's topic, and would answering it plausibly add relevant, on-topic information about the target node? Respond with exactly one word on the first line: "accept" or "reject". Then give a one-sentence reason on the next line.
"""

    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    # [S1] Read the FIRST WORD of the first non-empty line, exactly.
    # Markdown/punctuation is ignored ("**Accept.**" -> accept), and a label
    # like "Decision:" / "Answer:" / "Verdict:" is skipped. Anything else
    # (including "do not accept") is unparseable -> reject.
    text = (response.choices[0].message.content or "").strip().lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = re.sub(r"[^a-z]+", " ", lines[0]).split() if lines else []
    if words and words[0] in ("decision", "answer", "verdict"):
        words = words[1:]
    first_word = words[0] if words else ""

    if first_word == "accept":
        return "accept"
    if first_word == "reject":
        return "reject"
    return "reject"  # unparseable -- fail toward the safer, skip-this-node direction


# =============================================================================
# Logging
# =============================================================================
def _log_gate_decision(node_id: str, node_label: str, question: str, raw_signals: dict,
                        combined_score: float, decision: str, decision_path: str,
                        attempt_number: int) -> None:
    """print()-based logging, matching this codebase's existing style
    (no logging module used anywhere else in the pipeline). Logs every
    raw signal value (None shown explicitly, not hidden), the combined
    score, the decision, which path produced it, and the attempt number
    -- everything the finalized design requires be visible on every
    gate call.
    """
    print(f"[Question Memory Gate] node={node_label!r} (id={node_id}) attempt={attempt_number}")
    print(f"    question: {question}")
    goal = raw_signals.get("goal")
    print(f"    signals : semantic={raw_signals.get('semantic')}  "
          f"entity={raw_signals.get('entity')}  "
          f"goal={'None' if goal is None else round(goal, 4)}  "                 # [G3]
          f"accepted={raw_signals.get('accepted')}  "
          f"rejected={raw_signals.get('rejected')}")
    print(f"    combined_score={combined_score:.4f}  decision={decision}  path={decision_path}")


# =============================================================================
# Top-level gate
# =============================================================================
def question_memory_gate(question: str, node_id: str, node_label: str, cluster_memory,
                          cluster_member_node_ids: list, driver, embedder, llm_client,
                          attempt_number: int = 1, goal_question: Optional[str] = None,
                          goal_embedding=None) -> GateResult:
    """
    Evaluates ONE candidate enrichment question against ONE sparse
    node's Cluster Memory. Pure evaluator -- does NOT mutate
    cluster_memory (no exemplar is written here; that's the caller's
    responsibility once a final decision, including after regeneration
    attempts are exhausted, is known).

    Decision rule:
      combined_score >= T_ACCEPT  -> accept, deterministic
      combined_score <= T_REJECT  -> reject, deterministic
      otherwise                   -> LLM tiebreak decides accept/reject,
                                      but confidence_score stays the
                                      deterministic combined_score

    Parameters
    ----------
    question : str
        Candidate enrichment question (either Stage 3's original batch
        question, or a regenerated one from
        stage3_question_generation.regenerate_question_for_node).
    node_id : str
        Target sparse node's Neo4j elementId.
    node_label : str
        Target sparse node's human-readable label -- passed in rather
        than re-derived, since the caller (Stage 4) already has it from
        sparse_previews.
    cluster_memory : cluster_memory.ClusterMemory
        This node's cluster's ClusterMemory object (Representative +
        Exemplar memory). Read-only here.
    cluster_member_node_ids : list[str]
        elementIds of every __Entity__ node in this cluster -- needed
        only to call compute_cluster_embedding() for the semantic
        signal. See the module docstring for why this can't be derived
        from cluster_memory alone.
    driver : neo4j.Driver
        Existing driver, reused for the neighbor lookup and the
        embedding lookup inside compute_cluster_embedding.
    embedder : neo4j_graphrag.embeddings.SentenceTransformerEmbeddings
        The SAME embedder instance already loaded elsewhere in the
        pipeline. This file never constructs its own.
    llm_client : openai.OpenAI
        Existing Groq-backed client, used only if the score lands in
        the tiebreak band.
    attempt_number : int
        Which gate evaluation this is for this node (1 = original
        question, 2/3 = after regeneration). Purely for logging --
        this function does not enforce MAX_REGENERATION_ATTEMPTS
        itself; that retry loop lives in the caller (Stage 4), which is
        what decides whether to call this function again after a
        reject.
    goal_question : str, optional
        [G3] The user's question the whole run is enriching for. None ->
        the gate behaves exactly as before (4 signals, cluster-only
        tiebreak).
    goal_embedding : list[float], optional
        [G3] Embedding of goal_question, computed once by Stage 4 (embedded
        here if missing).

    Decision rule with a goal_question [G3]:
      goal < GOAL_FLOOR                               -> reject ("goal_floor")
      combined >= T_ACCEPT                            -> accept ("deterministic")
      combined <= T_REJECT                            -> reject ("deterministic")
      in between and goal >= GOAL_STRONG              -> accept ("goal_strong")
      in between otherwise                            -> LLM tiebreak, which
                                                         also sees the user's question

    Returns
    -------
    GateResult
    """
    question_embedding = embedder.embed_query(question)

    # [G3] Goal relevance: similarity of the candidate to the user's question.
    goal = None
    if goal_question:
        if goal_embedding is None:
            goal_embedding = embedder.embed_query(goal_question)
        goal = _cosine_similarity(question_embedding, goal_embedding)

    raw_signals = {
        "semantic": _semantic_similarity_signal(question_embedding, cluster_member_node_ids, driver),
        "entity": _entity_consistency_signal(question, node_id, node_label, driver),
        "goal": goal,                                                       # [G3]
        "accepted": _accepted_exemplar_signal(question_embedding, cluster_memory, embedder),
        "rejected": _rejected_exemplar_signal(question_embedding, cluster_memory, embedder),
    }

    combined_score = _combine_signals(raw_signals)

    if goal is not None and goal < GOAL_FLOOR:
        decision, decision_path = "reject", "goal_floor"                    # [G3]
    elif combined_score >= T_ACCEPT:
        decision, decision_path = "accept", "deterministic"
    elif combined_score <= T_REJECT:
        decision, decision_path = "reject", "deterministic"
    elif goal is not None and goal >= GOAL_STRONG:
        decision, decision_path = "accept", "goal_strong"                   # [G3]
    else:
        decision = _llm_tiebreak(llm_client, question, node_label, cluster_memory,
                                 goal_question=goal_question)               # [G3]
        decision_path = "llm_tiebreak"

    _log_gate_decision(node_id, node_label, question, raw_signals, combined_score,
                        decision, decision_path, attempt_number)

    return GateResult(
        decision=decision,
        confidence_score=combined_score,
        signals=raw_signals,
        decision_path=decision_path,
        attempt_number=attempt_number,
    )


# =============================================================================
# Post-Extraction Memory Gate
# =============================================================================
# SCOPE: separate gate, separate result type, separate constants from the
# Question Memory Gate above. Nothing above this line is touched or
# reused by name -- only compute_cluster_embedding() (from
# cluster_memory.py) and _cosine_similarity() (this file) are shared,
# exactly as they're already shared with question_memory_gate() above.
#
# WHY THIS GATE EXISTS
# ----------------------
# question_memory_gate() decides whether a QUESTION is worth spending
# retrieval/extraction resources on, BEFORE any Tavily call happens.
# This gate runs AFTER SimpleKGPipeline has already extracted candidate
# entities/relationships from the retrieved text, and decides whether
# that ACTUAL extracted content belongs to the target node's cluster
# and should really be written to the KG -- catching cases where a
# question passed the first gate but the retrieved snippet still turned
# out to be off-topic, or where extraction itself pulled out mostly
# noise (generic terms, fragments, etc.).
#
# PRE-INGESTION INTERCEPTION -- CONFIRMED, NOT ASSUMED
# -------------------------------------------------------
# Inspected against the installed neo4j_graphrag==1.19.0 source directly
# (experimental/pipeline/kg_builder.py, components/kg_writer.py):
# SimpleKGPipeline accepts an optional kg_writer component. Its internal
# pipeline always calls kg_writer.run(graph: Neo4jGraph,
# lexical_graph_config) with the fully-built extraction result --
# Neo4jNode/Neo4jRelationship objects, NOT YET WRITTEN to Neo4j -- right
# before Neo4jWriter.run() would otherwise immediately upsert them. This
# means a custom KGWriter subclass can genuinely intercept extraction
# results pre-ingestion; this is real interception, not an
# approximation or a post-hoc cleanup pass. See post_extraction_writer.py
# (new file) for the KGWriter subclass that does this and calls
# post_extraction_memory_gate() below with the extracted content.
#
# INTERFACE CHOICE -- plain lists, not raw Neo4jGraph objects
# --------------------------------------------------------------
# post_extraction_memory_gate() below takes plain lists of entity
# labels / relationship types, NOT a neo4j_graphrag Neo4jGraph object --
# mirrors question_memory_gate()'s existing pattern of taking primitive
# arguments rather than framework objects, keeps this file free of any
# neo4j_graphrag import, and makes the gate trivially testable with
# mocked/hand-written data (see test_post_extraction_gate.py) without
# constructing real pipeline objects. post_extraction_writer.py is
# responsible for translating a real Neo4jGraph into these primitives.
#
# NO LLM TIEBREAK IN THIS PHASE -- DELIBERATE SIMPLIFICATION
# ----------------------------------------------------------------
# Unlike question_memory_gate(), this gate has no LLM tiebreak path.
# The ambiguous middle band (T_REJECT < score < T_ACCEPT) is resolved
# deterministically toward REJECT rather than calling an LLM. This is a
# scope decision for this implementation phase (smallest possible
# change, no added API calls for an ambiguous case), not a discovered
# bug -- flagged explicitly as a known limitation/future improvement.
#
# accepted_triple IS POPULATED HERE, NOT FABRICATED
# ----------------------------------------------------
# AcceptedExemplar.accepted_triple (cluster_memory.py) has stayed
# Optional[str]=None throughout the Question Memory Gate's integration
# specifically because Stage 4 had no clean access to what was actually
# extracted. This gate is exactly the point where that access exists --
# the caller (Stage 4) can now build a real, non-fabricated
# accepted_triple string from extracted_entity_labels /
# extracted_relationship_types on an ACCEPT decision.
POST_EXTRACTION_W_SEMANTIC = 0.5
POST_EXTRACTION_W_ENTITY_OVERLAP = 0.5

POST_EXTRACTION_T_ACCEPT = 0.70
POST_EXTRACTION_T_REJECT = 0.35


@dataclass
class PostExtractionGateResult:
    """
    Everything post_extraction_memory_gate() reports back to its caller.

    signals holds the RAW, unweighted value of each signal -- "semantic"
    is None when cold-start (no computable cluster embedding yet),
    exactly mirroring GateResult's cold-start convention above.
    "entity_overlap" is never None (see _post_extraction_entity_overlap_signal's
    docstring for why it's always computable once there's at least one
    extracted entity).

    kept_entity_labels echoes extracted_entity_labels back on an ACCEPT
    decision (empty list on REJECT) -- purely for the caller's logging/
    exemplar-writing convenience, not used internally by this gate.
    """
    decision: str                 # "accept" or "reject"
    confidence_score: float       # combined weighted score, deterministic, always present
    signals: dict                 # {"semantic": float|None, "entity_overlap": float}
    decision_path: str            # always "deterministic" in this phase (no LLM tiebreak)
    kept_entity_labels: list
    # [PG2] Filled by post_extraction_filter(); empty for the legacy gate.
    kept_entity_ids: set = field(default_factory=set)
    kept_fact_indices: list = field(default_factory=list)
    fact_log: list = field(default_factory=list)


def _post_extraction_entity_overlap_signal(node_label: str, extracted_entity_labels: list,
                                            cluster_memory) -> float:
    """
    Fraction of extracted_entity_labels that plausibly belong to this
    target node's cluster: a case-insensitive substring match against
    EITHER the target node's own label OR any of the cluster's existing
    representative_entities (cluster_memory.representative.representative_entities
    -- the same list build_cluster_memory_store() already populates for
    every cluster; nothing new is stored to support this).

    Always computable (never None / cold-start) once
    extracted_entity_labels is non-empty: node_label alone is always a
    valid, non-empty comparison target, even if the cluster happens to
    have zero representative_entities recorded. The caller
    (post_extraction_memory_gate) never calls this with an empty
    extracted_entity_labels -- that case is rejected immediately,
    before any signal is computed at all.
    """
    reference_labels = set()
    if node_label:
        reference_labels.add(node_label.strip().lower())
    if cluster_memory is not None:
        for label in cluster_memory.representative.representative_entities:
            if label:
                reference_labels.add(str(label).strip().lower())

    if not reference_labels or not extracted_entity_labels:
        return 0.0

    matched = 0
    for label in extracted_entity_labels:
        label_lower = str(label).strip().lower()
        if any(ref in label_lower or label_lower in ref for ref in reference_labels):
            matched += 1

    return matched / len(extracted_entity_labels)


def _combine_post_extraction_signals(raw_signals: dict) -> float:
    """
    Same renormalized-weighted-average, cold-start-excluding approach as
    _combine_signals() above, applied to this gate's own two signals and
    its own weights (POST_EXTRACTION_W_*). Kept as a separate function
    (not a generalization of _combine_signals) so neither gate's
    combination logic can accidentally affect the other's.

    In practice "entity_overlap" is never None (see its docstring), so
    the all-cold-start midpoint-fallback branch below is unreachable
    with current callers -- kept anyway, for the same defensive reason
    _combine_signals() keeps it: never silently default an all-missing
    combination to a made-up neutral score.
    """
    weights = {"semantic": POST_EXTRACTION_W_SEMANTIC, "entity_overlap": POST_EXTRACTION_W_ENTITY_OVERLAP}
    available = {k: v for k, v in raw_signals.items() if v is not None}

    if not available:
        return (POST_EXTRACTION_T_ACCEPT + POST_EXTRACTION_T_REJECT) / 2

    total_weight = sum(weights[k] for k in available)
    return sum(available[k] * weights[k] for k in available) / total_weight


def _log_post_extraction_decision(node_label: str, extracted_entity_labels: list,
                                   raw_signals: dict, combined_score: float, decision: str) -> None:
    """Plain print-based logging, matching _log_gate_decision()'s existing
    convention above -- no logging framework introduced for this gate."""
    semantic_str = f"{raw_signals['semantic']:.4f}" if raw_signals["semantic"] is not None else "None (cold-start)"
    print(f"    [Post-Extraction Gate] node={node_label!r} "
          f"extracted_entities={extracted_entity_labels} "
          f"semantic={semantic_str} entity_overlap={raw_signals['entity_overlap']:.4f} "
          f"combined={combined_score:.4f} -> {decision.upper()}")


def post_extraction_memory_gate(node_label: str, extracted_entity_labels: list,
                                 extracted_relationship_types: list, cluster_memory,
                                 cluster_member_node_ids: list, driver, embedder) -> PostExtractionGateResult:
    """
    Evaluates ACTUAL extracted content (post-retrieval, post-extraction,
    pre-ingestion) against the target node's cluster context, and
    decides whether it should be written to the KG.

    Two signals (see their own docstrings for detail):
      - semantic: cosine similarity between an embedding of the
        extracted content and the cluster's on-demand embedding
        (compute_cluster_embedding, same mechanism the Question Memory
        Gate's semantic signal already uses). None (cold-start,
        excluded from the average) if the cluster has no computable
        embedding yet.
      - entity_overlap: fraction of extracted entity labels that match
        the target node's own label or the cluster's
        representative_entities. Always computable once there's at
        least one extracted entity.

    Immediate REJECT, no signals computed, if extracted_entity_labels is
    empty -- nothing was extracted at all, so there is nothing to gate
    on; this is not treated as an ambiguous/cold-start case.

    Deterministic accept/reject only (no LLM tiebreak in this phase --
    see this section's module-level comment above for why). The
    ambiguous middle band resolves to REJECT.

    Parameters
    ----------
    node_label : str
        The label of the sparse node this question/retrieval targeted
        (same node_label Stage 4 already threads through
        question_memory_gate() calls).
    extracted_entity_labels : list[str]
        Labels of the entity nodes SimpleKGPipeline's extractor produced
        for this retrieved snippet, EXCLUDING lexical-graph bookkeeping
        nodes (Chunk/Document) -- see post_extraction_writer.py for
        exactly how these are derived from the real Neo4jGraph.
    extracted_relationship_types : list[str]
        Relationship TYPES extracted alongside those entities (e.g.
        "OWNS", "LOCATED_IN") -- folded into the semantic signal's input
        text, not otherwise scored separately in this phase.
    cluster_memory : ClusterMemory
        Same cluster context question_memory_gate() already uses for
        this node. Never None when this function is called -- the
        caller (post_extraction_writer.py) checks for that upstream and
        accepts unconditionally (fail-open) if it is.
    cluster_member_node_ids : list[str]
        Same cluster member id list threaded through question_memory_gate().
    driver : neo4j.Driver
        Passed through to compute_cluster_embedding().
    embedder
        Same embedder instance used throughout this project. Only
        embed_query(text) is called on it, matching this file's
        existing embedder-reuse convention.

    Returns
    -------
    PostExtractionGateResult
    """
    if not extracted_entity_labels:
        raw_signals = {"semantic": None, "entity_overlap": 0.0}
        _log_post_extraction_decision(node_label, extracted_entity_labels, raw_signals, 0.0, "reject")
        return PostExtractionGateResult(
            decision="reject",
            confidence_score=0.0,
            signals=raw_signals,
            decision_path="deterministic",
            kept_entity_labels=[],
        )

    from cluster_memory import compute_cluster_embedding

    extracted_text = " ".join(list(extracted_entity_labels) + list(extracted_relationship_types))
    extracted_embedding = embedder.embed_query(extracted_text)

    cluster_embedding = (
        compute_cluster_embedding(driver, cluster_member_node_ids)
        if cluster_member_node_ids else None
    )
    semantic_score = (
        _cosine_similarity(extracted_embedding, cluster_embedding)
        if cluster_embedding is not None else None
    )

    entity_overlap_score = _post_extraction_entity_overlap_signal(
        node_label, extracted_entity_labels, cluster_memory,
    )

    raw_signals = {"semantic": semantic_score, "entity_overlap": entity_overlap_score}
    combined_score = _combine_post_extraction_signals(raw_signals)

    decision = "accept" if combined_score >= POST_EXTRACTION_T_ACCEPT else "reject"

    _log_post_extraction_decision(node_label, extracted_entity_labels, raw_signals, combined_score, decision)

    return PostExtractionGateResult(
        decision=decision,
        confidence_score=combined_score,
        signals=raw_signals,
        decision_path="deterministic",
        kept_entity_labels=list(extracted_entity_labels) if decision == "accept" else [],
    )


# =============================================================================
# [PG2] Post-Extraction Gate v2 -- judge each entity and fact separately
# =============================================================================
# WHY: the legacy gate above asks "what share of the extracted names do we
# ALREADY know?" and needs ~0.70 to pass. A page full of NEW, on-topic
# facts therefore scores low by design (live runs: ~0.31-0.35), and one
# bad fact sinks the whole page. v2 asks a different question: "is this
# entity / fact CONNECTED to the target or its cluster?" -- per entity and
# per fact, with no score threshold to tune.
#
# RULES (in order):
#   1. Anchors = the target node + the cluster's members (by name).
#   2. An extracted entity is an anchor if its name matches an anchor name
#      (same name after normalising, or whole-word surname match, see
#      _names_match).
#   3. Keep every entity reachable from an anchor within POST_V2_MAX_HOPS
#      links, walking only the facts extracted from THIS page.
#   4. Keep a fact only if BOTH its ends were kept.
#   5. Skip a kept fact that already exists in the graph (same subject,
#      relation, object -- either direction): status "duplicate".
#   6. For relations in FUNCTIONAL_RELATIONS (one value expected, e.g. a
#      date), log a conflict if the graph already has a DIFFERENT value.
#      Logged only -- the fact is still kept.
#   7. Reject the page only if no NEW fact survives.
# Optional per-fact relevance (cosine similarity of "subject relation
# object" to the question) is computed for the log only.

POST_V2_MAX_HOPS = 2
FUNCTIONAL_RELATIONS = {"OCCURRED_ON"}


def _norm_name(name) -> str:
    """Lower-case, collapse whitespace, strip surrounding punctuation."""
    return " ".join(str(name or "").lower().split()).strip(" .,;:'\"")


def _names_match(entity_name: str, anchor_name: str) -> bool:
    """[PG2] Does an extracted entity name refer to an anchor?
    True if the normalised names are equal, or the entity name mentions
    the anchor (whole word; multi-word anchors also match on their last
    word, e.g. "Portman" for "Natalie Portman"), or -- for entity names of
    4+ characters -- the anchor mentions the entity name ("Swan" vs
    "Black Swan")."""
    e, a = _norm_name(entity_name), _norm_name(anchor_name)
    if not e or not a:
        return False
    if e == a:
        return True
    if _mentions(anchor_name, entity_name):
        return True
    return len(e) >= 4 and _mentions(entity_name, anchor_name)


# [G4] Words that, together with digits and punctuation, make up a "literal"
# (a date, year, amount or count) rather than a named thing.
_MONTHS = ("january february march april may june july august september october november "
           "december jan feb mar apr jun jul aug sep sept oct nov dec")
_LITERAL_WORDS = set((_MONTHS + " million billion thousand hundred trillion mn bn k m "
                      "viewers viewer people percent per cent year years day days hour hours "
                      "minute minutes week weeks month months st nd rd th usd us dollars "
                      "dollar rupees euros pounds bc ad ce bce").split())


def is_literal_name(name) -> bool:
    """[G4] True if `name` is only a date / year / number / amount, e.g.
    "1871", "August 17, 2007", "3.3 million viewers", "$2.225 million",
    "Sept. 23, 2006", "17.24%". Such names say nothing about WHAT a page is
    about, so PG2 must not use them as anchors. A name with any other word
    ("Great Fire of 1871", "Apollo 11") is NOT literal."""
    text = _norm_name(name)
    if not text or not re.search(r"\d", text):
        return False
    words = re.findall(r"[a-z]+", text)
    return all(w in _LITERAL_WORDS for w in words)


def fact_key(subject_name, relation, object_name) -> tuple:
    """[PG2] Normalised identity of a fact, used for duplicate checks."""
    return (_norm_name(subject_name), str(relation).upper(), _norm_name(object_name))


def verbalize_fact(subject_name, relation, object_name) -> str:
    """'Natalie Portman' + 'CREATED_BY' + 'X' -> 'Natalie Portman created by X'."""
    return f"{subject_name} {str(relation).replace('_', ' ').lower()} {object_name}"


def post_extraction_filter(target_label: str, anchor_names: list, entities: list, facts: list,
                           existing_fact_keys=frozenset(), existing_functional=None,
                           relevance_fn=None, max_hops: int = POST_V2_MAX_HOPS,
                           exclude_literal_anchors: bool = True,
                           goal_relevance_fn=None) -> PostExtractionGateResult:
    """[PG2] Pure function -- no Neo4j, no LLM. post_extraction_writer.py
    gathers the inputs and applies the result.

    Parameters
    ----------
    target_label : str
        Name of the sparse node the question targeted.
    anchor_names : list[str]
        Target + cluster member names (duplicates/empties are fine).
    entities : list[dict]
        Extracted entities: {"id", "name", "type"}.
    facts : list[dict]
        Extracted facts between those entities: {"start", "type", "end"}
        (start/end are entity ids).
    existing_fact_keys : set[tuple]
        fact_key(...) of facts already in the graph (both directions is fine).
    existing_functional : dict[(subject_norm, RELATION)] -> set(object_norm)
        Existing values of FUNCTIONAL_RELATIONS, for conflict logging.
    relevance_fn : callable(str) -> float, optional
        Scores a verbalised fact against the enrichment question; logged only.
    exclude_literal_anchors : bool
        [G4] Dates / numbers (is_literal_name) are not used as anchors --
        neither as anchor names nor as extracted entities that match one.
    goal_relevance_fn : callable(str) -> float, optional
        [G4] Scores a verbalised fact against the USER'S question; logged
        only (key "goal_relevance").

    Returns PostExtractionGateResult with decision, kept_entity_ids,
    kept_fact_indices (NEW facts to write), fact_log, and count signals.
    """
    existing_functional = existing_functional or {}
    names = {e["id"]: e["name"] for e in entities}
    anchors = [a for a in dict.fromkeys([target_label] + list(anchor_names or [])) if a]
    ignored_literal_anchors = []
    if exclude_literal_anchors:                                              # [G4]
        ignored_literal_anchors = [a for a in anchors if is_literal_name(a)]
        anchors = [a for a in anchors if not is_literal_name(a)]

    anchor_ids = {e["id"] for e in entities
                  if not (exclude_literal_anchors and is_literal_name(e["name"]))   # [G4]
                  and any(_names_match(e["name"], a) for a in anchors)}

    # Walk the extracted facts (undirected) from the anchors, up to max_hops.
    neighbours = {e["id"]: set() for e in entities}
    for f in facts:
        if f["start"] in neighbours and f["end"] in neighbours:
            neighbours[f["start"]].add(f["end"])
            neighbours[f["end"]].add(f["start"])
    kept_ids, frontier = set(anchor_ids), set(anchor_ids)
    for _ in range(max_hops):
        frontier = {n for node in frontier for n in neighbours.get(node, ()) if n not in kept_ids}
        kept_ids |= frontier

    kept_fact_indices, fact_log, conflicts = [], [], 0
    counts = {"kept": 0, "duplicate": 0, "dropped_unconnected": 0}
    for i, f in enumerate(facts):
        s_name, o_name = names.get(f["start"], f["start"]), names.get(f["end"], f["end"])
        entry = {"subject": s_name, "relation": f["type"], "object": o_name}
        if relevance_fn is not None:
            try:
                entry["relevance"] = round(float(relevance_fn(verbalize_fact(s_name, f["type"], o_name))), 4)
            except Exception:
                entry["relevance"] = None
        if goal_relevance_fn is not None:                                    # [G4] log only
            try:
                entry["goal_relevance"] = round(float(goal_relevance_fn(verbalize_fact(s_name, f["type"], o_name))), 4)
            except Exception:
                entry["goal_relevance"] = None
        if not (f["start"] in kept_ids and f["end"] in kept_ids):
            entry["status"] = "dropped_unconnected"
        elif fact_key(s_name, f["type"], o_name) in existing_fact_keys or \
                fact_key(o_name, f["type"], s_name) in existing_fact_keys:
            entry["status"] = "duplicate"
        else:
            entry["status"] = "kept"
            kept_fact_indices.append(i)
            if str(f["type"]).upper() in FUNCTIONAL_RELATIONS:
                known = existing_functional.get((_norm_name(s_name), str(f["type"]).upper()), set())
                if known and _norm_name(o_name) not in known:
                    entry["conflict_with"] = sorted(known)
                    conflicts += 1
        counts[entry["status"]] += 1
        fact_log.append(entry)

    # New entities = kept, not anchors (anchors already exist in the graph).
    new_entity_count = len(kept_ids - anchor_ids)
    if kept_fact_indices:
        decision, reason = "accept", "new_connected_facts"
    elif not anchor_ids:
        decision, reason = "reject", "no_anchor_in_extraction"
    elif counts["duplicate"] and not counts["dropped_unconnected"]:
        decision, reason = "reject", "nothing_new_only_duplicates"
    else:
        decision, reason = "reject", "nothing_connected"

    signals = {
        "reason": reason,
        "anchors_found": len(anchor_ids),
        "entities_total": len(entities),
        "entities_kept": len(kept_ids),
        "new_entities_kept": new_entity_count,
        "facts_total": len(facts),
        "facts_kept": counts["kept"],
        "facts_duplicate": counts["duplicate"],
        "facts_dropped_unconnected": counts["dropped_unconnected"],
        "fact_conflicts": conflicts,
        "ignored_literal_anchors": ignored_literal_anchors,                  # [G4]
    }
    score = counts["kept"] / len(facts) if facts else 0.0
    kept_labels = [names[i] for i in sorted(kept_ids)] if decision == "accept" else []

    print(f"    [Post-Extraction Gate v2] node={target_label!r} {decision.upper()} ({reason}): "
          f"facts kept={counts['kept']} duplicate={counts['duplicate']} "
          f"dropped={counts['dropped_unconnected']} conflicts={conflicts}; "
          f"entities kept={len(kept_ids)}/{len(entities)} (anchors {len(anchor_ids)})")

    return PostExtractionGateResult(
        decision=decision,
        confidence_score=round(score, 4),
        signals=signals,
        decision_path="v2_connected_facts",
        kept_entity_labels=kept_labels,
        kept_entity_ids=kept_ids if decision == "accept" else set(),
        kept_fact_indices=kept_fact_indices if decision == "accept" else [],
        fact_log=fact_log,
    )

