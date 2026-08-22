"""
MKGE Cluster-Memory Extension -- Question Memory Gate
============================================================
Implements Phase 3a of the memory-augmented MKGE design: the Question
Memory Gate. Evaluates a candidate enrichment question against the
target sparse node's Cluster Memory BEFORE that question is allowed to
trigger a Tavily search + SimpleKGPipeline ingestion in Stage 4 -- the
goal is to catch semantic drift before it happens, not after.

SCOPE OF THIS FILE
-------------------
This file ONLY implements the Question Memory Gate. It does not
implement:
  - the Post-Extraction Gate (separate, later phase)
  - rejected-exemplar pruning (still a stub in cluster_memory.py)
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

from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Constants (placeholder values -- explicitly flagged as needing tuning
# against real gate-call data once Stage 4 integration produces some;
# see project manifesto, Source of Truth section)
# =============================================================================
W_SEMANTIC = 0.30
W_ENTITY = 0.30
W_ACCEPTED = 0.20
W_REJECTED = 0.20

T_ACCEPT = 0.70
T_REJECT = 0.35

# Entity-consistency signal's INTERNAL target/neighbour blend (Option 2a,
# chosen after comparing Option 1 -- tiered match -- against Option 2 at
# two weightings, via compare_entity_signal.py; see that script and
# Question_Memory_Gate.pdf sections 6-8 for the full comparison and
# rationale). Separate from W_ENTITY above: W_ENTITY weights this
# signal's OUTPUT against the other three signals in _combine_signals;
# these two weight what goes INTO that output in the first place.
ENTITY_W_TARGET = 0.5
ENTITY_W_NEIGHBOUR = 0.5

MAX_REGENERATION_ATTEMPTS = 2  # 3 total gate evaluations per node (1 original + 2 retries)


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
    signals: dict                 # {"semantic": float|None, "entity": float|None, "accepted": float|None, "rejected": float|None}
    decision_path: str            # "deterministic" or "llm_tiebreak"
    attempt_number: int           # 1, 2, or 3 -- which gate call this was for this node


# =============================================================================
# Question -> sparse node assignment (lightweight, non-LLM, post-hoc)
# =============================================================================
def assign_question_to_sparse_node(question: str, sparse_previews: dict) -> tuple:
    """
    Matches ONE generated question back to the ONE sparse node it was
    primarily targeting. This is a separate, lightweight, non-LLM step
    -- Stage 3's original batch question generation (one LLM call per
    metric, QUESTIONS_PER_METRIC questions back) is completely
    unchanged; this just figures out afterward which node each of those
    questions was about.

    Strategy (in order):
      1. Label substring match -- if any sparse node's label appears
         as a case-insensitive substring of the question text, that
         node wins immediately (this is the strong, common case: a
         generated question naturally names the entity it's asking
         about, e.g. "What sector does Lakshmi Mittal's company operate
         in?" contains the label "Lakshmi Mittal").
      2. Word-overlap fallback -- if no label match is found, score
         each candidate node by how many words its label + 160-char
         preview text share with the question (case-insensitive,
         simple set intersection), and return the highest-scoring node.
         If every node scores 0, falls back to the first candidate in
         sparse_previews (arbitrary but deterministic -- dict order is
         insertion order in Python 3.7+), since the caller has already
         decided this question needs *some* assignment.

    Parameters
    ----------
    question : str
        One generated enrichment question.
    sparse_previews : dict[str, tuple[str, str]]
        Same shape already used throughout stage3/stage4:
        {node_id: (label, preview)}.

    Returns
    -------
    tuple[str, str]
        (node_id, method) -- method is "label_match" or "word_overlap",
        included so gate logging can show which strategy found the
        assignment (useful for spot-checking whether word-overlap is
        firing more often than expected, which would suggest Stage 3's
        questions aren't naming their target entities clearly).
    """
    question_lower = question.lower()

    # Strategy 1: label substring match.
    for node_id, (label, _preview) in sparse_previews.items():
        if label and label.lower() in question_lower:
            return node_id, "label_match"

    # Strategy 2: word-overlap fallback.
    question_words = set(question_lower.replace("?", "").split())
    best_node_id = None
    best_score = -1
    for node_id, (label, preview) in sparse_previews.items():
        candidate_text = f"{label} {preview}".lower()
        candidate_words = set(candidate_text.split())
        overlap = len(question_words & candidate_words)
        if overlap > best_score:
            best_score = overlap
            best_node_id = node_id

    if best_node_id is not None:
        return best_node_id, "word_overlap"

    # Degenerate case: sparse_previews was empty. Caller should never
    # reach this (it wouldn't have a question to assign in the first
    # place), but fail loudly rather than returning a bad tuple.
    raise ValueError("assign_question_to_sparse_node: sparse_previews is empty, nothing to assign to.")


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
    with driver.session() as session:
        result = session.run(
            "MATCH (n:__Entity__)-[r]-(m:__Entity__) "
            "WHERE elementId(n) = $node_id "
            "RETURN DISTINCT coalesce(m.name, m.id, 'unnamed') AS label",
            node_id=node_id,
        )
        return [record["label"] for record in result]


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

    Matching is case-insensitive substring, same convention as before.

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

    question_lower = question.lower()

    target_match = 1.0 if node_label and node_label.lower() in question_lower else 0.0

    if neighbours:
        matched_neighbours = [n for n in neighbours if n and n.lower() in question_lower]
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
        "accepted": W_ACCEPTED,
        "rejected": W_REJECTED,
    }

    support = {}
    if raw_signals.get("semantic") is not None:
        support["semantic"] = raw_signals["semantic"]
    if raw_signals.get("entity") is not None:
        support["entity"] = raw_signals["entity"]
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
def _llm_tiebreak(llm_client, question: str, node_label: str, cluster_memory) -> str:
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
        "accept" or "reject". Defaults to "reject" if the LLM's response
        can't be parsed as either -- the safer failure direction, since
        an unparseable tiebreak response is itself a signal something
        is off, and this design never terminates the pipeline on a
        single node's rejection (Stage 4 just skips that node and
        continues).
    """
    rep = cluster_memory.representative
    entities = ", ".join(rep.representative_entities) if rep.representative_entities else "(none)"

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
    text = response.choices[0].message.content.strip().lower()
    first_line = text.splitlines()[0].strip() if text else ""

    if "accept" in first_line:
        return "accept"
    if "reject" in first_line:
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
    print(f"    signals : semantic={raw_signals.get('semantic')}  "
          f"entity={raw_signals.get('entity')}  "
          f"accepted={raw_signals.get('accepted')}  "
          f"rejected={raw_signals.get('rejected')}")
    print(f"    combined_score={combined_score:.4f}  decision={decision}  path={decision_path}")


# =============================================================================
# Top-level gate
# =============================================================================
def question_memory_gate(question: str, node_id: str, node_label: str, cluster_memory,
                          cluster_member_node_ids: list, driver, embedder, llm_client,
                          attempt_number: int = 1) -> GateResult:
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

    Returns
    -------
    GateResult
    """
    question_embedding = embedder.embed_query(question)

    raw_signals = {
        "semantic": _semantic_similarity_signal(question_embedding, cluster_member_node_ids, driver),
        "entity": _entity_consistency_signal(question, node_id, node_label, driver),
        "accepted": _accepted_exemplar_signal(question_embedding, cluster_memory, embedder),
        "rejected": _rejected_exemplar_signal(question_embedding, cluster_memory, embedder),
    }

    combined_score = _combine_signals(raw_signals)

    if combined_score >= T_ACCEPT:
        decision, decision_path = "accept", "deterministic"
    elif combined_score <= T_REJECT:
        decision, decision_path = "reject", "deterministic"
    else:
        decision = _llm_tiebreak(llm_client, question, node_label, cluster_memory)
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