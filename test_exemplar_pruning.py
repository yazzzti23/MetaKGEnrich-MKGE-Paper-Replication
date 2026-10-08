"""
Focused test for rejected-exemplar pruning -- NOT part of the pipeline,
does NOT touch live Neo4j, does NOT call any API. Pure in-memory
dataclass behavior.

Run with:  python test_exemplar_pruning.py
"""

from cluster_memory import ExemplarMemory, AcceptedExemplar, RejectedExemplar


def _make_rejected(i: int) -> RejectedExemplar:
    """Builds a distinguishable RejectedExemplar -- sparse_node encodes
    its creation order (0, 1, 2, ...) so tests can assert exactly which
    ones survived pruning by identity, not just by count."""
    return RejectedExemplar(
        sparse_node=f"Node{i}",
        generated_question=f"Question {i}?",
        failure_type="question_gate_reject_deterministic",
        rejection_reason=f"rejection #{i}",
        confidence_score=0.1,
    )


def _make_accepted(i: int) -> AcceptedExemplar:
    return AcceptedExemplar(
        sparse_node=f"AcceptedNode{i}",
        generated_question=f"Accepted question {i}?",
        retrieved_evidence=f"evidence {i}",
        acceptance_reason="question_gate_accept_deterministic",
    )


# =============================================================================
# 1. Below-limit behavior
# =============================================================================
def test_below_limit_no_pruning():
    memory = ExemplarMemory()
    for i in range(3):
        memory.add_rejected(_make_rejected(i), max_rejected=5)

    assert len(memory.rejected) == 3, f"expected 3, got {len(memory.rejected)}"
    assert [r.sparse_node for r in memory.rejected] == ["Node0", "Node1", "Node2"]
    print("test_below_limit_no_pruning: PASS ->", [r.sparse_node for r in memory.rejected])


# =============================================================================
# 2. Exactly-at-limit behavior
# =============================================================================
def test_exactly_at_limit_no_pruning():
    memory = ExemplarMemory()
    for i in range(5):
        memory.add_rejected(_make_rejected(i), max_rejected=5)

    assert len(memory.rejected) == 5, f"expected 5, got {len(memory.rejected)}"
    assert [r.sparse_node for r in memory.rejected] == [f"Node{i}" for i in range(5)]
    print("test_exactly_at_limit_no_pruning: PASS ->", [r.sparse_node for r in memory.rejected])


# =============================================================================
# 3. Pruning when the limit is exceeded -- deterministic FIFO (oldest first)
# =============================================================================
def test_pruning_evicts_oldest_first():
    memory = ExemplarMemory()
    for i in range(8):
        memory.add_rejected(_make_rejected(i), max_rejected=5)

    assert len(memory.rejected) == 5, f"expected 5 after pruning, got {len(memory.rejected)}"
    # The oldest 3 (Node0, Node1, Node2) must be gone; the 5 most
    # recently added (Node3..Node7) must remain, in original order.
    surviving = [r.sparse_node for r in memory.rejected]
    assert surviving == ["Node3", "Node4", "Node5", "Node6", "Node7"], surviving
    print("test_pruning_evicts_oldest_first: PASS ->", surviving)


def test_pruning_stays_bounded_over_many_additions():
    """Adds far more than the limit, one at a time, and checks the list
    never exceeds max_rejected at any point and always holds the most
    recent max_rejected entries."""
    memory = ExemplarMemory()
    for i in range(37):
        memory.add_rejected(_make_rejected(i), max_rejected=10)
        assert len(memory.rejected) <= 10, f"exceeded cap at i={i}: len={len(memory.rejected)}"

    surviving = [r.sparse_node for r in memory.rejected]
    expected = [f"Node{i}" for i in range(27, 37)]  # last 10 of 0..36
    assert surviving == expected, surviving
    print("test_pruning_stays_bounded_over_many_additions: PASS ->", surviving)


# =============================================================================
# 4. Accepted exemplars remain untouched by rejected-exemplar pruning
# =============================================================================
def test_accepted_exemplars_unaffected_by_rejected_pruning():
    memory = ExemplarMemory()
    for i in range(4):
        memory.add_accepted(_make_accepted(i))
    for i in range(10):
        memory.add_rejected(_make_rejected(i), max_rejected=3)

    assert len(memory.accepted) == 4, f"accepted list was mutated: len={len(memory.accepted)}"
    assert [a.sparse_node for a in memory.accepted] == [f"AcceptedNode{i}" for i in range(4)]
    assert len(memory.rejected) == 3, f"expected rejected pruned to 3, got {len(memory.rejected)}"
    print("test_accepted_exemplars_unaffected_by_rejected_pruning: PASS -> "
          f"accepted={len(memory.accepted)}, rejected={len(memory.rejected)}")


# =============================================================================
# 5. Default cap (module constant) applies with no call-site changes
# =============================================================================
def test_default_cap_applies_without_explicit_argument():
    from cluster_memory import MAX_REJECTED_EXEMPLARS_PER_CLUSTER

    memory = ExemplarMemory()
    # Add one more than the real default cap, WITHOUT passing max_rejected
    # explicitly -- exercises the exact call shape Stage 4 uses
    # (`exemplars.add_rejected(RejectedExemplar(...))`, no second arg).
    for i in range(MAX_REJECTED_EXEMPLARS_PER_CLUSTER + 1):
        memory.add_rejected(_make_rejected(i))

    assert len(memory.rejected) == MAX_REJECTED_EXEMPLARS_PER_CLUSTER, (
        f"expected default cap {MAX_REJECTED_EXEMPLARS_PER_CLUSTER}, got {len(memory.rejected)}"
    )
    assert memory.rejected[0].sparse_node == "Node1", "oldest (Node0) should have been evicted"
    print(f"test_default_cap_applies_without_explicit_argument: PASS -> "
          f"len={len(memory.rejected)}, oldest survivor={memory.rejected[0].sparse_node}")


if __name__ == "__main__":
    test_below_limit_no_pruning()
    test_exactly_at_limit_no_pruning()
    test_pruning_evicts_oldest_first()
    test_pruning_stays_bounded_over_many_additions()
    test_accepted_exemplars_unaffected_by_rejected_pruning()
    test_default_cap_applies_without_explicit_argument()
    print("\nALL EXEMPLAR PRUNING TESTS PASSED")