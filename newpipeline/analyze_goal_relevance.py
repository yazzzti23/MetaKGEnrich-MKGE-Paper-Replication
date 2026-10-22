"""
MKGE experiment helper -- how relevant were the enrichment questions to the
USER'S question? (goal relevance)
==========================================================================
READ-ONLY. No Groq, no Neo4j, no Tavily. Only the local MiniLM embedder.

For every Stage 4 question in the saved runs, this computes

    goal_sim = cosine( embed(enrichment question), embed(user's question) )

-- the same embedder and cosine the pipeline already uses. It is used for
two things:
  1. Evidence for the report: how far each pipeline's questions drifted
     from what the user asked (original vs ours).
  2. Calibration for fix G3 (the Question Gate's new "goal" signal): where
     clearly-good and clearly-off-topic questions land, so the weight and
     floor are set from data, not guessed.

It also re-scores OUR logged Question-Gate calls with the proposed 5-signal
formula (semantic, entity, goal, accepted, rejected) to show which decisions
would change. The LLM tiebreak cannot be replayed offline, so those rows are
shown as "tiebreak".

Usage (from the repo root, inside the .venv):
    python newpipeline\\analyze_goal_relevance.py | Tee-Object runs\\goal_relevance_analysis.txt
"""

import json
import os
import re
import sys

RUNS_DIR = "runs"
QUESTIONS_FILE = "test_questions.json"

# Which saved runs to analyse, per question.
RUNS = {
    "q1": {
        "original_txt": "q1_original_stage4.txt",
        "ours_logs": {
            "ours_PG1 (Run 1)": "2026-10-20_133554_ours.json",
            "ours_PG2 (Run 2)": "2026-10-20_140247_ours.json",
        },
    },
    "q2": {
        "original_txt": "q2_original_stage4.txt",
        "ours_logs": {
            "ours_PG2": "2026-10-21_135354_ours.json",
        },
    },
}

# Proposed 5-signal weights for G3 (sum = 1.0) and the existing thresholds.
PROPOSED_WEIGHTS = {"semantic": 0.25, "entity": 0.25, "goal": 0.20, "accepted": 0.15, "rejected": 0.15}
T_ACCEPT, T_REJECT = 0.70, 0.35

FLOORS = (0.20, 0.30, 0.40)   # count questions below each goal_sim value


# ----------------------------------------------------------------------------
# Reading the saved outputs
# ----------------------------------------------------------------------------
def _fix_mojibake(text: str) -> str:
    """PowerShell's Tee-Object stored Python's UTF-8 output as cp437 text
    (e.g. a narrow no-break space shows up as 'ΓÇ»'). Undo that if possible."""
    try:
        return text.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _read_text(path: str) -> str:
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def original_questions(txt_name: str) -> list:
    """Questions the original Stage 4 actually searched for (from its terminal output)."""
    path = os.path.join(RUNS_DIR, txt_name)
    if not os.path.exists(path):
        print(f"[warn] {path} not found -- skipping original run.")
        return []
    out = []
    for line in _read_text(path).splitlines():
        m = re.match(r"\s*\[Stage 4\] Question: (.*)", line)
        if m:
            out.append(" ".join(_fix_mojibake(m.group(1)).split()))
    return out


def ours_gate_calls(log_name: str) -> list:
    """Every Question-Gate evaluation in one of our run logs (attempt 1 and 2)."""
    path = os.path.join(RUNS_DIR, log_name)
    if not os.path.exists(path):
        print(f"[warn] {path} not found -- skipping.")
        return []
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    calls = []
    for rec in log.get("questions", []):
        for attempt, q_key, g_key in ((1, "question", "question_gate_1"),
                                      (2, "regenerated_question", "question_gate_2")):
            gate = rec.get(g_key)
            if rec.get(q_key) and gate:
                calls.append({
                    "attempt": attempt,
                    "question": " ".join(rec[q_key].split()),
                    "target": rec.get("target_node", ""),
                    "decision": gate.get("decision"),
                    "path": gate.get("path"),
                    "score": gate.get("score"),
                    "signals": gate.get("signals", {}),
                    "outcome": rec.get("outcome"),
                    "post_gate": (rec.get("post_gate") or {}).get("decision"),
                })
    return calls


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0 or nb == 0 else max(0.0, min(1.0, dot / (na * nb)))


def proposed_score(signals: dict, goal: float) -> float:
    support = {}
    for key in ("semantic", "entity", "accepted"):
        if signals.get(key) is not None:
            support[key] = signals[key]
    if signals.get("rejected") is not None:
        support["rejected"] = 1.0 - signals["rejected"]
    support["goal"] = goal
    wsum = sum(PROPOSED_WEIGHTS[k] for k in support)
    return sum(support[k] * PROPOSED_WEIGHTS[k] for k in support) / wsum


def band(score: float) -> str:
    if score >= T_ACCEPT:
        return "accept"
    if score <= T_REJECT:
        return "reject"
    return "tiebreak"


def summary(label: str, sims: list) -> str:
    if not sims:
        return f"  {label:<20} (no questions)"
    below = "  ".join(f"<{f:.2f}: {sum(s < f for s in sims):>2}" for f in FLOORS)
    return (f"  {label:<20} n={len(sims):>2}  mean={sum(sims) / len(sims):.2f}  "
            f"min={min(sims):.2f}  max={max(sims):.2f}  |  {below}")


def main(embed=None):
    if embed is None:
        from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
        embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
        embed = embedder.embed_query

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        user_questions = {q["id"]: q["question"] for q in json.load(f)["questions"]}

    for qid, cfg in RUNS.items():
        goal_text = user_questions[qid]
        goal_vec = embed(goal_text)
        print("=" * 100)
        print(f"{qid}: {goal_text}")
        print("=" * 100)

        summaries = []

        # ---- Original
        sims = []
        print("\n-- ORIGINAL MKGE: questions searched  (goal_sim | question)")
        for q in original_questions(cfg["original_txt"]):
            s = cosine(embed(q), goal_vec)
            sims.append(s)
            print(f"  {s:.2f} | {q[:110]}")
        summaries.append(summary("original", sims))

        # ---- Ours
        for run_label, log_name in cfg["ours_logs"].items():
            calls = ours_gate_calls(log_name)
            sims_all, sims_searched = [], []
            changed = 0
            print(f"\n-- {run_label}: Question-Gate calls")
            print("  goal | logged: decision/path/score | proposed 5-signal score -> band | target | question")
            for c in calls:
                s = cosine(embed(c["question"]), goal_vec)
                sims_all.append(s)
                searched = c["decision"] == "accept"
                if searched:
                    sims_searched.append(s)
                new = proposed_score(c["signals"], s)
                new_band = band(new)
                old_band = band(c["score"]) if c["score"] is not None else "?"
                mark = "  <-- band changes" if new_band != old_band else ""
                changed += new_band != old_band
                print(f"  {s:.2f} | {c['decision']:<6} {str(c['path'])[:5]:<5} {c['score']:.2f} | "
                      f"{new:.2f} -> {new_band:<8} | {c['target'][:20]:<20} | {c['question'][:70]}{mark}")
            print(f"  ({changed} of {len(calls)} calls would land in a different band)")
            summaries.append(summary(f"{run_label} all", sims_all))
            summaries.append(summary(f"{run_label} searched", sims_searched))

        print(f"\n-- SUMMARY {qid}  (goal_sim; 'searched' = passed the Question Gate)")
        for line in summaries:
            print(line)
        print()


if __name__ == "__main__":
    main()