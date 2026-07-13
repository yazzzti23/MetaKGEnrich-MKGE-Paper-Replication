# MetaKGEnrich (MKGE) — Paper Replication

This repo is my attempt to replicate the pipeline from the paper
**"Enhancing Metacognitive AI: Knowledge-Graph Population with Graph-Theoretic LLM Enrichment"**
(Askin, Hadar & Conway-Smith), as part of my research internship.

Status: **work in progress.** The core 5-stage pipeline runs end-to-end on a test
question. What's left is running it properly across a full test set and building
out the research extensions described below.

---

## What is this paper actually about?

LLMs are bad at knowing what they *don't* know. This paper builds a system that:

1. Builds a small knowledge graph (KG) from a single web search result about a question.
2. Uses graph theory to mathematically find the "thin" or "weak" spots in that graph —
   nodes that are isolated, poorly connected, or sitting in tiny disconnected clusters.
3. Asks an LLM to generate targeted questions specifically about those weak spots.
4. Searches the web for answers to those questions and adds them back into the graph.
5. Re-answers the original question using the now-enriched graph, and checks
   (with a second LLM acting as judge) whether the answer actually got better.

The clever part is step 2 — instead of randomly enriching the graph, it uses 7 graph
metrics (clique size, isolation, clustering, degree, betweenness, diameter, and
community size) to decide *where* the graph is weak, the same way a person might
notice "I don't actually know how these two facts connect."

## The 5 stages, in this repo

| File | What it does |
|---|---|
| `stage1_seed_kg.py` | Takes one question, gets the top web search result (Tavily), and feeds it through Neo4j's GraphRAG builder to create a tiny seed knowledge graph. |
| `stage2_sparsity_metrics.py` | Loads that graph and computes the 7 sparsity metrics per node using NetworkX. Any node scoring at/below the median on a metric is flagged "sparse" on that metric. |
| `stage3_question_generation.py` | For each of the 7 metrics, sends the sparse nodes (with a short text preview + a "cognitive label" like *"bridge missing"*) to the LLM and asks for exactly 5 fact-seeking questions per metric. |
| `stage4_retrieval_ingestion.py` | For each generated question, searches the web, takes the top result, and ingests it into the graph — **recomputing sparsity fresh before each metric's turn**, so later metrics see the graph as it's been enriched so far. |
| `stage5_evaluation.py` | Answers the original question using GraphRAG (vector search over the graph) before and after enrichment, then has an LLM judge decide if the answer improved. |
| `reset_graph.py` | Utility — wipes the Neo4j graph clean between test questions so results don't bleed into each other. |

Run order for one test question:
```
python reset_graph.py
python stage1_seed_kg.py
python stage5_evaluation.py before
python stage4_retrieval_ingestion.py
python stage5_evaluation.py after
```

## What my part in this actually was

I didn't hand-write this code line by line from a blank file — I worked through it
with AI (Claude) as a coding assistant. My actual contributions were:

- **Studying the paper closely** — the metrics, the cognitive labels, Algorithm 1's
  pseudocode — before any code was written, so I understood *why* each stage exists,
  not just what it does.
- **Directing the implementation** to run on free/accessible tools instead of the
  paper's original GPT-4o + FAISS setup: Groq's free API standing in for GPT-4o,
  `all-MiniLM-L6-v2` for embeddings, Neo4j Aura, Tavily for search.
- **Catching a real bug**: the first version of Stage 4 generated all 35 enrichment
  questions upfront from the unchanged seed graph. Re-reading Algorithm 1 closely,
  I noticed the paper actually recomputes sparsity detection *inside* the per-metric
  loop — meaning each metric should see the graph as enriched by every metric before
  it. I traced this back to the paper's own reported ~5 average Tavily calls per
  question (which only makes sense if later metrics increasingly run out of sparse
  nodes to ask about) and had the loop restructured to match.
- **Identifying an original research gap**: the paper's methodology doesn't account
  for what happens when retrieved web content is mostly boilerplate/noise rather
  than useful facts — I noticed this could cascade and degrade the graph rather than
  improve it.
- **Proposing three research extensions** to address that gap (see below).
- Writing up professor-facing progress notes and confirming a real implementation
  gotcha: `SimpleKGPipeline` embeds chunks automatically but does **not** create the
  vector index needed for retrieval — that has to be done separately.

## What's still to do / research extensions

1. **Relevance filtering at retrieval time** — filter out boilerplate/irrelevant
   snippets before they get ingested into the graph, to avoid the noise cascade issue.
2. **Effectiveness-weighted question budgeting** — the paper's Table 2 shows some
   metrics (e.g. Non-Clique, Betweenness) yield much higher improvement rates than
   others (e.g. Clique). Instead of a flat 5 questions per metric, weight the budget
   toward the metrics that historically pay off more.
3. **Multi-LLM comparison** — test whether the choice of LLM (for question
   generation and/or judging) changes the improvement rate.

## Setup

```
pip install -r requirements.txt
```

Create a `.env` file in the same folder with:
```
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
TAVILY_API_KEY=...
GROQ_API_KEY=...
```

## Reference

Askin, D., Hadar, G., & Conway-Smith, B. *Enhancing Metacognitive AI: Knowledge-Graph
Population with Graph-Theoretic LLM Enrichment.*