# Research Vision

NeuroPager is designed to eventually accompany an academic paper on
memory management for lifelong LLM agents. This document lays out the
research problem, hypotheses, and open questions the project intends to
explore.

## Motivation

LLM agents deployed over long horizons (days, months, indefinitely)
accumulate context far exceeding any practical context window. Current
practice — flat retrieval-augmented generation over a single vector index —
has no notion of:

- **Working set locality:** which memories are "hot" right now versus
  merely stored.
- **Principled eviction:** what to forget, and when, under a fixed budget.
- **Structured recall:** explicit relational reasoning over accumulated
  facts, as opposed to purely similarity-based recall.
- **Provenance:** why a given memory was surfaced for a given query.

Operating systems solved an analogous problem — programs whose working
data exceeds physical RAM — decades ago, via virtual memory: page tables,
demand paging, and replacement policies (LRU, LFU, Clock, ARC, etc.). We
hypothesize that this framing transfers productively to LLM agent memory.

## Research Questions

1. **Does an OS-inspired paging abstraction improve long-horizon agent
   task performance** compared to flat RAG, at equal or lower retrieval
   cost?
2. **Which page replacement policies best fit agent memory access
   patterns?** Agent memory access is not uniform-random like typical OS
   workloads — does it exhibit exploitable structure (bursty topic
   locality, recurring sub-tasks) that classical policies (LRU/LFU) fail
   to capture, motivating learned or hybrid policies?
3. **How should symbolic (knowledge graph) and dense (vector) retrieval be
   fused**, and does the fusion strategy matter more than either component
   individually?
4. **What is the right notion of a "page" for unstructured agent memory?**
   Unlike OS pages (fixed-size byte ranges), memory "pages" here are
   semantically meaningful units (an episode, a fact, a summary) — how
   does variable granularity affect paging efficiency?
5. **How should memory be compacted/summarized over time** to bound the
   growth of cold storage without losing retrievable fidelity (analogous to
   OS memory compaction / garbage collection)?
6. **What are appropriate benchmarks** for evaluating lifelong agent
   memory systems? (See `benchmarks/`.)

## Related Work (To Be Expanded)

This section will track related work across:

- Retrieval-augmented generation (RAG) and long-context LLMs
- Memory-augmented neural networks
- Neuro-symbolic AI and knowledge graph reasoning
- Classical virtual memory and cache replacement research (LRU, LFU, ARC,
  Belady's algorithm as an optimal offline baseline)
- Cognitive architectures (episodic vs. semantic vs. procedural memory)

*Contributions of citations and summaries are welcome — see
[CONTRIBUTING.md](../CONTRIBUTING.md).*

## Evaluation Philosophy

NeuroPager aims to support rigorous, reproducible evaluation:

- **Offline-optimal baselines:** comparing replacement policies against
  Belady's algorithm (optimal with perfect future knowledge) to quantify
  the cost of online decision-making.
- **Ablations:** vector-only vs. graph-only vs. hybrid retrieval; policy
  A/B comparisons under identical working-set traces.
- **Reproducibility:** experiments tracked under `experiments/`, with
  configs under `configs/`, so results are re-runnable from a committed
  configuration rather than ad hoc scripts.

## Citation

If this project informs your research, please cite it — see
[CITATION.cff](../CITATION.cff).
