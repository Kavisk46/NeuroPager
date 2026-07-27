# Roadmap

This roadmap tracks NeuroPager's progression from structural scaffold to a
research-validated, usable library. Scope and ordering will evolve as
design discussions in [research.md](research.md) mature.

## Phase 0 — Repository Scaffold (Current)

- [x] Professional repository structure
- [x] Modular Python package layout (interfaces only, no logic)
- [x] Documentation set (architecture, memory, design, api, research,
      roadmap)
- [x] Config management scaffolding (`config.yaml`, `logging.yaml`,
      `settings.py`)
- [x] CI-ready GitHub Actions workflows (lint, type-check, test)
- [x] Contribution, security, and governance docs
- [x] Docker development environment scaffold
- [x] Test suite scaffold

## Phase 1 — Core Virtual Memory Substrate

- [ ] Implement `WorkingMemory` with real capacity enforcement
- [ ] Implement `PageTable` residency + metadata tracking
- [ ] Implement `PageFaultHandler` wiring to a stub retriever
- [ ] Implement `LRUPolicy`, `FIFOPolicy` as first concrete replacement
      policies
- [ ] Unit tests covering hit/miss/eviction behavior

## Phase 2 — Memory Subsystems

- [ ] Implement `EpisodicMemory` (append + time-range query)
- [ ] Implement `VectorStore` with an in-memory/FAISS backend
- [ ] Implement `KnowledgeGraph` with an in-memory backend
- [ ] Implement `DiskPageStore` with a local filesystem backend

## Phase 3 — Hybrid Retrieval

- [ ] Implement `HybridRetriever` fusing vector + graph results
- [ ] Add configurable fusion/ranking strategies
- [ ] Integration tests: end-to-end page fault → retrieval → install

## Phase 4 — Learned & Adaptive Policies

- [ ] `LFUPolicy` and additional classical policies (Clock, ARC)
- [ ] Belady's-algorithm offline-optimal baseline for evaluation
- [ ] Exploration of learned/adaptive replacement policies

## Phase 5 — Agent Integration & Benchmarks

- [ ] `BaseAgent` reference implementation
- [ ] Example agents under `examples/`
- [ ] Benchmark suite under `benchmarks/` for long-horizon memory tasks
- [ ] Public experiment results under `experiments/`

## Phase 6 — Research Publication

- [ ] Stabilize public API (target `1.0`)
- [ ] Write accompanying paper draft
- [ ] Publish benchmark results and reproducibility artifacts

## Non-Goals (For Now)

- Production-hardened distributed deployment
- Multi-tenant memory isolation
- A hosted/managed service

These may be reconsidered post-`1.0` depending on community interest.

## How to Influence the Roadmap

Open a discussion or issue referencing the relevant phase — see
[CONTRIBUTING.md](../CONTRIBUTING.md).
