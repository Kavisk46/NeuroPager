# Design Notes

This document covers module-level design decisions and interface contracts
for NeuroPager. It complements [architecture.md](architecture.md) (system
structure) and [memory.md](memory.md) (memory content model) with the "how"
of individual components.

> **Status:** All interfaces below are defined as typed stubs with
> docstrings and `TODO` markers in `src/neuropager/`. No algorithmic
> implementations exist yet — this document describes contracts, not
> behavior.

## Design Principles

1. **Interfaces before implementations.** Every pluggable subsystem
   (retrieval backend, replacement policy, memory store) is defined as an
   abstract base class first, so multiple implementations can be swapped
   without touching the memory manager.
2. **Explicit over implicit.** Page residency, eviction decisions, and
   retrieval provenance should be introspectable — a researcher should be
   able to ask "why is this page here" and get an answer.
3. **Composable, not monolithic.** The `MemoryManager` orchestrates
   collaborators; it does not itself implement retrieval, storage, or
   eviction logic.
4. **Config-driven.** Behavior (which policy, which backends, capacity
   limits) is controlled via `configs/config.yaml` and `settings.py`, not
   hardcoded.

## Module Contracts

### `core.memory_manager.MemoryManager`

The façade the agent interacts with. Expected responsibilities (interfaces
only at this stage):

- `get(key)` — resolve a memory unit, resulting in a page hit or triggering
  a page fault.
- `put(key, value)` — write a new or updated memory unit.
- `evict(key)` — explicit eviction, delegating to the active policy.

Collaborators (injected, not constructed internally): `WorkingMemory`,
`PageTable`, `PageFaultHandler`, `PageReplacementPolicy`.

### `core.working_memory.WorkingMemory`

Bounded container representing resident memory. Expected to expose
capacity, current occupancy, and simple insert/remove/contains operations.

### `core.page_table.PageTable`

Tracks, per logical memory ID: residency location, access metadata (last
access time, access count), and dirty/clean state. This is the data
structure replacement policies read from to make eviction decisions.

### `core.page_fault.PageFaultHandler`

Given a logical memory ID not resident in working memory, coordinates with
`retrieval.HybridRetriever` to fetch it and with `WorkingMemory` /
`PageReplacementPolicy` to install it, evicting if necessary.

### `policies.base.PageReplacementPolicy`

Abstract interface all replacement policies implement:

- `select_victim(page_table)` — choose a page to evict.
- `on_access(key)` — hook invoked on every page access, to update policy
  state (e.g., recency for LRU, frequency for LFU).

Concrete stub implementations: `policies.lru.LRUPolicy`,
`policies.lfu.LFUPolicy`, `policies.fifo.FIFOPolicy`.

### `memory.episodic.EpisodicMemory`

Append-only(-ish) store of timestamped events. Expected interface:
`record(event)`, `query(filter)`.

### `memory.knowledge_graph.KnowledgeGraph`

Symbolic store of entities/relations. Expected interface: `add_triple`,
`query`, with pluggable backends (in-memory graph, external graph DB).

### `memory.vector_store.VectorStore`

Dense retrieval interface: `add(id, embedding, metadata)`,
`search(query_embedding, top_k)`.

### `memory.disk_store.DiskPageStore`

Durable key-value interface for cold storage: `write(key, page)`,
`read(key)`, `delete(key)`.

### `retrieval.hybrid_retriever.HybridRetriever`

Combines `VectorStore` and `KnowledgeGraph` query results into one ranked
list. Expected interface: `retrieve(query, top_k)`.

### `agents.base_agent.BaseAgent`

Minimal interface an LLM agent implementation would satisfy to plug into
NeuroPager's memory manager (e.g., `act(observation)`,
`remember(memory_manager)`). Present as a placeholder to make the "LLM
Agent" box in the architecture diagram concrete and testable.

## Configuration Surface

`settings.py` (via `configs/config.yaml`) is expected to expose, at
minimum:

- Working memory capacity
- Active page replacement policy name
- Vector store backend + parameters
- Knowledge graph backend + parameters
- Disk store path/backend
- Logging configuration (via `configs/logging.yaml`)

## Testing Strategy

- `tests/unit/` — one test module per package module, exercising interface
  contracts (e.g., abstract methods raise `NotImplementedError`, dataclass
  validation) once implementations land.
- `tests/integration/` — end-to-end flows across memory manager, page
  table, and retrieval once components are implemented.

At the current scaffold stage, tests are placeholders asserting the package
imports cleanly and public interfaces exist with the expected shape.
