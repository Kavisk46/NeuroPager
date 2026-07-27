# API Reference

> **Placeholder.** NeuroPager does not yet implement algorithmic behavior,
> so this page documents the *public surface* currently defined as typed
> stubs, not runtime behavior. Once implementations land, this page should
> be replaced by generated API docs (e.g., via `mkdocstrings`).

## Generating API Docs (Planned)

Once the package has real implementations, API reference documentation
should be generated automatically from docstrings rather than hand-written,
to avoid drift. Planned approach:

```bash
pip install -e ".[docs]"
mkdocs serve
```

using [mkdocstrings](https://mkdocstrings.github.io/) against the Google-style
docstrings already present throughout `src/neuropager/`.

## Current Public Surface

### `neuropager.core`

| Symbol | Description |
| --- | --- |
| `MemoryManager` | Top-level orchestrator for agent-facing memory operations. |
| `WorkingMemory` | Bounded resident memory container. |
| `PageTable` | Residency and access metadata tracking. |
| `PageFaultHandler` | Services misses by invoking retrieval and installing results. |

### `neuropager.memory`

| Symbol | Description |
| --- | --- |
| `EpisodicMemory` | Time-ordered event log. |
| `KnowledgeGraph` | Symbolic entity/relation store. |
| `VectorStore` | Dense embedding index. |
| `DiskPageStore` | Durable cold storage. |

### `neuropager.retrieval`

| Symbol | Description |
| --- | --- |
| `HybridRetriever` | Combines vector + graph retrieval into ranked results. |

### `neuropager.policies`

| Symbol | Description |
| --- | --- |
| `PageReplacementPolicy` | Abstract base for eviction strategies. |
| `LRUPolicy` | Least-recently-used eviction (stub). |
| `LFUPolicy` | Least-frequently-used eviction (stub). |
| `FIFOPolicy` | First-in-first-out eviction (stub). |

### `neuropager.agents`

| Symbol | Description |
| --- | --- |
| `BaseAgent` | Minimal interface for an LLM agent to integrate with the memory manager. |

### `neuropager.utils`

| Symbol | Description |
| --- | --- |
| `types` | Shared type aliases and dataclasses (e.g., `MemoryKey`, `MemoryPage`). |
| `logging_utils` | Logging configuration helpers. |

### `neuropager.settings`

Configuration loading via `config.yaml` / `logging.yaml`, exposed as a
typed `Settings` object.

## Stability

Everything above is **pre-alpha and unstable**. No backward-compatibility
guarantees are made until a `1.0` release. See [roadmap.md](roadmap.md).
