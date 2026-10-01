#!/usr/bin/env python3
"""Microbenchmark: raw per-file JSON write/read cost vs. in-memory dict cost.

Isolates DiskPageStore's actual I/O cost from everything else in a replay,
using the exact same write/read pattern (SHA-256 filename, json.dumps of a
small record, Path.write_text) on a small N so it finishes in seconds
rather than hours -- a full 2000-tick episode replay was observed taking
far longer than the previously-measured ~165s/replay baseline (roughly
22s/eviction rather than ~0.1s/eviction), so this isolates the write path
directly before spending more wall-clock time waiting on a full replay.

Usage:
    PYTHONPATH=src python scripts/microbench_disk_io.py
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from neuropager.memory.disk_store import DiskPageStore
from neuropager.utils.types import MemoryKey, MemoryPage, MemoryTier

N = 200


def make_page(i: int) -> MemoryPage:
    """Build a small representative page, matching what the experiment writes."""
    now = datetime.now(UTC)
    return MemoryPage(
        key=MemoryKey(f"key-{i}"),
        content=f"key-{i}",
        tier=MemoryTier.WORKING,
        created_at=now,
        last_accessed_at=now,
        access_count=1,
        metadata={},
    )


def bench_disk_store_write(root: Path) -> float:
    """Time N DiskPageStore.write() calls to a fresh directory."""
    store = DiskPageStore(root)
    start = time.perf_counter()
    for i in range(N):
        store.write(MemoryKey(f"key-{i}"), make_page(i))
    return time.perf_counter() - start


def bench_disk_store_read(root: Path) -> float:
    """Time N DiskPageStore.read() calls (files already written by the write bench)."""
    store = DiskPageStore(root)
    start = time.perf_counter()
    for i in range(N):
        store.read(MemoryKey(f"key-{i}"))
    return time.perf_counter() - start


def bench_raw_file_write(root: Path) -> float:
    """Time N raw Path.write_text calls with the same JSON payload shape, no class overhead."""
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for i in range(N):
        digest = hashlib.sha256(f"key-{i}".encode()).hexdigest()
        record = {
            "key": f"key-{i}",
            "content": f"key-{i}",
            "tier": "working",
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_accessed_at": "2026-01-01T00:00:00+00:00",
            "access_count": 1,
            "metadata": {},
        }
        (root / f"{digest}.json").write_text(json.dumps(record), encoding="utf-8")
    return time.perf_counter() - start


def bench_in_memory_dict(n: int) -> float:
    """Time N dict insertions of the same record shape -- the zero-I/O baseline."""
    store: dict[str, dict] = {}
    start = time.perf_counter()
    for i in range(n):
        record = {
            "key": f"key-{i}",
            "content": f"key-{i}",
            "tier": "working",
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_accessed_at": "2026-01-01T00:00:00+00:00",
            "access_count": 1,
            "metadata": {},
        }
        store[f"key-{i}"] = record
    return time.perf_counter() - start


def main() -> None:
    """Run each microbenchmark and print seconds/op for each."""
    scratch = Path("experiments/_microbench_scratch")

    print(f"N = {N} operations each\n")

    t = bench_disk_store_write(scratch / "diskstore_write")
    print(f"DiskPageStore.write x{N}:       {t:.4f}s total, {t / N * 1000:.2f}ms/op")

    t = bench_disk_store_read(scratch / "diskstore_write")
    print(f"DiskPageStore.read  x{N}:       {t:.4f}s total, {t / N * 1000:.2f}ms/op")

    t = bench_raw_file_write(scratch / "raw_write")
    print(f"raw Path.write_text x{N}:       {t:.4f}s total, {t / N * 1000:.2f}ms/op")

    t = bench_in_memory_dict(N)
    print(f"in-memory dict insert x{N}:     {t:.4f}s total, {t / N * 1000:.4f}ms/op")

    t = bench_in_memory_dict(10000)
    print(f"in-memory dict insert x10000:  {t:.4f}s total, {t / 10000 * 1000:.4f}ms/op")


if __name__ == "__main__":
    main()
