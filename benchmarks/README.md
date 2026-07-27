# benchmarks/

Benchmark suites for evaluating NeuroPager's memory subsystems.

## Purpose

Benchmarks here are intended to evaluate:

- **Page replacement policy quality** — hit rate under realistic agent
  memory access traces, compared against Belady's-optimal offline baseline.
- **Retrieval quality** — precision/recall of hybrid retrieval vs.
  vector-only / graph-only baselines.
- **End-to-end task performance** — long-horizon agent task success rate
  with vs. without NeuroPager-managed memory.

## Conventions

```
benchmarks/
├── traces/        # synthetic or recorded memory access traces
├── tasks/         # long-horizon agent task definitions
└── results/       # benchmark run outputs
```

See [docs/research.md](../docs/research.md) for the evaluation philosophy
this suite is meant to support.

*No benchmarks are implemented yet — this is a structural placeholder.*
