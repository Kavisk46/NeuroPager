# experiments/

Reproducible experiment configurations and results.

## Purpose

Each subdirectory represents a single, reproducible experiment: a named
configuration override layered on top of `configs/config.yaml`, plus the
resulting metrics/artifacts. This keeps research results traceable back to
the exact configuration that produced them.

## Conventions

```
experiments/
└── <experiment-name>/
    ├── config.yaml       # overrides layered on configs/config.yaml
    ├── README.md         # hypothesis, setup, and findings
    └── results/          # metrics, logs, plots (gitignored if large)
```

- Name experiments descriptively and include a date prefix for ordering,
  e.g. `2026-08-01-lru-vs-lfu-baseline/`.
- Large result artifacts (raw logs, model outputs) should not be committed
  directly — summarize in the experiment's `README.md` and link to external
  storage if needed.
- See [docs/research.md](../docs/research.md) for the research questions
  experiments here are expected to inform.

*No experiments have been run yet — this is a structural placeholder.*
