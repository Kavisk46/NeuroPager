# configs/

Configuration files for NeuroPager runtime behavior.

| File | Purpose |
| --- | --- |
| [`config.yaml`](config.yaml) | Default memory manager configuration (working memory capacity, backend selection, retrieval fusion weights, etc.). Loaded via `neuropager.settings`. |
| [`logging.yaml`](logging.yaml) | Standard-library `logging.config.dictConfig`-compatible logging setup. Loaded via `neuropager.utils.logging_utils`. |

## Conventions

- Configuration is layered: defaults live here; environment-specific or
  experiment-specific overrides should live under `experiments/<name>/` and
  be merged over these defaults rather than duplicating the whole file.
- Do not hardcode secrets or credentials in these files. Use environment
  variables and `.env` (gitignored) for anything sensitive.
