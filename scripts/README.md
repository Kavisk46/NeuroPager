# scripts/

Developer and operational utility scripts.

## Purpose

Command-line scripts supporting development workflows that don't belong in
the installable package itself — e.g., environment setup helpers,
benchmark runners, data preparation utilities.

## Conventions

- Scripts should be executable and include a `#!/usr/bin/env python3`
  shebang plus a module docstring explaining usage.
- Prefer importing logic from `neuropager` rather than duplicating it here;
  scripts should be thin entry points.

*No scripts exist yet — this is a structural placeholder.*
