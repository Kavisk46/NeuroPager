# docker/

Development container environment for NeuroPager.

## Usage

```bash
# Build and start an interactive dev shell
docker compose -f docker/docker-compose.yml up -d
docker exec -it neuropager-dev bash

# Or build/run directly
docker build -f docker/Dockerfile -t neuropager:dev .
docker run -it --rm -v "$(pwd)":/workspace neuropager:dev bash
```

## Contents

| File | Purpose |
| --- | --- |
| [`Dockerfile`](Dockerfile) | Editable-install dev image with dev/test dependencies. |
| [`docker-compose.yml`](docker-compose.yml) | Dev service definition, with commented-out placeholders for future vector DB / graph DB backend services. |

## Notes

This image is intended for **local development and CI parity**, not
production deployment — there is no production deployment target yet since
no algorithmic implementation exists. Revisit once `neuropager` has runnable
components (see [docs/roadmap.md](../docs/roadmap.md)).
