# NeuroPager development image.
#
# This is a scaffold image for local development and CI parity. It installs
# the package in editable mode with dev dependencies; no algorithmic
# implementation exists yet.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# System dependencies (kept minimal; extend as real backends are added,
# e.g. build tools for vector index libraries).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -e ".[dev]"

COPY . .

CMD ["python", "-c", "import neuropager; print(neuropager.__version__)"]
