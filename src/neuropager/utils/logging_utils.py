"""Logging configuration helpers for NeuroPager.

Wraps the standard library's ``logging.config.dictConfig`` around the
project's ``configs/logging.yaml`` file so all subsystems share consistent
formatting and handler configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_LOGGING_CONFIG_PATH = Path("configs/logging.yaml")


def configure_logging(config_path: Path = DEFAULT_LOGGING_CONFIG_PATH) -> None:
    """Configure process-wide logging from a YAML dictConfig file.

    Args:
        config_path: Path to a logging configuration file matching the
            schema in ``configs/logging.yaml``.

    Raises:
        NotImplementedError: Always — logging configuration loading is not
            yet implemented. See ``docs/roadmap.md`` (Phase 1).
    """
    # TODO(neuropager): load YAML at `config_path` and pass the resulting
    # dict to `logging.config.dictConfig`.
    raise NotImplementedError("Logging configuration is not yet implemented.")


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger under the ``neuropager`` namespace.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A standard library :class:`logging.Logger` instance.
    """
    return logging.getLogger(f"neuropager.{name}")
