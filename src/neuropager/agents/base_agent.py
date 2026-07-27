"""Base agent interface for integrating an LLM agent with NeuroPager.

Defines the minimal contract an agent implementation satisfies to read from
and write to a :class:`~neuropager.core.memory_manager.MemoryManager`. This
is intentionally minimal — NeuroPager manages memory, not agent control
flow or tool use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from neuropager.core.memory_manager import MemoryManager


class BaseAgent(ABC):
    """Abstract interface for an LLM agent that uses NeuroPager for memory.

    Attributes:
        memory: The memory manager instance this agent reads from and
            writes to.
    """

    def __init__(self, memory: MemoryManager) -> None:
        """Initialize the agent with a bound memory manager.

        Args:
            memory: The memory manager instance this agent uses.
        """
        self.memory = memory

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Produce an action/response given an observation.

        Implementations are expected to consult :attr:`memory` as part of
        forming a response, and to record the resulting interaction back
        into memory.

        Args:
            observation: The input observation (e.g., a user message,
                environment state).

        Returns:
            The agent's action or response.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError
