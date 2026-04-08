"""Agent base class — borrowed pattern from GPUProfiler, simplified."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models import AgentContext, Task


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    name: str = "base"

    @abstractmethod
    def can_handle(self, task: Task) -> bool:
        """Return True if this agent can handle the given task kind."""
        ...

    @abstractmethod
    def run(self, task: Task, ctx: AgentContext) -> dict[str, Any]:
        """Execute the task and return a result dict."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
