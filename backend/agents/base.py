"""Base agent class with common interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    async def run(self, ticker: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task and return structured output."""
        ...

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{self.name}] {msg}")
