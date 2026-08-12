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


def extract_claude_text(response) -> str:
    """
    Pull the text reply out of a Claude Messages API response.

    Some models (e.g. with extended thinking) return non-text content
    blocks — like a "thinking" block — ahead of the actual text block,
    so content[0] isn't reliably the answer. Also strips a surrounding
    ```json / ``` markdown fence, which models commonly wrap JSON in
    even when told not to.
    """
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise ValueError("Claude response contained no text block")

    text = text_block.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    return text
