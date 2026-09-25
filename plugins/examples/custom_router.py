"""Example router plugin proving the ``workbench.routers`` extension point (PLAN §3).

Defines an example keyword-based router backend that routes based on explicit
keywords or difficulty signals in the prompt.
Can be activated in .workbench.toml:
    [routing]
    backend = "keyword"
"""

from __future__ import annotations

from workbench.core.protocols import Classification
from workbench.core.registry import Registry


class KeywordRouter:
    """Simple keyword-matching router implementing the Router protocol."""

    async def classify(self, message: str) -> Classification:
        lowered = message.lower()
        if any(w in lowered for w in ("plan", "roadmap", "architecture", "design")):
            intent = "planning"
            difficulty = 2.5
            needs_tools = True
        elif any(w in lowered for w in ("edit", "fix", "refactor", "change", "replace")):
            intent = "file_edit"
            difficulty = 1.5
            needs_tools = True
        elif any(w in lowered for w in ("run", "bash", "shell", "exec", "test")):
            intent = "shell_task"
            difficulty = 1.0
            needs_tools = True
        elif any(w in lowered for w in ("write", "create", "implement", "code")):
            intent = "code_gen"
            difficulty = 2.0
            needs_tools = True
        elif any(w in lowered for w in ("what", "how", "why", "who", "explain")):
            intent = "chat_qa"
            difficulty = 0.5
            needs_tools = False
        else:
            intent = "chat_qa"
            difficulty = 0.5
            needs_tools = False

        return Classification(
            intent=intent,
            difficulty=difficulty,
            confidence=0.85,
            needs_tools=needs_tools,
            backend="keyword",
        )


def register(registry: Registry) -> None:
    """Registration hook when loaded from ~/.config/workbench/plugins/."""
    registry.register("routers", "keyword", KeywordRouter())
