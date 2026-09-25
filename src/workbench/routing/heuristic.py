"""HeuristicRouter: deterministic length/keyword/code-fence classification.

Trust floor of the routing subsystem: used by ``LayaRouter`` whenever laya is
unavailable, its inference errors, or its confidence falls below the gate.
Kept simple and auditable on purpose — it is a fallback, not a second model.
"""

from __future__ import annotations

import re

from workbench.core.protocols import Classification

# First match wins; anything unmatched is plain conversation (chat_qa).
_INTENT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "meta",
        re.compile(
            r"\b(?:who|what) are you\b|\byour (?:capabilit|limits|name)\b"
            r"|\bwhich model\b|\babout yourself\b",
            re.IGNORECASE,
        ),
    ),
    (
        "planning",
        re.compile(
            r"\b(?:plan|roadmap|strateg\w*|architect\w*|design|steps|approach"
            r"|blueprint)\b|\bbreak (?:it|this) down\b",
            re.IGNORECASE,
        ),
    ),
    (
        "code_gen",
        re.compile(
            r"\b(?:write|create|implement|generate|add|refactor|build)\b"
            r"[^.?!]*\b(?:function|class|script|module|component|feature"
            r"|program|test|api|endpoint|snippet|parser)\b|\brefactor\w*\b",
            re.IGNORECASE,
        ),
    ),
    (
        "shell_task",
        re.compile(
            r"\b(?:run|execute|install|uninstall|deploy|restart|chmod|kill"
            r"|ssh|curl|docker|git|npm|pip|uv|apt|grep|make|build)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "file_edit",
        re.compile(
            r"\b(?:edit|patch|rename|move|delete|append|prepend|replace|update)\b"
            r"|\b[\w./~-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|toml|json"
            r"|yaml|yml|md|cfg|ini|txt|c|h|cpp)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "search_analysis",
        re.compile(
            r"\b(?:find|locate|search|look for|explain|why|how does|how do"
            r"|analy[sz]e|compare|summar\w*|trace|diagnos\w*)\b",
            re.IGNORECASE,
        ),
    ),
)

# Complexity markers raise the ordinal difficulty (0-3, plan §4.3).
_COMPLEXITY = re.compile(
    r"```|\b(?:refactor\w*|entire|across|redesign|rewrite|migrat\w*"
    r"|integrat\w*|from scratch|codebase)\b",
    re.IGNORECASE,
)
_MULTISTEP = re.compile(
    r"^\s*\d+[.)]\s|^\s*[-*]\s|\bthen\b|\bstep by step\b",
    re.IGNORECASE | re.MULTILINE,
)
_PATH_REFERENCE = re.compile(
    r"\b[\w./~-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|toml|json|yaml|yml"
    r"|md|cfg|ini|txt|c|h|cpp)\b|\b(?:src|tests|test|lib|docs|config)/",
    re.IGNORECASE,
)

_TOOL_INTENTS = frozenset({"file_edit", "code_gen", "shell_task"})
_DEFAULT_INTENT = "chat_qa"

_LONG_WORDS = 30
_LONG_CHARS = 200
_MAX_DIFFICULTY = 3
_VAGUE_WORDS = 4

_CONFIDENCE_MATCHED = 0.85
_CONFIDENCE_PLAIN = 0.7
_CONFIDENCE_VAGUE = 0.5


def _difficulty(message: str) -> float:
    """Ordinal 0-3 from length, multi-step structure and complexity markers."""
    score = 0
    if len(message.split()) >= _LONG_WORDS or len(message) >= _LONG_CHARS:
        score += 1
    if message.count("\n") >= 2 or _MULTISTEP.search(message):
        score += 1
    if _COMPLEXITY.search(message):
        score += 1
    return float(min(score, _MAX_DIFFICULTY))


class HeuristicRouter:
    """Deterministic ``Router`` implementation (no model, no I/O)."""

    async def classify(self, message: str) -> Classification:
        intent = _DEFAULT_INTENT
        for name, pattern in _INTENT_RULES:
            if pattern.search(message):
                intent = name
                break

        words = len(message.split())
        if intent != _DEFAULT_INTENT:
            confidence = _CONFIDENCE_MATCHED
        else:
            confidence = _CONFIDENCE_PLAIN if words >= _VAGUE_WORDS else _CONFIDENCE_VAGUE

        needs_tools = intent in _TOOL_INTENTS or bool(_PATH_REFERENCE.search(message))
        return Classification(
            intent=intent,
            difficulty=_difficulty(message),
            confidence=confidence,
            needs_tools=needs_tools,
            backend="heuristic",
        )
