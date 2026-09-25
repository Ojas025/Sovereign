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
            r"\b(?:who|what) are you\b"
            r"|\byour (?:capabilit\w*|limits?|name|prompt|instructions?|system)\b"
            r"|\bwhich model\b|\babout yourself\b|\bare you (?:an?|a)\b|\bwhat can you do\b",
            re.IGNORECASE,
        ),
    ),
    (
        "planning",
        re.compile(
            r"^(?!.*?\b(?:write|create|generate)\b.*?\b(?:docx|pdf|invoice)\b)"
            r".*?\b(?:plan\w*|roadmap|strateg\w*|architect\w*|design|steps|approach|blueprint)\b"
            r"|\bbreak (?:it|this) down\b"
            r"|\bhow (?:should|to) (?:we|i) (?:approach|structure|plan)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "code_gen",
        re.compile(
            r"\b(?:write|create|implement|generate|add|refactor|build|develop|code|scaffold|draft|produce|construct)\b"
            r"[^.?!]*\b(?:function|class|script|module|component|feature|program|test|api|endpoint|snippet|parser"
            r"|code|app|application|pdf|docx|doc|document|report|file|markdown|tool|handler|service|scaffold"
            r"|algorithm|bot|scraper|server|client|query|prompt|benchmark|solution|model)\b"
            r"|\brefactor\w*\b"
            r"|\b(?:write|code|create|generate)\b.*?\b(?:in|using|with)\s+(?:python|bash|sh|c|cpp|rust|go|javascript|typescript|sql|html|css)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "shell_task",
        re.compile(
            r"\b(?:run|execute|install|uninstall|deploy|restart|chmod|kill"
            r"|ssh|curl|docker|git|npm|pip|uv|apt|grep|make|build|pytest|cargo|poetry|yarn|pnpm|bun"
            r"|compile|diff|commit|push|pull|checkout|branch|rebase|stash|clone)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "file_edit",
        re.compile(
            r"\b(?:edit|patch|rename|move|delete|append|prepend|replace|update|fix|modify|change|rewrite"
            r"|clean|tweak|adjust|remove|revert|format|lint)\b"
            r"|\bfix\b.*?\b(?:bug|issue|error|crash|problem|typo|warning|exception|traceback|leak)\b"
            r"|\b[\w./~-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|toml|json"
            r"|yaml|yml|md|cfg|ini|txt|c|h|cpp|docx|pdf|env|sql|graphql|proto|lock|log|csv|tsv|xml|html|css)\b"
            r"|\b(?:Dockerfile|Makefile)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "search_analysis",
        re.compile(
            r"\b(?:find|locate|search|look for|explain|why|how does|how do"
            r"|analy[sz]e|compare|summar\w*|trace|diagnos\w*|check|inspect"
            r"|review|audit|read|show me|where is|list|overview|examine|investigate)\b",
            re.IGNORECASE,
        ),
    ),
)

# Complexity markers raise the ordinal difficulty (0-3, plan §4.3).
_COMPLEXITY = re.compile(
    r"```|\b(?:refactor\w*|entire|across|redesign|rewrite|migrat\w*"
    r"|integrat\w*|from scratch|codebase|concurren\w*|multithread\w*"
    r"|architecture|async\w*|performance|optimiz\w*|deadlock|race condition"
    r"|memory leak|security|vulnerabilit\w*)\b",
    re.IGNORECASE,
)
_MULTISTEP = re.compile(
    r"^\s*\d+[.)]\s|^\s*[-*]\s|\bthen\b|\bstep by step\b|\bafter that\b|\bnext\b",
    re.IGNORECASE | re.MULTILINE,
)
_PATH_REFERENCE = re.compile(
    r"\b[\w./~-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|toml|json|yaml|yml"
    r"|md|cfg|ini|txt|c|h|cpp|docx|pdf|env|sql|graphql|proto|lock|log|csv|tsv|xml|html|css)\b"
    r"|\b(?:src|tests|test|lib|docs|config)/|\b(?:Dockerfile|Makefile)\b",
    re.IGNORECASE,
)

_TOOL_INTENTS = frozenset({"file_edit", "code_gen", "shell_task"})
_DEFAULT_INTENT = "chat_qa"

_LONG_WORDS = 25
_LONG_CHARS = 160
_MAX_DIFFICULTY = 3
_VAGUE_WORDS = 4

_CONFIDENCE_MATCHED = 0.85
_CONFIDENCE_PLAIN = 0.7
_CONFIDENCE_VAGUE = 0.5


def _difficulty(message: str) -> float:
    """Ordinal 0-3 from length, multi-step structure and complexity markers."""
    score = 0
    words = len(message.split())
    chars = len(message)
    if words >= _LONG_WORDS or chars >= _LONG_CHARS:
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
