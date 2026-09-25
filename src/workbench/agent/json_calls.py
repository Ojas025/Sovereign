"""Tool calls for models whose chat template has no native tool support.

The system prompt asks the model to emit fenced ```tool blocks; the parser
extracts them in document order. Blocks that are not a ``{"name": ..., "arguments": {...}}``
object come back as errors so the loop can feed the failure back for a retry
instead of silently dropping the call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_BLOCK_PATTERN = re.compile(r"```tool\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class JsonToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class JsonToolError:
    message: str


JsonToolAction = JsonToolCall | JsonToolError


def parse_json_tool_calls(text: str) -> tuple[JsonToolAction, ...]:
    """Parse every ```tool fence in ``text``, preserving order; no fences → empty."""
    actions: list[JsonToolAction] = []
    for match in _BLOCK_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as error:
            actions.append(JsonToolError(f"invalid JSON in tool block: {error}"))
            continue
        if not isinstance(payload, dict):
            actions.append(JsonToolError("tool block must be a JSON object"))
            continue
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            actions.append(JsonToolError("tool block needs a non-empty string 'name'"))
            continue
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            actions.append(JsonToolError(f"'arguments' for {name!r} must be a JSON object"))
            continue
        actions.append(JsonToolCall(name=name, arguments=arguments))
    return tuple(actions)
