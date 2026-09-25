"""Display shaping for tool-call segments: compact args and inline diffs.

Raw tool arguments can carry a whole file (``write``/``edit``); rendering them
as JSON would bury the tool block under payload, so those two tools drop the
content from the args panel and get a unified diff segment instead (ruling
M6.6). Patch headers are built from the emitted lines themselves so counts
always match what the DiffBlock parser will see — including the truncation
marker lines.
"""

from __future__ import annotations

import json
from typing import Any

# Display cap: a diff panel longer than this scrolls a small terminal for no
# reader; sources are prefix-truncated and marked, never sliced mid-hunk.
_DIFF_MAX_LINES = 80

_TRUNCATION_MARK = "… (more lines not shown)"


def compact_arguments(tool: str, arguments_json: str) -> dict[str, Any] | str:
    """Args for the block header: drop payloads that are about to be rendered
    as a diff of their own. Unparsable JSON passes through verbatim — the error
    path shows the model's raw arguments anyway."""
    payload = _payload(arguments_json)
    if not isinstance(payload, dict):
        return arguments_json
    if tool == "write":
        return {key: payload[key] for key in ("path",) if key in payload}
    if tool == "edit":
        return {key: payload[key] for key in ("path", "replace_all") if key in payload}
    return payload


def build_diff(tool: str, arguments_json: str) -> tuple[str, str] | None:
    """``(path, patch)`` a successful write/edit should render; None for the rest."""
    payload = _payload(arguments_json)
    if not isinstance(payload, dict):
        return None
    path = payload.get("path")
    if not isinstance(path, str):
        return None
    if tool == "write":
        content = payload.get("content")
        if not isinstance(content, str):
            return None
        return path, _added_file_patch(path, content)
    if tool == "edit":
        old_string = payload.get("old_string")
        new_string = payload.get("new_string")
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            return None
        return path, _replacement_patch(path, old_string, new_string)
    return None


def _payload(arguments_json: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(arguments_json) if arguments_json.strip() else None
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _added_file_patch(path: str, content: str) -> str:
    """New-file patch: every content line added, capped with a marked tail."""
    lines = content.splitlines()
    shown = lines[:_DIFF_MAX_LINES]
    omitted = len(lines) - len(shown)
    if omitted > 0:
        shown = [*shown, f"{_TRUNCATION_MARK} ({omitted} lines)"]
    header = f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(shown)} @@\n"
    body = "".join(f"+{line}\n" for line in shown)
    return header + body


def _replacement_patch(path: str, old_string: str, new_string: str) -> str:
    """Edit patch as one context-free hunk; oversized sides are prefix-truncated
    and get a marker line so the header counts still match the emitted lines."""
    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()
    omitted_old, omitted_new = _fit(old_lines, new_lines)
    old_count = len(old_lines) + omitted_old
    new_count = len(new_lines) + omitted_new
    header = f"--- a/{path}\n+++ b/{path}\n@@ -1,{old_count} +1,{new_count} @@\n"
    body = "".join(f"-{line}\n" for line in old_lines)
    if omitted_old:
        body += f"-{_TRUNCATION_MARK} ({omitted_old} lines)\n"
    body += "".join(f"+{line}\n" for line in new_lines)
    if omitted_new:
        body += f"+{_TRUNCATION_MARK} ({omitted_new} lines)\n"
    return header + body


def _fit(old_lines: list[str], new_lines: list[str]) -> tuple[int, int]:
    """Trim the longer side until lines + markers fit the display budget.

    Both lists are trimmed in place (the caller keeps using them); returns the
    omitted-line counts for (old, new) so the caller can size the header and
    append the marker lines.
    """
    omitted_old = omitted_new = 0
    while len(old_lines) + len(new_lines) + omitted_old + omitted_new > _DIFF_MAX_LINES:
        if len(old_lines) >= len(new_lines) and old_lines:
            old_lines.pop()
            omitted_old += 1
        elif new_lines:
            new_lines.pop()
            omitted_new += 1
        else:
            break
    return omitted_old, omitted_new
