"""File tools: read / write / edit with caps and path-jail enforcement (plan §5.2)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from workbench.config import ToolsConfig
from workbench.core.protocols import ToolContext, ToolResult
from workbench.tools.jail import PathJail, PathJailError

# Stable reason token for tool_blocked_total (M7.5): every PathJail rejection
# is a policy denial, whatever the underlying message says.
_PATH_JAIL = "path_jail"


class _BadArgument(Exception):
    """A model-supplied argument failed structural validation."""


def _require_str(arguments: Mapping[str, object], name: str, *, allow_empty: bool = False) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or (not allow_empty and not value):
        expectation = "a string" if allow_empty else "a non-empty string"
        raise _BadArgument(f"{name} must be {expectation}")
    return value


def _optional_positive_int(arguments: Mapping[str, object], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _BadArgument(f"{name} must be a positive integer")
    return value


def _line_numbered(lines: list[str], start: int) -> str:
    return "\n".join(f"{line_no:5d}: {line}" for line_no, line in enumerate(lines, start=start))


def _jail(tools: ToolsConfig, context: ToolContext) -> PathJail:
    """Per-call jail: workspace from the call context, extra read roots from config."""
    return PathJail(context.workspace_root, [Path(root) for root in tools.read_roots])


class ReadTool:
    name: str = "read"
    description: str = (
        "Read a file from the workspace. Returns line-numbered content; "
        "use offset (1-based) and limit to page through large files."
    )
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "file to read (workspace-relative)"},
            "offset": {"type": "integer", "description": "1-based line number to start from"},
            "limit": {"type": "integer", "description": "maximum number of lines to return"},
        },
        "required": ["path"],
    }

    def __init__(self, tools: ToolsConfig) -> None:
        self._tools = tools

    async def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        try:
            raw_path = _require_str(arguments, "path")
            offset = _optional_positive_int(arguments, "offset")
            limit = _optional_positive_int(arguments, "limit")
            path = _jail(self._tools, context).resolve_read(raw_path)
        except _BadArgument as exc:
            return ToolResult(str(exc), is_error=True)
        except PathJailError as exc:
            return ToolResult(f"blocked: {exc}", is_error=True, blocked_reason=_PATH_JAIL)

        if not path.exists():
            return ToolResult(f"file not found: {raw_path}", is_error=True)
        if path.is_dir():
            return ToolResult(f"is a directory: {raw_path}", is_error=True)
        size = path.stat().st_size
        if size > self._tools.read_max_bytes:
            return ToolResult(
                f"file too large: {size} bytes exceeds the "
                f"{self._tools.read_max_bytes}-byte read cap",
                is_error=True,
            )
        try:
            # newline="" preserves CRLF endings exactly as stored on disk
            text = path.read_text(encoding="utf-8", newline="")
        except UnicodeDecodeError:
            return ToolResult(f"not valid UTF-8 text: {raw_path}", is_error=True)
        except OSError as exc:
            return ToolResult(f"cannot read {raw_path}: {exc}", is_error=True)

        lines = text.splitlines()
        if offset is not None and offset > len(lines):
            return ToolResult(
                f"offset {offset} is beyond end of file ({len(lines)} lines)", is_error=True
            )
        selected = lines[(offset - 1) if offset else 0 :]
        if limit is not None:
            selected = selected[:limit]
        return ToolResult(_line_numbered(selected, start=offset or 1))


class WriteTool:
    name: str = "write"
    description: str = "Create or overwrite a file in the workspace with the given content."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "file to write (workspace-relative)"},
            "content": {"type": "string", "description": "full file content"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, tools: ToolsConfig) -> None:
        self._tools = tools

    async def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        try:
            raw_path = _require_str(arguments, "path")
            content = _require_str(arguments, "content", allow_empty=True)
            path = _jail(self._tools, context).resolve_write(raw_path)
        except _BadArgument as exc:
            return ToolResult(str(exc), is_error=True)
        except PathJailError as exc:
            return ToolResult(f"blocked: {exc}", is_error=True, blocked_reason=_PATH_JAIL)

        encoded = content.encode("utf-8")
        if len(encoded) > self._tools.write_max_bytes:
            return ToolResult(
                f"content too large: {len(encoded)} bytes exceeds the "
                f"{self._tools.write_max_bytes}-byte write cap",
                is_error=True,
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="")
        except OSError as exc:
            return ToolResult(f"cannot write {raw_path}: {exc}", is_error=True)
        return ToolResult(f"wrote {len(encoded)} bytes to {raw_path}")


class EditTool:
    name: str = "edit"
    description: str = (
        "Replace an exact string in a workspace file. old_string must be unique "
        "unless replace_all is true; the file must already exist."
    )
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "file to edit (workspace-relative)"},
            "old_string": {"type": "string", "description": "exact text to replace"},
            "new_string": {"type": "string", "description": "replacement text (may be empty)"},
            "replace_all": {"type": "boolean", "description": "replace every occurrence"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, tools: ToolsConfig) -> None:
        self._tools = tools

    async def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        try:
            raw_path = _require_str(arguments, "path")
            old_string = _require_str(arguments, "old_string")
            new_string = _require_str(arguments, "new_string", allow_empty=True)
            replace_all = arguments.get("replace_all", False)
            if not isinstance(replace_all, bool):
                raise _BadArgument("replace_all must be a boolean")
            # editing writes, so it is workspace-only even for readable extra roots
            path = _jail(self._tools, context).resolve_write(raw_path)
        except _BadArgument as exc:
            return ToolResult(str(exc), is_error=True)
        except PathJailError as exc:
            return ToolResult(f"blocked: {exc}", is_error=True, blocked_reason=_PATH_JAIL)

        if not path.exists():
            return ToolResult(f"file not found: {raw_path}", is_error=True)
        size = path.stat().st_size
        if size > self._tools.read_max_bytes:
            return ToolResult(
                f"file too large: {size} bytes exceeds the "
                f"{self._tools.read_max_bytes}-byte cap for editing",
                is_error=True,
            )
        try:
            text = path.read_text(encoding="utf-8", newline="")
        except UnicodeDecodeError:
            return ToolResult(f"not valid UTF-8 text: {raw_path}", is_error=True)
        except OSError as exc:
            return ToolResult(f"cannot read {raw_path}: {exc}", is_error=True)

        matches = text.count(old_string)
        if matches == 0:
            return ToolResult(f"old_string not found in {raw_path}", is_error=True)
        if matches > 1 and not replace_all:
            return ToolResult(
                f"{matches} matches for old_string in {raw_path}; it must be unique "
                f"(set replace_all=true to replace every occurrence)",
                is_error=True,
            )

        updated = text.replace(old_string, new_string) if replace_all else text.replace(
            old_string, new_string, 1
        )
        try:
            path.write_text(updated, encoding="utf-8", newline="")
        except OSError as exc:
            return ToolResult(f"cannot write {raw_path}: {exc}", is_error=True)
        occurrences = matches if replace_all else 1
        plural = "s" if occurrences != 1 else ""
        return ToolResult(f"replaced {occurrences} occurrence{plural} in {raw_path}")
