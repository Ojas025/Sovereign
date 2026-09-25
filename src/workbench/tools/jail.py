"""Path jail: file-tool paths must resolve inside the workspace (plan §5.3.1).

Layer 1 of the guardrails, applied by read/write/edit regardless of sandbox
backend: `..` traversal, symlink hops, absolute paths, and `~` expansion are
all resolved first, then checked for containment. Writes are workspace-only;
reads may additionally target explicitly declared extra read roots.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path


class PathJailError(Exception):
    """A path escaped the workspace or failed structural validation."""


class PathJail:
    """Resolves paths and enforces containment against the allowed roots."""

    def __init__(self, root: Path, read_roots: Sequence[Path] = ()) -> None:
        self._root = root.expanduser().resolve()
        self._read_roots = tuple(r.expanduser().resolve() for r in read_roots)

    def resolve_read(self, raw: str | Path) -> Path:
        """Resolve a read path inside the workspace or any declared read root."""
        return self._resolve_within(raw, (self._root, *self._read_roots))

    def resolve_write(self, raw: str | Path) -> Path:
        """Resolve a write path; only the workspace root is writable."""
        return self._resolve_within(raw, (self._root,))

    def _resolve_within(self, raw: str | Path, roots: tuple[Path, ...]) -> Path:
        try:
            text = os.fspath(raw)
        except TypeError as exc:
            raise PathJailError(f"path must be a string, got {type(raw).__name__}") from exc
        if not text or "\x00" in text:
            raise PathJailError("path must be non-empty and free of NUL bytes")

        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate
        # strict=False: the leaf may not exist yet (writes), but symlinks in the
        # existing prefix are followed — a hop outside is visible before any I/O.
        resolved = candidate.resolve()
        for root in roots:
            if resolved.is_relative_to(root):
                return resolved
        raise PathJailError(f"path resolves outside the allowed roots: {text!r} -> {resolved}")
