"""Local GGUF store: search-path scan plus a per-directory index.json registry.

PLAN §6.1 promises a store the user can point at their own models: the scan
never hashes (listing must stay instant), while `register` — called after an
explicit download — computes the checksum and records it so later scans can
report provenance without paying for it again.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

_INDEX_NAME = "index.json"
_INDEX_VERSION = 1
_SHA_CHUNK_BYTES = 1 << 20
# Exporters append the quant to the file stem: MiniCPM5-2B-Q4_K_M -> family
# "MiniCPM5-2B", quant "Q4_K_M". IQ-variants (IQ2_XXS) share the pattern.
_QUANT_SUFFIX = re.compile(r"-((?:I)?Q\d(?:_[A-Z0-9]+)+)$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    """Checksum a model file in chunks — files here are gigabytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_SHA_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def human_size(size: int) -> str:
    """Render bytes for `models list`: B below 1 KiB, one decimal above."""
    if size < 1024:
        return f"{size} B"
    value = float(size)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PiB"


@dataclass(frozen=True, slots=True)
class StoreEntry:
    """One GGUF file as the store knows it (name, size, sha, quant, family)."""

    name: str
    path: Path
    size: int
    quant: str
    family: str
    sha: str | None

    @classmethod
    def for_path(cls, path: Path) -> StoreEntry:
        """Describe a file by its name alone — no hashing, sha stays None."""
        stem = path.stem
        match = _QUANT_SUFFIX.search(stem)
        family = stem[: match.start()] if match else stem
        quant = match.group(1) if match else ""
        size = path.stat().st_size if path.is_file() else 0
        return cls(name=stem, path=path, size=size, quant=quant, family=family, sha=None)


class Store:
    """GGUFs under the configured search paths, deduped by resolved path."""

    def __init__(self, search_paths: Sequence[str]) -> None:
        self._paths = tuple(Path(raw).expanduser() for raw in search_paths)

    def scan(self) -> list[StoreEntry]:
        """Every model found, name-sorted; cached shas trusted only while size matches.

        Recursive: the hf provider preserves repo subpaths, so downloads land
        nested (e.g. store/tinyllamas/...) with their index beside them — a
        top-level glob would leave them invisible.
        """
        entries: dict[Path, StoreEntry] = {}
        indexes: dict[Path, dict[str, dict[str, object]]] = {}
        for directory in self._paths:
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*.gguf")):
                key = path.resolve()
                if key in entries:
                    continue
                if path.parent not in indexes:
                    indexes[path.parent] = _read_index(path.parent)
                index = indexes[path.parent]
                entry = StoreEntry.for_path(path)
                cached = index.get(path.name)
                cached_sha = cached.get("sha") if cached is not None else None
                if (
                    cached is not None
                    and isinstance(cached_sha, str)
                    and cached.get("size") == entry.size
                ):
                    entry = replace(entry, sha=cached_sha)
                entries[key] = entry
        return sorted(entries.values(), key=lambda item: (item.name, str(item.path)))

    def register(self, path: Path) -> StoreEntry:
        """Checksum the file and upsert it into the index beside it."""
        path = Path(path).expanduser()
        entry = replace(StoreEntry.for_path(path), sha=_sha256(path))
        index = _read_index(path.parent)
        index[path.name] = {
            "size": entry.size,
            "sha": entry.sha,
            "quant": entry.quant,
            "family": entry.family,
        }
        _write_index(path.parent, index)
        return entry


def _read_index(directory: Path) -> dict[str, dict[str, object]]:
    """Cached entries for one directory; a missing or corrupt index is just empty."""
    try:
        payload = json.loads((directory / _INDEX_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    valid: dict[str, dict[str, object]] = {}
    for name, value in entries.items():
        if isinstance(name, str) and isinstance(value, dict):
            valid[name] = value
    return valid


def _write_index(directory: Path, entries: dict[str, dict[str, object]]) -> None:
    payload = {"version": _INDEX_VERSION, "entries": entries}
    (directory / _INDEX_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
