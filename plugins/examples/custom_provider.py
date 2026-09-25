"""Example download provider plugin proving the ``workbench.providers`` extension point (PLAN §3).

Defines an example ``localfile`` provider that copies or links existing GGUF files
from an arbitrary local directory into the store.
Usage:
    workbench models download localfile:/path/to/model.gguf
"""

from __future__ import annotations

import shutil
from pathlib import Path

from workbench.core.registry import Registry
from workbench.models.errors import DownloadError


class LocalFileProvider:
    """ModelProvider that copies local files into the store."""

    def download(self, spec: str, destination: Path) -> Path:
        source = Path(spec).expanduser().resolve()
        if not source.is_file():
            raise DownloadError(f"local source file not found: {source}")
        if not source.name.endswith(".gguf"):
            raise DownloadError(f"file must be a .gguf file: {source.name}")

        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source.name
        if target.exists() and target.resolve() == source:
            return target
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            raise DownloadError(f"cannot copy {source} to {target}: {exc}") from exc
        return target


def register(registry: Registry) -> None:
    """Registration hook when loaded from ~/.config/workbench/plugins/."""
    registry.register("providers", "localfile", LocalFileProvider())
