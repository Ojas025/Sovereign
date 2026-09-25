"""Hugging Face provider — the only workbench code that talks to the Hub (§6.2).

Transfer itself is delegated to huggingface_hub's snapshot_download (its
Range/etag logic is the resume story), constrained by a *.gguf filter so an
explicit download can never pull a file the store would not list.
"""

from __future__ import annotations

from pathlib import Path

from workbench.models.errors import DownloadError

_GGUF = ".gguf"


def parse_spec(spec: str) -> tuple[str, list[str]]:
    """Split `repo-id[/path/file.gguf]` into repo id + allow_patterns.

    A bare repo-id maps to every `*.gguf` (plan §6.2); pointing at a file
    inside the repo narrows the fetch to that file. Specs that can never
    yield a GGUF are rejected before any network is touched.
    """
    parts = spec.split("/")
    if len(parts) < 2 or any(not part for part in parts):
        raise DownloadError(
            f"invalid huggingface spec {spec!r}: expected repo-id or repo-id/path/file.gguf"
        )
    if len(parts) == 2:
        return spec, ["*.gguf"]
    if not spec.lower().endswith(_GGUF):
        raise DownloadError(f"invalid huggingface spec {spec!r}: file path must end in .gguf")
    return "/".join(parts[:2]), ["/".join(parts[2:])]


class HuggingFaceProvider:
    """snapshot_download behind the M1 ModelProvider contract."""

    def download(self, spec: str, destination: Path) -> Path:
        repo_id, patterns = parse_spec(spec)
        destination.mkdir(parents=True, exist_ok=True)
        before = set(destination.rglob("*.gguf"))
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                allow_patterns=patterns,
                local_dir=str(destination),
            )
        except Exception as exc:  # the Hub client raises a zoo of types
            raise DownloadError(f"downloading {spec} failed: {exc}") from exc
        # Diff instead of listing: a repo without GGUFs must fail even when the
        # destination already holds other models.
        new_files = sorted(
            path for path in set(destination.rglob("*.gguf")) - before if path.is_file()
        )
        if not new_files:
            raise DownloadError(f"{repo_id} contains no .gguf files (patterns: {patterns})")
        return new_files[0]
