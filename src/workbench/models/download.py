"""Download command orchestration — the only module allowed to touch the network (§6.2).

Providers do the fetching (huggingface / ollama shims behind the ModelProvider
protocol); this picks the provider from the spec's shape, fetches into the
store's first search path, checksums the file into the index, then prefetches
the laya checkpoint so everything afterwards can run with the offline guard on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from workbench.config import Config
from workbench.core.bootstrap import build_registry
from workbench.core.protocols import ModelProvider
from workbench.core.registry import Registry
from workbench.models.errors import DownloadError
from workbench.models.store import Store, StoreEntry
from workbench.routing.laya_router import checkpoint_location

# `name:` prefixes that override the slash heuristic (anything else is a spec as-is).
_PROVIDER_PREFIXES = {"hf": "huggingface", "ollama": "ollama"}
# Keep in sync with laya's snapshot request (laya/agent.py allow_patterns):
# prefetching any more would waste bandwidth, any less would break offline load.
_LAYA_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
# The two files laya refuses to load without — cache counts as ready only with them.
_LAYA_MARKERS = ("rl_agent_config.json", "model.safetensors")


def select_provider(spec: str) -> tuple[str, str]:
    """Map a spec to (provider, spec for it): an explicit prefix wins, else a slash means hub."""
    head, colon, rest = spec.partition(":")
    if colon and head in _PROVIDER_PREFIXES:
        return _PROVIDER_PREFIXES[head], rest
    return ("huggingface", spec) if "/" in spec else ("ollama", spec)


@dataclass(frozen=True, slots=True)
class DownloadReport:
    """What `models download` produced: the registered file plus the prefetch status."""

    path: Path
    entry: StoreEntry
    laya: str


def run_download(
    config: Config, spec: str, *, registry: Registry | None = None
) -> DownloadReport:
    """Fetch spec into the store, checksum it into the index, warm the laya checkpoint."""
    if not config.models.search_paths:
        raise DownloadError("models.search_paths is empty — nowhere to download into")
    provider_name, provider_spec = select_provider(spec)
    registry = registry or build_registry(config)
    provider = registry.get("providers", provider_name)  # RegistryError: unknown provider
    if not isinstance(provider, ModelProvider):
        raise DownloadError(f"provider {provider_name!r} does not implement ModelProvider")
    destination = Path(config.models.search_paths[0]).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    path = provider.download(provider_spec, destination)
    entry = Store(config.models.search_paths).register(path)
    try:
        laya_status = prefetch_laya_checkpoint(config.routing.checkpoint)
    except DownloadError as exc:
        raise DownloadError(f"downloaded {path} but {exc}") from exc
    return DownloadReport(path=path, entry=entry, laya=laya_status)


def prefetch_laya_checkpoint(checkpoint: str) -> str:
    """Warm the HF cache with exactly what laya's load() will request.

    Returns "cached" when a complete checkpoint was already local, "downloaded"
    after fetching it; a fetch failure raises DownloadError (loudly — the
    runtime would only notice once the offline guard blocks it).
    """
    from huggingface_hub import snapshot_download  # lazy: only this path pays for the import

    repo, subfolder = checkpoint_location(checkpoint)
    prefix = f"{subfolder}/" if subfolder else ""
    patterns = [prefix + name for name in _LAYA_FILES]
    try:
        model_dir = snapshot_download(repo_id=repo, allow_patterns=patterns, local_files_only=True)
        if all((Path(model_dir) / f"{prefix}{marker}").is_file() for marker in _LAYA_MARKERS):
            return "cached"
    except Exception:
        pass  # nothing (or not enough) in the cache yet — fetch it for real
    try:
        snapshot_download(repo_id=repo, allow_patterns=patterns)
    except Exception as exc:
        raise DownloadError(f"cannot prefetch laya checkpoint {repo}: {exc}") from exc
    return "downloaded"
