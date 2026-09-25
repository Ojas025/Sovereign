"""Ollama shim: `ollama pull` owns the network, we resolve its blob locally (§6.2).

After a pull, the model sits in ollama's content-addressed store as an
unnamed blob referenced by a manifest layer. The provider's whole job is
that resolution — symlink the model layer into our store so `models scan`
sees a real GGUF — never touching the network itself.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from workbench.models.errors import DownloadError

_DEFAULT_TAG = "latest"
_DEFAULT_REGISTRY = "registry.ollama.ai"
_DEFAULT_NAMESPACE = "library"
_MODEL_LAYER = "application/vnd.ollama.image.model"
_BLOBS = "blobs"
_MANIFESTS = "manifests"
_PULL_TAIL_CHARS = 500


def _models_root() -> Path:
    """Ollama's models dir: OLLAMA_MODELS when set, else ~/.ollama/models."""
    return Path(os.environ.get("OLLAMA_MODELS") or "~/.ollama/models").expanduser()


def _manifest_path(root: Path, name: str, tag: str) -> Path:
    """manifests/<registry>/<namespace>/<name>/<tag>; bare names live under `library`."""
    parts = name.split("/")
    if len(parts) == 1:
        parts = [_DEFAULT_REGISTRY, _DEFAULT_NAMESPACE, name]
    elif len(parts) == 2:
        parts = [_DEFAULT_REGISTRY, *parts]
    return root.joinpath(_MANIFESTS, *parts, tag)


class OllamaProvider:
    """ModelProvider over the ollama CLI: pull, then symlink the model layer."""

    def download(self, spec: str, destination: Path) -> Path:
        name, _, raw_tag = spec.partition(":")
        tag = raw_tag or _DEFAULT_TAG
        if not name:
            raise DownloadError(f"invalid ollama spec {spec!r}: expected name[:tag]")

        binary = shutil.which("ollama")
        if binary is None:
            raise DownloadError(f"ollama not found on PATH (cannot pull {spec!r})")
        result = subprocess.run([binary, "pull", spec], capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip()[-_PULL_TAIL_CHARS:]
            raise DownloadError(f"ollama pull {spec!r} failed: {tail}")

        root = _models_root()
        manifest = _manifest_path(root, name, tag)
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DownloadError(
                f"manifest for {spec!r} not found at {manifest} (pull succeeded?)"
            ) from exc

        digest = _model_layer_digest(payload, spec)
        blob = root / _BLOBS / digest.replace(":", "-", 1)
        if not blob.is_file():
            raise DownloadError(f"ollama blob {digest} missing under {root / _BLOBS}")

        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"{name.replace('/', '-')}-{tag}.gguf"
        if target.is_symlink():
            if target.resolve() == blob.resolve():
                return target  # re-pull of the same content: already in the store
            raise DownloadError(f"refusing to overwrite {target} (points elsewhere)")
        if target.exists():
            raise DownloadError(f"refusing to overwrite {target}")
        try:
            target.symlink_to(blob.resolve())
        except OSError:
            # symlink across devices fails — a copy keeps the store usable
            shutil.copy2(blob, target)
        return target


def _model_layer_digest(payload: object, spec: str) -> str:
    """The GGUF digest from the manifest's model layer — the blob's identity."""
    layers = payload.get("layers") if isinstance(payload, dict) else None
    if isinstance(layers, list):
        for layer in layers:
            if (
                isinstance(layer, dict)
                and layer.get("mediaType") == _MODEL_LAYER
                and isinstance(layer.get("digest"), str)
            ):
                return str(layer["digest"])
    raise DownloadError(f"manifest for {spec!r} has no model layer")
