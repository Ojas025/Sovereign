"""T8.3 — ollama shim: pull via the real CLI contract, resolve the blob locally (§6.2).

The fake `ollama` script plays the real one's contract: `pull <spec>`
materializes a manifest and a content-addressed blob under OLLAMA_MODELS,
which is all the provider is allowed to depend on.
"""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import pytest

from workbench.models.errors import DownloadError
from workbench.models.ollama import OllamaProvider

_BLOB = b"ollama-gguf-bytes"
_BLOB_SHA = hashlib.sha256(_BLOB).hexdigest()

_FAKE_OLLAMA = '''#!/usr/bin/env python3
"""Fake ollama CLI: pull writes a manifest + blob under OLLAMA_MODELS."""
import hashlib, json, os, sys
from pathlib import Path

if os.environ.get("FAKE_OLLAMA_FAIL") == "1":
    sys.stderr.write("pull failed: model not found\\n")
    raise SystemExit(1)

if len(sys.argv) < 3 or sys.argv[1] != "pull":
    sys.stderr.write("usage: ollama pull <name[:tag]>\\n")
    raise SystemExit(2)

spec = sys.argv[2]
name, _, tag = spec.partition(":")
tag = tag or "latest"
blob = os.environ.get("FAKE_OLLAMA_BLOB", "ollama-gguf-bytes").encode()
digest = "sha256:" + hashlib.sha256(blob).hexdigest()

root = Path(os.environ["OLLAMA_MODELS"])
(root / "blobs").mkdir(parents=True, exist_ok=True)
(root / "blobs" / digest.replace(":", "-")).write_bytes(blob)

layers = [{"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:0" * 8}]
if os.environ.get("FAKE_OLLAMA_NO_MODEL_LAYER") != "1":
    layers.append({"mediaType": "application/vnd.ollama.image.model", "digest": digest})

manifest = {"schemaVersion": 2, "layers": layers}
if os.environ.get("FAKE_OLLAMA_NO_MANIFEST") != "1":
    # real layout: manifests/registry.ollama.ai/<namespace>/<name>/<tag>
    parts = name.split("/") if "/" in name else ["library", name]
    manifest_dir = root / "manifests" / "registry.ollama.ai"
    for part in parts:
        manifest_dir = manifest_dir / part
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / tag).write_text(json.dumps(manifest), encoding="utf-8")

sys.stdout.write(f"pulling {spec}...\\n")
'''


@pytest.fixture
def ollama_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(bin_dir, models_root): fake ollama on PATH, isolated OLLAMA_MODELS."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ollama"
    script.write_text(_FAKE_OLLAMA, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    models_root = tmp_path / "ollama-models"
    monkeypatch.setenv("OLLAMA_MODELS", str(models_root))
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    return bin_dir, models_root


def test_pull_symlinks_the_model_layer_blob_into_the_store(
    ollama_env: tuple[Path, Path], tmp_path: Path
) -> None:
    destination = tmp_path / "store"

    result = OllamaProvider().download("qwen3:2b", destination)

    assert result.name == "qwen3-2b.gguf"
    assert result.is_symlink()
    blob = ollama_env[1] / "blobs" / f"sha256-{_BLOB_SHA}"
    assert result.resolve() == blob.resolve()
    assert result.read_bytes() == _BLOB


def test_spec_without_a_tag_defaults_to_latest(
    ollama_env: tuple[Path, Path], tmp_path: Path
) -> None:
    result = OllamaProvider().download("qwen3", tmp_path / "store")

    assert result.name == "qwen3-latest.gguf"
    root = ollama_env[1] / "manifests" / "registry.ollama.ai"
    assert (root / "library" / "qwen3" / "latest").is_file()


def test_repull_of_the_same_blob_is_idempotent(
    ollama_env: tuple[Path, Path], tmp_path: Path
) -> None:
    provider = OllamaProvider()
    first = provider.download("qwen3", tmp_path / "store")
    second = provider.download("qwen3", tmp_path / "store")

    assert first == second
    assert second.is_symlink()


def test_failed_pull_surfaces_stderr_as_a_download_error(
    ollama_env: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_OLLAMA_FAIL", "1")

    with pytest.raises(DownloadError, match="pull failed: model not found"):
        OllamaProvider().download("nope", tmp_path / "store")


def test_missing_manifest_raises_a_download_error(
    ollama_env: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_OLLAMA_NO_MANIFEST", "1")

    with pytest.raises(DownloadError, match="manifest"):
        OllamaProvider().download("qwen3", tmp_path / "store")


def test_manifest_without_a_model_layer_raises(
    ollama_env: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_OLLAMA_NO_MODEL_LAYER", "1")

    with pytest.raises(DownloadError, match="model layer"):
        OllamaProvider().download("qwen3", tmp_path / "store")


def test_missing_ollama_binary_is_a_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(DownloadError, match="ollama"):
        OllamaProvider().download("qwen3", tmp_path / "store")


def test_existing_regular_file_at_the_target_is_never_clobbered(
    ollama_env: tuple[Path, Path], tmp_path: Path
) -> None:
    destination = tmp_path / "store"
    destination.mkdir()
    victim = destination / "qwen3-latest.gguf"
    victim.write_bytes(b"precious")

    with pytest.raises(DownloadError, match="refusing to overwrite"):
        OllamaProvider().download("qwen3", destination)

    assert victim.read_bytes() == b"precious"
