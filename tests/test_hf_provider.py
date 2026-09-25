"""T8.2 — huggingface provider against a mocked Hub endpoint (PLAN §6.2, §9)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from fixtures.fake_hf_server import FakeHub, free_port, serve
from workbench.models.errors import DownloadError
from workbench.models.huggingface import HuggingFaceProvider, parse_spec

_Q4 = b"gguf-q4-bytes"
_Q8 = b"gguf-q8-bytes"


def _download(spec: str, destination: Path, endpoint: str) -> subprocess.CompletedProcess[str]:
    """Run the provider in a fresh interpreter: HF_ENDPOINT must exist before
    huggingface_hub imports, which freezes its endpoint constant."""
    snippet = (
        "import sys\n"
        "from pathlib import Path\n"
        "from workbench.models.huggingface import HuggingFaceProvider\n"
        "print(HuggingFaceProvider().download(sys.argv[1], Path(sys.argv[2])))\n"
    )
    env = {
        **os.environ,
        "HF_ENDPOINT": endpoint,
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_VERSION_CHECK": "1",
    }
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    return subprocess.run(
        [sys.executable, "-c", snippet, spec, str(destination)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("mock/model", ("mock/model", ["*.gguf"])),
        ("mock/model/quant/file.gguf", ("mock/model", ["quant/file.gguf"])),
    ],
)
def test_parse_spec_splits_repo_and_allow_patterns(
    spec: str, expected: tuple[str, list[str]]
) -> None:
    assert parse_spec(spec) == expected


@pytest.mark.parametrize("spec", ["bare", "mock//model", "mock/model/notes.txt"])
def test_parse_spec_rejects_specs_that_cannot_yield_a_gguf(spec: str) -> None:
    with pytest.raises(DownloadError):
        parse_spec(spec)


def test_provider_rejects_a_bad_spec_before_touching_the_network(tmp_path: Path) -> None:
    with pytest.raises(DownloadError):
        HuggingFaceProvider().download("bare", tmp_path / "dest")

    assert not (tmp_path / "dest").exists()


def test_bare_repo_downloads_every_gguf_and_skips_other_files(tmp_path: Path) -> None:
    hub = FakeHub()
    hub.add_repo("mock/model", {"a-Q4_K_M.gguf": _Q4, "b-Q8_0.gguf": _Q8, "README.md": b"docs"})
    with serve(hub) as endpoint:
        result = _download("mock/model", tmp_path / "dest", endpoint)

    assert result.returncode == 0, result.stderr
    dest = tmp_path / "dest"
    assert (dest / "a-Q4_K_M.gguf").read_bytes() == _Q4
    assert (dest / "b-Q8_0.gguf").read_bytes() == _Q8
    assert not (dest / "README.md").exists()
    assert not any("README.md" in request for request in hub.requests)


def test_file_spec_fetches_only_the_named_file(tmp_path: Path) -> None:
    hub = FakeHub()
    hub.add_repo("mock/model", {"keep/small.gguf": _Q8, "other/big-Q4.gguf": _Q4})
    with serve(hub) as endpoint:
        result = _download("mock/model/keep/small.gguf", tmp_path / "dest", endpoint)

    assert result.returncode == 0, result.stderr
    dest = tmp_path / "dest"
    assert (dest / "keep" / "small.gguf").read_bytes() == _Q8
    assert not (dest / "other").exists()
    assert not any("big-Q4" in request for request in hub.requests)


def test_repo_without_gguf_files_fails_with_a_download_error(tmp_path: Path) -> None:
    hub = FakeHub()
    hub.add_repo("mock/empty", {"README.md": b"docs"})
    with serve(hub) as endpoint:
        result = _download("mock/empty", tmp_path / "dest", endpoint)

    assert result.returncode != 0
    assert "no .gguf" in result.stderr


def test_unreachable_endpoint_reports_a_download_error(tmp_path: Path) -> None:
    dead = f"http://127.0.0.1:{free_port()}"
    result = _download("mock/model", tmp_path / "dest", dead)

    assert result.returncode != 0
    assert "downloading mock/model failed" in result.stderr
