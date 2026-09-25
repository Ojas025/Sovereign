"""T8.4 — laya checkpoint prefetch against a mocked Hub (PLAN §6.2, §9).

`models download` warms the HF cache with exactly the files laya's own load()
will request (same allow_patterns), so the runtime afterwards can run with the
offline guard on. The subprocesses prove both halves: fetch from an endpoint,
then answer "cached" from a warm cache with the endpoint dead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from fixtures.fake_hf_server import FakeHub, free_port, serve
from workbench.routing.laya_router import checkpoint_location

_REPO = "convaiinnovations/laya"
_FILES = {
    "rl_agent_config.json": b"{}",
    "model.safetensors": b"weights",
    "tokenizer/tokenizer.json": b"tok",
    "encoder/config.json": b"enc",
    "README.md": b"docs",
    "multilingual/rl_agent_config.json": b"{}",
    "multilingual/model.safetensors": b"weights2",
}


def _prefetch(
    checkpoint: str,
    endpoint: str,
    hf_home: Path,
    *,
    offline: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run prefetch in a fresh interpreter: HF_HOME/HF_ENDPOINT must exist before
    huggingface_hub imports, which freezes both."""
    snippet = (
        "import sys\n"
        "from workbench.models.download import prefetch_laya_checkpoint\n"
        "print(prefetch_laya_checkpoint(sys.argv[1]))\n"
    )
    env = dict(os.environ)
    env.update(
        {
            "HF_ENDPOINT": endpoint,
            "HF_HOME": str(hf_home),
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_VERSION_CHECK": "1",
        }
    )
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    return subprocess.run(
        [sys.executable, "-c", snippet, checkpoint],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def _last_line(result: subprocess.CompletedProcess[str]) -> str:
    lines = result.stdout.splitlines()
    return lines[-1].strip() if lines else ""


def _cached_files(hf_home: Path) -> set[str]:
    """Repo-relative paths present under the local snapshot cache."""
    root = hf_home / "hub" / f"models--{_REPO.replace('/', '--')}" / "snapshots"
    return {
        str(path.relative_to(snap))
        for snap in root.glob("*")
        for path in snap.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("laya", ("convaiinnovations/laya", None)),
        ("laya-multilingual", ("convaiinnovations/laya", "multilingual")),
        ("laya-typed-decisions", ("convaiinnovations/laya", "typed-decisions")),
        ("acme/custom", ("acme/custom", None)),
    ],
)
def test_checkpoint_location_maps_named_checkpoints(
    checkpoint: str, expected: tuple[str, str | None]
) -> None:
    assert checkpoint_location(checkpoint) == expected


def test_prefetch_caches_exactly_the_files_the_runtime_will_request(tmp_path: Path) -> None:
    hub = FakeHub()
    hub.add_repo(_REPO, _FILES)
    with serve(hub) as endpoint:
        result = _prefetch("laya", endpoint, tmp_path / "hf")

    assert result.returncode == 0, result.stderr
    assert _last_line(result) == "downloaded"
    files = _cached_files(tmp_path / "hf")
    assert {
        "rl_agent_config.json",
        "model.safetensors",
        "tokenizer/tokenizer.json",
        "encoder/config.json",
    } <= files
    assert "README.md" not in files
    assert not any(name.startswith("multilingual/") for name in files)
    assert not any("README" in request for request in hub.requests)


def test_subfolder_checkpoint_prefetches_only_its_subfolder(tmp_path: Path) -> None:
    hub = FakeHub()
    hub.add_repo(_REPO, _FILES)
    with serve(hub) as endpoint:
        result = _prefetch("laya-multilingual", endpoint, tmp_path / "hf")

    assert result.returncode == 0, result.stderr
    files = _cached_files(tmp_path / "hf")
    assert "multilingual/rl_agent_config.json" in files
    assert "multilingual/model.safetensors" in files
    assert "rl_agent_config.json" not in files
    assert "README.md" not in files


def test_prefetch_answers_cached_from_a_warm_cache_with_the_endpoint_dead(
    tmp_path: Path,
) -> None:
    hub = FakeHub()
    hub.add_repo(_REPO, _FILES)
    with serve(hub) as endpoint:
        first = _prefetch("laya", endpoint, tmp_path / "hf")
    assert first.returncode == 0, first.stderr
    assert _last_line(first) == "downloaded"

    dead = f"http://127.0.0.1:{free_port()}"
    second = _prefetch("laya", dead, tmp_path / "hf", offline=True)

    assert second.returncode == 0, second.stderr
    assert _last_line(second) == "cached"


def test_prefetch_without_cache_or_network_fails_loudly(tmp_path: Path) -> None:
    dead = f"http://127.0.0.1:{free_port()}"

    result = _prefetch("laya", dead, tmp_path / "cold")

    assert result.returncode != 0
    assert f"cannot prefetch laya checkpoint {_REPO}" in result.stderr
