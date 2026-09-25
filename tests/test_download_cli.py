"""T8.4 — `workbench models download`: provider dispatch, store registration,
and the one command exempt from the offline guard (PLAN §6.2, constraint 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixtures.cli_runner import run_cli
from workbench.config import Config, load_config
from workbench.core.bootstrap import build_registry
from workbench.core.protocols import ModelProvider
from workbench.core.registry import Registry, RegistryError
from workbench.models import DownloadError
from workbench.models import download as download_module


class _FakeProvider:
    """Stands in for the network: records its calls and drops a tiny GGUF."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def download(self, spec: str, destination: Path) -> Path:
        self.calls.append((spec, destination))
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "tiny-model-Q4_K_M.gguf"
        path.write_bytes(b"gguf-bytes")
        return path


def _config(tmp_path: Path, models_toml: str) -> Config:
    path = tmp_path / "cfg.toml"
    path.write_text(models_toml, encoding="utf-8")
    return load_config(config_path=path, project_dir=tmp_path)


def _registry_with(provider: object, name: str = "huggingface") -> Registry:
    registry = Registry()
    registry.register("providers", name, provider)
    return registry


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("hf:org/repo", ("huggingface", "org/repo")),
        ("ollama:qwen3:2b", ("ollama", "qwen3:2b")),
        ("org/model", ("huggingface", "org/model")),
        ("qwen3:2b", ("ollama", "qwen3:2b")),
        ("qwen3", ("ollama", "qwen3")),
    ],
)
def test_select_provider_prefers_an_explicit_prefix(spec: str, expected: tuple[str, str]) -> None:
    assert download_module.select_provider(spec) == expected


def test_run_download_fetches_registers_and_prefetches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    config = _config(tmp_path, f'[models]\nsearch_paths = ["{store}"]\n')
    provider = _FakeProvider()
    registry = _registry_with(provider)
    seen: list[str] = []

    def _fake_prefetch(ck: str) -> str:
        seen.append(ck)
        return "cached"

    monkeypatch.setattr(download_module, "prefetch_laya_checkpoint", _fake_prefetch)

    report = download_module.run_download(config, "hf:mock/model", registry=registry)

    assert provider.calls == [("mock/model", store)]
    assert report.path == store / "tiny-model-Q4_K_M.gguf"
    assert report.laya == "cached"
    assert seen == ["laya"]  # the default routing.checkpoint was warmed
    assert report.entry.sha is not None and len(report.entry.sha) == 64
    index = json.loads((store / "index.json").read_text(encoding="utf-8"))
    assert index["entries"]["tiny-model-Q4_K_M.gguf"]["sha"] == report.entry.sha


def test_run_download_discloses_the_file_when_the_prefetch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, f'[models]\nsearch_paths = ["{tmp_path / "store"}"]\n')
    registry = _registry_with(_FakeProvider())

    def _boom(checkpoint: str) -> str:
        raise DownloadError("cannot prefetch laya checkpoint convaiinnovations/laya: no route")

    monkeypatch.setattr(download_module, "prefetch_laya_checkpoint", _boom)

    with pytest.raises(DownloadError) as excinfo:
        download_module.run_download(config, "hf:mock/model", registry=registry)

    assert "tiny-model-Q4_K_M.gguf" in str(excinfo.value)
    assert "cannot prefetch" in str(excinfo.value)


def test_run_download_lets_an_unknown_provider_propagate(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, f'[models]\nsearch_paths = ["{tmp_path / "store"}"]\n')
    provider = _FakeProvider()

    with pytest.raises(RegistryError):
        download_module.run_download(
            config, "hf:mock/model", registry=_registry_with(provider, "ollama")
        )

    assert provider.calls == []  # nothing was fetched


def test_run_download_rejects_a_registry_object_that_is_not_a_provider(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, f'[models]\nsearch_paths = ["{tmp_path / "store"}"]\n')

    with pytest.raises(DownloadError, match="ModelProvider"):
        download_module.run_download(
            config, "hf:mock/model", registry=_registry_with(object())
        )


def test_run_download_requires_a_search_path(tmp_path: Path) -> None:
    config = _config(tmp_path, '[models]\nsearch_paths = []\n')
    provider = _FakeProvider()

    with pytest.raises(DownloadError, match="search_paths"):
        download_module.run_download(config, "hf:mock/model", registry=_registry_with(provider))

    assert provider.calls == []


def test_build_registry_registers_the_builtin_providers() -> None:
    registry = build_registry(load_config())

    assert registry.entries("providers") == ["huggingface", "ollama"]
    for name in ("huggingface", "ollama"):
        assert isinstance(registry.get("providers", name), ModelProvider)


# --- cli dispatch (child process: the guard is process-wide) -----------------

_OK_SETUP = """
from pathlib import Path

from workbench.models.download import DownloadReport
from workbench.models.store import StoreEntry
import workbench.cli as _cli


def _fake_run_download(config, spec, *, registry=None):
    target = Path("/x/tiny-model.gguf")
    return DownloadReport(
        path=target,
        entry=StoreEntry(
            name="tiny-model",
            path=target,
            size=1024,
            quant="Q4_K_M",
            family="tiny-model",
            sha="a" * 64,
        ),
        laya="cached",
    )


_cli.run_download = _fake_run_download
"""

_FAIL_SETUP = """
from workbench.models.errors import DownloadError
import workbench.cli as _cli


def _fake_run_download(config, spec, *, registry=None):
    raise DownloadError("boom")


_cli.run_download = _fake_run_download
"""


def test_models_download_prints_the_report_and_skips_the_offline_guard(
    tmp_path: Path,
) -> None:
    state, result = run_cli(
        ["models", "download", "hf:org/repo"], setup=_OK_SETUP, cwd=tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert state["code"] == 0
    expected = f"downloaded /x/tiny-model.gguf (1.0 KiB, sha {'a' * 12})"
    assert expected in result.stdout
    assert "laya checkpoint: cached" in result.stdout
    assert state["hf"] is None, "download is the one command that must stay online"
    assert not state["hub"]


def test_models_download_reports_a_failure_with_exit_1(tmp_path: Path) -> None:
    state, result = run_cli(
        ["models", "download", "hf:org/repo"], setup=_FAIL_SETUP, cwd=tmp_path
    )

    assert state["code"] == 1
    assert "workbench: boom" in result.stderr
    assert state["hf"] is None
