"""T8.4 — process-wide offline guard (PLAN constraint 1: runtime never reaches the Hub).

The child-process cases assert both the env flags AND huggingface_hub's frozen
constant: the guard only works if it runs before the hub imports, and the
constant is what pins that ordering.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fixtures.cli_runner import run_cli
from workbench.models.offline import ensure_offline


def test_ensure_offline_sets_both_hub_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    ensure_offline()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_run_commands_force_offline_before_the_hub_can_import(tmp_path: Path) -> None:
    state, result = run_cli(["models", "list"], cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert state["code"] == 0
    assert state["hf"] == "1"
    assert state["tf"] == "1"
    assert state["hub"], "huggingface_hub must have frozen HF_HUB_OFFLINE as on"
