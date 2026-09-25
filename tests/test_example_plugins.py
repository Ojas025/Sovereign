"""Tests for plugins/examples: verify that all sample plugins conform to their protocols."""

from __future__ import annotations

from pathlib import Path

from workbench.core.protocols import ModelProvider, Router, Sandbox, Tool
from workbench.core.registry import Registry

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "plugins" / "examples"


def test_load_all_example_plugins() -> None:
    registry = Registry()
    registry.load_directory(EXAMPLES_DIR)

    # 1. Tool
    assert "calc" in registry.entries("tools")
    calc = registry.get("tools", "calc")
    assert isinstance(calc, Tool)
    assert calc.name == "calc"

    # 2. Router
    assert "keyword" in registry.entries("routers")
    router = registry.get("routers", "keyword")
    assert isinstance(router, Router)

    # 3. Provider
    assert "localfile" in registry.entries("providers")
    provider = registry.get("providers", "localfile")
    assert isinstance(provider, ModelProvider)

    # 4. Sandbox
    assert "passthrough" in registry.entries("sandboxes")
    sandbox = registry.get("sandboxes", "passthrough")
    assert isinstance(sandbox, Sandbox)

    # 5. Command
    assert "echo" in registry.entries("commands")
    command = registry.get("commands", "echo")
    assert callable(command)


async def test_calc_tool_execution(tmp_path: Path) -> None:
    from workbench.core.protocols import ToolContext

    registry = Registry()
    registry.load_directory(EXAMPLES_DIR)
    calc = registry.get("tools", "calc")
    assert isinstance(calc, Tool)

    async def fake_confirm(p: str) -> bool:
        return True

    ctx = ToolContext(workspace_root=tmp_path, confirm=fake_confirm)

    res = await calc.execute({"expression": "10 + 5 * 2"}, ctx)
    assert not res.is_error
    assert res.content == "20"

    res_err = await calc.execute({"expression": "__import__('os')"}, ctx)
    assert res_err.is_error


async def test_keyword_router_classification() -> None:
    registry = Registry()
    registry.load_directory(EXAMPLES_DIR)
    router = registry.get("routers", "keyword")
    assert isinstance(router, Router)

    res = await router.classify("Please plan the project architecture")
    assert res.intent == "planning"
    assert res.needs_tools is True
    assert res.difficulty > 2.0


def test_localfile_provider_import(tmp_path: Path) -> None:
    registry = Registry()
    registry.load_directory(EXAMPLES_DIR)
    provider = registry.get("providers", "localfile")
    assert isinstance(provider, ModelProvider)

    source = tmp_path / "source"
    source.mkdir()
    model = source / "test-model-Q4_0.gguf"
    model.write_bytes(b"dummy-gguf-bytes")

    dest = tmp_path / "store"
    imported = provider.download(str(model), dest)
    assert imported.is_file()
    assert imported.read_bytes() == b"dummy-gguf-bytes"
