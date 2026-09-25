"""Core wiring: build_registry ships the four tools + configured sandbox; headless policy."""

import importlib.metadata
from pathlib import Path

import pytest

from workbench.config import ConfigError, load_config
from workbench.core.bootstrap import build_registry, resolve_router
from workbench.core.protocols import Router, Sandbox, Tool
from workbench.core.registry import RegistryError
from workbench.sandbox.bwrap import BwrapSandbox
from workbench.tools import headless_confirm


def config(tmp_path: Path, **sandbox_overrides: str):
    project = tmp_path / "project"
    project.mkdir()
    if sandbox_overrides:
        body = "[sandbox]\n" + "".join(f"{k} = '{v}'\n" for k, v in sandbox_overrides.items())
        (project / ".workbench.toml").write_text(body, encoding="utf-8")
    return load_config(home=tmp_path / "home", project_dir=project)


class TestBuildRegistry:
    def test_registers_the_four_core_tools(self, tmp_path: Path) -> None:
        registry = build_registry(config(tmp_path))

        assert registry.entries("tools") == ["bash", "edit", "read", "write"]

    def test_registered_tools_satisfy_the_tool_protocol(self, tmp_path: Path) -> None:
        registry = build_registry(config(tmp_path))

        for name in registry.entries("tools"):
            tool = registry.get("tools", name)
            assert isinstance(tool, Tool), name

    def test_registers_configured_sandbox_backend(self, tmp_path: Path) -> None:
        registry = build_registry(config(tmp_path))

        sandbox = registry.get("sandboxes", "bwrap")
        assert isinstance(sandbox, BwrapSandbox)
        assert isinstance(sandbox, Sandbox)

    def test_bash_tool_is_wired_to_the_registered_sandbox(self, tmp_path: Path) -> None:
        registry = build_registry(config(tmp_path))

        bash = registry.get("tools", "bash")
        assert bash._sandbox is registry.get("sandboxes", "bwrap")  # noqa: SLF001

    def test_unknown_backend_raises_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="docker"):
            build_registry(config(tmp_path, backend="docker"))

    def test_entry_point_plugin_tools_are_loaded_alongside_core(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeEntryPoint:
            name = "frobnicate"

            def load(self) -> object:
                return object()

        def fake_entry_points(*, group: str) -> list[object]:
            return [FakeEntryPoint()] if group == "workbench.tools" else []

        monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

        registry = build_registry(config(tmp_path))

        assert "frobnicate" in registry.entries("tools")

    def test_plugin_cannot_shadow_a_core_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeEntryPoint:
            name = "read"

            def load(self) -> object:
                return object()

        def fake_entry_points(*, group: str) -> list[object]:
            return [FakeEntryPoint()] if group == "workbench.tools" else []

        monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

        with pytest.raises(RegistryError, match="read"):
            build_registry(config(tmp_path))

    def test_loads_plugins_from_user_plugins_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_home = tmp_path / "user_home"
        plugins_dir = user_home / ".config" / "workbench" / "plugins"
        plugins_dir.mkdir(parents=True)
        (plugins_dir / "custom.py").write_text(
            "def register(registry):\n"
            "    registry.register('tools', 'custom_tool', object())\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(user_home))

        registry = build_registry(config(tmp_path))

        assert "custom_tool" in registry.entries("tools")


class TestHeadlessConfirmation:
    async def test_declines_every_prompt(self) -> None:
        confirm = headless_confirm()

        assert await confirm("run 'rm -rf /' ?") is False
        assert await confirm("anything") is False

    async def test_logs_the_declined_prompt(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="workbench.tools"):
            await headless_confirm()("dangerous command")

        assert any("dangerous command" in record.message for record in caplog.records)


class TestRouterWiring:
    def test_registers_both_routing_backends(self, tmp_path: Path) -> None:
        registry = build_registry(config(tmp_path))

        assert registry.entries("routers") == ["heuristic", "laya"]

    def test_resolve_router_returns_the_configured_backend(self, tmp_path: Path) -> None:
        cfg = config(tmp_path)

        router = resolve_router(build_registry(cfg), cfg)

        assert isinstance(router, Router)

    def test_resolve_router_unknown_backend_raises_config_error(
        self, tmp_path: Path
    ) -> None:
        project = tmp_path / "routed"
        project.mkdir()
        (project / ".workbench.toml").write_text(
            '[routing]\nbackend = "routed-llm"\n', encoding="utf-8"
        )
        cfg = load_config(home=tmp_path / "home", project_dir=project)

        with pytest.raises(ConfigError, match="routed-llm"):
            resolve_router(build_registry(cfg), cfg)
