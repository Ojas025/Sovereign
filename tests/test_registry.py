"""Plugin registry: group registration, entry points, and local plugin directory loading."""

import importlib.metadata
from pathlib import Path

import pytest

from workbench.core.registry import (
    GROUPS,
    Registry,
    RegistryError,
)


class TestRegistration:
    def test_register_and_get_roundtrip(self) -> None:
        registry = Registry()
        tool = object()

        registry.register("tools", "read", tool)

        assert registry.get("tools", "read") is tool

    def test_unknown_group_is_rejected(self) -> None:
        registry = Registry()

        with pytest.raises(RegistryError, match="widgets"):
            registry.register("widgets", "x", object())

    def test_duplicate_name_is_rejected(self) -> None:
        registry = Registry()
        registry.register("tools", "read", object())

        with pytest.raises(RegistryError, match="read"):
            registry.register("tools", "read", object())

    def test_missing_entry_raises(self) -> None:
        registry = Registry()

        with pytest.raises(RegistryError, match="ghost"):
            registry.get("tools", "ghost")

    def test_default_groups_match_the_plan(self) -> None:
        assert set(GROUPS) == {"tools", "routers", "providers", "sandboxes", "commands"}

    def test_entries_lists_group_contents(self) -> None:
        registry = Registry()
        registry.register("routers", "laya", object())
        registry.register("routers", "heuristic", object())

        assert sorted(registry.entries("routers")) == ["heuristic", "laya"]


class TestEntryPoints:
    def test_discovers_objects_from_entry_points(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loaded = object()

        class FakeEntryPoint:
            name = "remote"

            def load(self) -> object:
                return loaded

        def fake_entry_points(*, group: str) -> list[object]:
            return [FakeEntryPoint()] if group == "workbench.routers" else []

        monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

        registry = Registry()
        registry.load_entry_points()

        assert registry.get("routers", "remote") is loaded

    def test_entry_point_group_maps_to_registry_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen_groups: list[str] = []

        class FakeEntryPoint:
            name = "x"

            def load(self) -> object:
                return object()

        def fake_entry_points(*, group: str) -> list[object]:
            seen_groups.append(group)
            return []

        monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

        Registry().load_entry_points()

        assert seen_groups == [f"workbench.{g}" for g in GROUPS]


class TestLocalPlugins:
    def test_loads_plugin_module_calling_register(self, tmp_path: Path) -> None:
        plugin = tmp_path / "my_plugin.py"
        plugin.write_text(
            "def register(registry):\n"
            "    registry.register('tools', 'frobnicate', object())\n",
            encoding="utf-8",
        )

        registry = Registry()
        registry.load_directory(tmp_path)

        assert registry.get("tools", "frobnicate")

    def test_ignores_subdirectories_and_non_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("not a plugin")
        (tmp_path / "package").mkdir()

        Registry().load_directory(tmp_path)  # must not raise

    def test_broken_plugin_reports_its_path(self, tmp_path: Path) -> None:
        (tmp_path / "broken.py").write_text("def register(registry)\n    pass\n", encoding="utf-8")

        with pytest.raises(RegistryError, match="broken.py"):
            Registry().load_directory(tmp_path)
