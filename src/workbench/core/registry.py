"""Plugin registry: entry points plus an explicit project plugins directory."""

import importlib.util
import logging
from importlib import metadata
from pathlib import Path

logger = logging.getLogger("workbench.registry")

GROUPS = ("tools", "routers", "providers", "sandboxes", "commands")


class RegistryError(Exception):
    """Unknown group, duplicate registration, or a plugin failed to load."""


class Registry:
    """Namespaced name -> object store for all pluggable components."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, object]] = {group: {} for group in GROUPS}

    def register(self, group: str, name: str, obj: object) -> None:
        if group not in self._entries:
            raise RegistryError(f"unknown plugin group {group!r} (known: {', '.join(GROUPS)})")
        if name in self._entries[group]:
            raise RegistryError(f"{group} plugin {name!r} already registered")
        self._entries[group][name] = obj

    def get(self, group: str, name: str) -> object:
        try:
            return self._entries[group][name]
        except KeyError as exc:
            raise RegistryError(f"no {group} plugin named {name!r}") from exc

    def entries(self, group: str) -> list[str]:
        if group not in self._entries:
            raise RegistryError(f"unknown plugin group {group!r}")
        return sorted(self._entries[group])

    def load_entry_points(self) -> None:
        """Load installed ``workbench.<group>`` entry points in group order."""
        for group in GROUPS:
            for entry_point in metadata.entry_points(group=f"workbench.{group}"):
                self.register(group, entry_point.name, entry_point.load())

    def load_directory(self, path: Path) -> None:
        """Import every ``*.py`` file and call its optional ``register(registry)``."""
        if not path.is_dir():
            return
        for file in sorted(path.glob("*.py")):
            if not file.is_file():
                continue
            spec = importlib.util.spec_from_file_location(f"workbench_plugin_{file.stem}", file)
            if spec is None or spec.loader is None:
                raise RegistryError(f"cannot load plugin {file}")
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                raise RegistryError(f"failed to load plugin {file}: {exc}") from exc
            register = getattr(module, "register", None)
            if callable(register):
                register(self)
                logger.debug("loaded plugin %s from %s", file.stem, file)
