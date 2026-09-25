"""Example TUI slash command plugin proving the ``workbench.commands`` extension point (PLAN §3).

Defines an example ``/echo`` slash command for the interactive TUI.
"""

from __future__ import annotations

from typing import Any

from workbench.core.registry import Registry


async def echo_command(ui: Any, args: str) -> None:
    """Print the arguments back to the TUI."""
    text = args.strip() or "(empty)"
    ui.print(f"[bold cyan]echo:[/bold cyan] {text}")


def register(registry: Registry) -> None:
    """Registration hook when loaded from ~/.config/workbench/plugins/."""
    registry.register("commands", "echo", echo_command)
