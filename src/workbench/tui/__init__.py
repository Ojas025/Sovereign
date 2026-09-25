"""TUI surface built on ``agentic-tui`` (import name ``agentui``, ruling M6.1)."""

from workbench.tui.app import run_tui
from workbench.tui.patch import patch_live_region
from workbench.tui.state import TuiState
from workbench.tui.transcript import TranscriptController

patch_live_region()

__all__ = ["TranscriptController", "TuiState", "patch_live_region", "run_tui"]
