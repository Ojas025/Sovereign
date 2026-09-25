"""Confirmation policies for guarded tool operations (``ConfirmFn`` implementations)."""

from __future__ import annotations

from workbench.core.protocols import ConfirmFn
from workbench.logging_setup import get_logger

logger = get_logger("tools")


def headless_confirm() -> ConfirmFn:
    """Decline every prompt: ``-p``/``--json`` runs have no one to ask.

    Unattended autonomy is opt-in via ``sandbox.mode = "auto"`` (deny rules still
    apply); by default a flagged command stays blocked instead of silently running.
    """

    async def confirm(prompt: str) -> bool:
        logger.info("confirmation declined (headless): %s", prompt)
        return False

    return confirm
