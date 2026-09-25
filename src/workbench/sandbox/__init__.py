"""Sandbox backends (plugin group ``workbench.sandboxes``; core ships bwrap)."""

from workbench.sandbox.bwrap import BwrapSandbox

__all__ = ["BwrapSandbox"]
