"""Errors the download path raises; the CLI turns these into a clean exit."""

from __future__ import annotations


class DownloadError(Exception):
    """A download failed after its provider tried; the message is user-facing."""
