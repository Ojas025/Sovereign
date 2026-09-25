"""Bash guardrails: deny rules, risk flags, allowlist — independent of the sandbox (plan §5.3.3).

Three layers of decision, in order:
1. deny rules    — destructive classics (`rm -rf /`, `mkfs`, `dd of=/dev/*`, fork bombs)
                   block unconditionally; no mode or allowlist overrides them.
2. allowlist     — every program in the command is explicitly trusted → run.
3. mode          — ``strict`` denies whatever is not allowlisted; ``auto`` runs
                   everything that survived (1)+(2); ``confirm`` (default) prompts
                   for commands flagged by the risk rules.

The classifier is a heuristic over the command string (shell parsing is out of
scope); the sandbox is the OS-level backstop behind it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["allow", "confirm", "deny"]


@dataclass(frozen=True, slots=True)
class GuardDecision:
    verdict: Verdict
    # stable token for logs/metrics: deny:<rule> | risk:<rule> | allowlisted | auto | default
    reason: str


def _regex_rule(pattern: str) -> Callable[[str], bool]:
    compiled = re.compile(pattern)
    return lambda command: compiled.search(command) is not None


# Command prefixes that merely wrap another program (strip before looking for `rm`).
_WRAPPERS = frozenset({"sudo", "doas", "env", "command", "exec", "nohup", "nice", "time"})

# rm targets that mean "everything", not a project subdirectory.
_ROOT_TARGETS = frozenset({"/", "/*", "~", "~/*", "$HOME", "$HOME/*", ".", ".."})


def _segments(command: str) -> list[str]:
    """Split a command line on shell separators (heuristic, not a parser)."""
    return re.split(r"[;&|]+", command)


def _rm_root(command: str) -> bool:
    for segment in _segments(command):
        tokens = segment.split()
        while tokens and tokens[0] in _WRAPPERS:
            tokens = tokens[1:]
            while tokens and tokens[0].startswith("-"):  # wrapper options, e.g. `sudo -n`
                tokens = tokens[1:]
        if not tokens or tokens[0] != "rm":
            continue
        flags: set[str] = set()
        targets: list[str] = []
        options_done = False
        for token in tokens[1:]:
            if not options_done and token == "--":
                options_done = True
            elif not options_done and token.startswith("-"):
                flags.update(token.lstrip("-"))
            else:
                targets.append(token)
        if {"r", "f"} <= flags and any(_is_root_target(target) for target in targets):
            return True
    return False


def _is_root_target(target: str) -> bool:
    """`/` normalizes to `/` (rstrip would erase it); `~`/`$HOME` keep their shape."""
    normalized = target.rstrip("/") or "/"
    return normalized in _ROOT_TARGETS


_DENY_RULES: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("rm_root", _rm_root),
    ("mkfs", _regex_rule(r"\bmkfs(?:\.[A-Za-z0-9]+)?\b")),
    ("dd_device", _regex_rule(r"\bdd\b[^;&|]*\bof=/dev/")),
    ("fork_bomb", _regex_rule(r":\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:\s*&\s*\}\s*;?\s*:")),
)

_RISK_RULES: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("privilege", _regex_rule(r"\b(?:sudo|doas)\b|\bsu\b")),
    ("network", _regex_rule(r"\b(?:curl|wget|ssh|scp|sftp|nc|ncat|netcat|telnet|ftp)\b")),
    (
        "install",
        _regex_rule(
            r"\b(?:apt|apt-get|dnf|yum|pacman|zypper)\b"
            r"|\bpip3?\s+install\b"
            r"|\bnpm\s+(?:install|ci|i)\b"
            r"|\byarn\s+(?:add|install)\b"
            r"|\bcargo\s+install\b"
        ),
    ),
    ("git_push", _regex_rule(r"\bgit\s+(?:-\S+\s+)*push\b")),
    ("service", _regex_rule(r"\b(?:systemctl|reboot|shutdown|halt|poweroff)\b")),
)


def _programs(command: str) -> list[str]:
    """Program names (basenames) invoked anywhere in the command line."""
    found: list[str] = []
    for segment in _segments(command):
        tokens = segment.split()
        while tokens and "=" in tokens[0] and not tokens[0].startswith("="):
            tokens = tokens[1:]  # skip VAR=value prefixes
        if tokens:
            found.append(tokens[0].rsplit("/", 1)[-1])
    return found


def classify(command: str, *, mode: str, allowlist: Sequence[str]) -> GuardDecision:
    """Decide whether a command may run, must prompt, or is denied outright."""
    for name, rule in _DENY_RULES:
        if rule(command):
            return GuardDecision("deny", f"deny:{name}")

    programs = _programs(command)
    if programs and all(program in allowlist for program in programs):
        return GuardDecision("allow", "allowlisted")

    if mode == "strict":
        return GuardDecision("deny", "deny:not_allowlisted")
    if mode == "auto":
        return GuardDecision("allow", "auto")

    for name, rule in _RISK_RULES:
        if rule(command):
            return GuardDecision("confirm", f"risk:{name}")
    return GuardDecision("allow", "default")
