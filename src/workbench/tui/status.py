"""Status line for the input prompt's bottom toolbar (PLAN §7, decision 11).

Rendered at prompt time from the projected event state: tier, model, context
pressure (green → amber → red as the active profile's ctx budget fills),
intent, rounds, tokens, escalations, sandbox backend, and a short session id.
"""

from __future__ import annotations

from workbench.config import Config
from workbench.tui.state import TuiState

# Context pressure thresholds (ruling M6.8): at/under 60% is comfortable, at/under
# 85% warns, beyond it is danger — measured against the ACTIVE tier's profile
# ctx_len so the bar tracks the model that is actually streaming.
_CTX_OK_MAX = 0.6
_CTX_WARN_MAX = 0.85

_FIELD_SEPARATOR = " · "


def build_status_line(
    state: TuiState, config: Config, *, pinned: bool = False
) -> list[tuple[str, str]]:
    """Formatted-text segments for prompt_toolkit's ``bottom_toolbar``."""
    tier = state.tier or "-"
    if pinned and state.tier:
        tier = f"{tier} (pinned)"
    fields: list[tuple[str, str]] = [
        ("", f"tier:{tier}"),
        ("", f"model:{state.model or '-'}"),
        _ctx_segment(state, config),
        ("", f"intent:{state.intent or '-'}"),
        ("", f"rounds:{state.rounds}/{config.agent.max_rounds}"),
        ("", f"tok:{_count(state.tokens)}"),
        ("", f"esc:{state.escalations}"),
        ("", f"sandbox:{config.sandbox.backend}"),
        ("", f"ses:{state.session_id[:4] or '-'}"),
    ]
    parts: list[tuple[str, str]] = []
    for index, field in enumerate(fields):
        if index:
            parts.append(("", _FIELD_SEPARATOR))
        parts.append(field)
    return parts


def _ctx_segment(state: TuiState, config: Config) -> tuple[str, str]:
    """ctx usage against the routed tier's profile; '-' until a route exists."""
    if not state.tier:
        return ("", "ctx:-")
    profile_name = config.routing.tiers.get(state.tier)
    profile = config.models.profiles.get(profile_name) if profile_name else None
    if profile is None:
        return ("", "ctx:-")
    limit = profile.ctx_len
    ratio = state.prompt_tokens / limit if limit else 0.0
    if ratio <= _CTX_OK_MAX:
        style = "ansigreen"
    elif ratio <= _CTX_WARN_MAX:
        style = "ansiyellow"
    else:
        style = "ansired"
    return (style, f"ctx:{_count(state.prompt_tokens)}/{_count(limit)} ({int(ratio * 100)}%)")


def _count(value: int) -> str:
    """Compact counts: 999 → "999", 4100 → "4.1k", 32000 → "32k"."""
    if value < 1000:
        return str(value)
    scaled = f"{value / 1000:.1f}".rstrip("0").rstrip(".")
    return f"{scaled}k"
