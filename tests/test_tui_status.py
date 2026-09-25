"""Status line for the input prompt toolbar (PLAN §7, decision 11, ruling M6.8)."""

from __future__ import annotations

from workbench.config import (
    AgentConfig,
    Config,
    ModelProfile,
    ModelsConfig,
    RoutingConfig,
    SandboxConfig,
)
from workbench.tui.state import TuiState
from workbench.tui.status import build_status_line

_CTX_LIMIT = 32768


def _config() -> Config:
    return Config(
        routing=RoutingConfig(tiers={"small": "small", "mid": "coder"}),
        models=ModelsConfig(profiles={"coder": ModelProfile(ctx_len=_CTX_LIMIT)}),
        agent=AgentConfig(max_rounds=15),
        sandbox=SandboxConfig(backend="bwrap"),
    )


def _state(**overrides: object) -> TuiState:
    state = TuiState()
    state.session_id = "a1b2c3d4e5f6"
    state.tier = "mid"
    state.model = "coder7b"
    state.intent = "code_gen"
    state.rounds = 3
    state.tokens = 4100
    state.prompt_tokens = 4096
    state.escalations = 0
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _text(parts: list[tuple[str, str]]) -> str:
    return "".join(text for _style, text in parts)


def test_status_line_shows_every_documented_field() -> None:
    parts = build_status_line(_state(), _config())

    assert _text(parts) == (
        "tier:mid │ model:coder7b │ ctx:4.1k/32.8k (12%) │ intent:code_gen │ "
        "rounds:3/15 │ tok:4.1k │ esc:0 │ sandbox:bwrap │ ses:a1b2 │ /model to switch"
    )


def test_ctx_segment_is_green_below_the_warn_threshold() -> None:
    parts = build_status_line(_state(prompt_tokens=int(_CTX_LIMIT * 0.5)), _config())

    ctx_style = next(style for style, text in parts if text.startswith("ctx:"))
    assert ctx_style == "ansigreen"


def test_ctx_segment_is_amber_then_red_as_the_limit_approaches() -> None:
    amber = build_status_line(_state(prompt_tokens=int(_CTX_LIMIT * 0.7)), _config())
    red = build_status_line(_state(prompt_tokens=int(_CTX_LIMIT * 0.95)), _config())

    amber_style = next(style for style, text in amber if text.startswith("ctx:"))
    red_style = next(style for style, text in red if text.startswith("ctx:"))
    assert amber_style == "ansiyellow"
    assert red_style == "ansired"


def test_ctx_threshold_boundaries_are_inclusive() -> None:
    at_ok = build_status_line(_state(prompt_tokens=int(_CTX_LIMIT * 0.6)), _config())
    at_warn = build_status_line(_state(prompt_tokens=int(_CTX_LIMIT * 0.85)), _config())

    assert next(style for style, text in at_ok if text.startswith("ctx:")) == "ansigreen"
    assert next(style for style, text in at_warn if text.startswith("ctx:")) == "ansiyellow"


def test_unrouted_session_shows_placeholders_not_crashes() -> None:
    state = TuiState()
    state.session_id = "ffff"

    text = _text(build_status_line(state, _config()))

    assert text.startswith("tier:- │ model:- │ ctx:- │ intent:- │ rounds:0/15")
    assert text.endswith("sandbox:bwrap │ ses:ffff │ /model to switch")


def test_pinned_tier_is_marked_in_the_status_line() -> None:
    text = _text(build_status_line(_state(), _config(), pinned=True))

    assert "tier:mid (pinned)" in text


def test_ctx_segment_resolves_direct_profile_name() -> None:
    # state.tier set to profile name "coder" directly, rather than "mid"
    parts = build_status_line(_state(tier="coder"), _config())
    ctx_text = next(text for _style, text in parts if text.startswith("ctx:"))
    assert "ctx:4.1k/32.8k" in ctx_text

