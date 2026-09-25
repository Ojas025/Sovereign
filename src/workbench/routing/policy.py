"""TierPolicy: deterministic tier selection over one turn's classification.

Precedence (plan §4.3), first rule that applies wins:

1. pinned            → advisory passthrough (the caller honors the pin)
2. confidence gate   → config default tier (default-to-larger on uncertainty)
3. config overrides  → explicit ``intent → tier`` rules from config
4. escalation        → small→mid once after N consecutive tool failures
5. union             → (planning | needs_tools) → mid, else difficulty threshold
6. hysteresis        → Schmitt band ±δ around the difficulty threshold; bypassed
                       when the intent class changed or when a binary rule (3/4/5
                       other than difficulty) drove the candidate tier

The policy is a pure function: the caller owns emitting ``route_decided`` /
``route_escalated`` events and keeping per-task state (errors, escalation flag).
"""

from __future__ import annotations

from dataclasses import dataclass

from workbench.config import RoutingConfig
from workbench.core.protocols import Classification, Phase

__all__ = ["Phase", "PolicyInputs", "RouteDecision", "TierPolicy"]

# MVP is two-tier by design (plan decision 7): the tier names are fixed.
_SMALL = "small"
_MID = "mid"

# 2-consecutive-call stability: two recent calls of one class move the phase.
_TOOL_PHASES: dict[str, Phase] = {
    "read": "explore",
    "bash": "verify",
    "write": "implement",
    "edit": "implement",
}


@dataclass(frozen=True, slots=True)
class PolicyInputs:
    """Everything the policy needs for one turn (caller tracks state between turns)."""

    classification: Classification
    current_tier: str | None = None  # tier chosen on the previous turn
    previous_intent: str | None = None  # intent chosen on the previous turn
    prior_phase: Phase = "none"
    recent_tools: tuple[str, ...] = ()  # two most recent tool names, newest first
    consecutive_tool_errors: int = 0
    escalated: bool = False  # escalation already spent for this task
    pinned: bool = False  # /model pin active → router is advisory-only


@dataclass(frozen=True, slots=True)
class RouteDecision:
    tier: str
    reason: str  # stable audit token, e.g. "needs_tools", "intent_override:meta"
    suggest_plan: bool
    escalated: bool  # this decision performed the one-shot escalation
    phase: Phase  # phase updated via the 2-consecutive-call stability rule


def _next_phase(prior: Phase, recent_tools: tuple[str, ...]) -> Phase:
    """Phase moves only after two consecutive calls of the same tool class."""
    if len(recent_tools) < 2:
        return prior
    first, second = recent_tools[0], recent_tools[1]
    if first not in _TOOL_PHASES or second not in _TOOL_PHASES:
        return prior
    first_phase = _TOOL_PHASES[first]
    return first_phase if first_phase == _TOOL_PHASES[second] else prior


class TierPolicy:
    def __init__(self, routing: RoutingConfig) -> None:
        policy = routing.policy
        self._high = policy.high_threshold
        self._gate = policy.confidence_gate
        self._default = policy.default_tier
        self._delta = policy.hysteresis_delta
        self._escalate_after = policy.escalate_after_failures
        self._overrides = dict(routing.intent_overrides)

    def decide(self, inputs: PolicyInputs) -> RouteDecision:
        classification = inputs.classification
        phase = _next_phase(inputs.prior_phase, inputs.recent_tools)
        suggest = classification.intent == "planning" or classification.needs_tools

        if inputs.pinned:
            tier = inputs.current_tier or self._default
            return RouteDecision(tier, "pinned", False, False, phase)

        if classification.confidence < self._gate:
            return RouteDecision(self._default, "low_confidence", False, False, phase)

        override = self._overrides.get(classification.intent)
        if override is not None:
            reason = f"intent_override:{classification.intent}"
            return RouteDecision(override, reason, suggest, False, phase)

        if (
            self._escalate_after > 0
            and inputs.consecutive_tool_errors >= self._escalate_after
            and inputs.current_tier == _SMALL
            and not inputs.escalated
        ):
            return RouteDecision(_MID, "tool_failures", suggest, True, phase)

        if suggest:
            reason = "planning" if classification.intent == "planning" else "needs_tools"
            candidate = _MID
        elif classification.difficulty >= self._high:
            reason, candidate = "difficulty", _MID
        else:
            reason, candidate = "difficulty", _SMALL

        tier = self._stabilize(candidate, inputs, reason)
        if tier != candidate:
            reason = "hysteresis"
        return RouteDecision(tier, reason, suggest, False, phase)

    def _stabilize(self, candidate: str, inputs: PolicyInputs, reason: str) -> str:
        """Apply the ±δ band to difficulty-driven changes only (no deadlocks)."""
        current = inputs.current_tier
        classification = inputs.classification
        if current is None or candidate == current:
            return candidate
        previous = inputs.previous_intent
        if previous is not None and classification.intent != previous:
            return candidate  # intent class changed → accept immediately
        if reason != "difficulty":
            return candidate  # binary rules (needs_tools/planning) bypass the band
        if candidate == _MID:
            return _MID if classification.difficulty >= self._high + self._delta else current
        return _SMALL if classification.difficulty < self._high - self._delta else current
