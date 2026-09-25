"""TierPolicy: threshold + confidence gate + overrides, escalation, hysteresis."""

from workbench.config import RoutingConfig, TierPolicyConfig
from workbench.core.protocols import Classification
from workbench.routing.policy import Phase, PolicyInputs, TierPolicy


def cls(
    intent: str = "chat_qa",
    difficulty: float = 1.0,
    confidence: float = 0.9,
    needs_tools: bool = False,
) -> Classification:
    return Classification(
        intent=intent,
        difficulty=difficulty,
        confidence=confidence,
        needs_tools=needs_tools,
        backend="test",
    )


def inputs(classification: Classification, **kwargs: object) -> PolicyInputs:
    return PolicyInputs(classification=classification, **kwargs)  # type: ignore[call-arg]


def test_difficulty_threshold_routes_first_turn() -> None:
    policy = TierPolicy(RoutingConfig())

    high = policy.decide(inputs(cls(difficulty=2.5)))
    low = policy.decide(inputs(cls(difficulty=1.0)))

    assert (high.tier, high.reason) == ("mid", "difficulty")
    assert (low.tier, low.reason) == ("small", "difficulty")
    assert high.suggest_plan is False and low.suggest_plan is False
    assert high.escalated is False


def test_needs_tools_routes_mid_and_suggests_plan() -> None:
    policy = TierPolicy(RoutingConfig())

    decision = policy.decide(inputs(cls(difficulty=0.5, needs_tools=True)))

    assert (decision.tier, decision.reason) == ("mid", "needs_tools")
    assert decision.suggest_plan is True


def test_planning_intent_routes_mid_and_suggests_plan() -> None:
    policy = TierPolicy(RoutingConfig())

    decision = policy.decide(inputs(cls(intent="planning", difficulty=0.5)))

    assert (decision.tier, decision.reason) == ("mid", "planning")
    assert decision.suggest_plan is True


def test_low_confidence_defaults_to_config_tier() -> None:
    policy = TierPolicy(RoutingConfig())

    decision = policy.decide(inputs(cls(confidence=0.55)))

    assert (decision.tier, decision.reason) == ("mid", "low_confidence")
    assert decision.suggest_plan is False

    strict = TierPolicy(RoutingConfig(policy=TierPolicyConfig(default_tier="small")))
    assert strict.decide(inputs(cls(confidence=0.55))).tier == "small"


def test_confidence_gate_is_strict_inequality() -> None:
    policy = TierPolicy(RoutingConfig())

    decision = policy.decide(inputs(cls(confidence=0.6)))

    assert (decision.tier, decision.reason) == ("small", "difficulty")


def test_config_intent_override_wins_over_threshold() -> None:
    config = RoutingConfig(intent_overrides={"meta": "small"})
    policy = TierPolicy(config)

    decision = policy.decide(inputs(cls(intent="meta", difficulty=2.5)))

    assert decision.tier == "small"
    assert decision.reason == "intent_override:meta"
    assert decision.suggest_plan is False


def test_escalation_fires_once_after_two_failures() -> None:
    policy = TierPolicy(RoutingConfig())

    fired = policy.decide(
        inputs(cls(), current_tier="small", consecutive_tool_errors=2)
    )
    spent = policy.decide(
        inputs(cls(), current_tier="small", consecutive_tool_errors=2, escalated=True)
    )

    assert (fired.tier, fired.reason, fired.escalated) == ("mid", "tool_failures", True)
    assert (spent.tier, spent.escalated) == ("small", False)
    assert policy.decide(
        inputs(cls(), current_tier="small", consecutive_tool_errors=1)
    ).tier == "small"


def test_hysteresis_blocks_downgrade_inside_band() -> None:
    policy = TierPolicy(RoutingConfig())

    blocked = policy.decide(
        inputs(cls(difficulty=1.8), current_tier="mid", previous_intent="chat_qa")
    )
    clear = policy.decide(
        inputs(cls(difficulty=1.5), current_tier="mid", previous_intent="chat_qa")
    )

    assert (blocked.tier, blocked.reason) == ("mid", "hysteresis")
    assert (clear.tier, clear.reason) == ("small", "difficulty")


def test_hysteresis_blocks_upgrade_inside_band() -> None:
    policy = TierPolicy(RoutingConfig())

    blocked = policy.decide(
        inputs(cls(difficulty=2.1), current_tier="small", previous_intent="chat_qa")
    )
    clear = policy.decide(
        inputs(cls(difficulty=2.4), current_tier="small", previous_intent="chat_qa")
    )

    assert (blocked.tier, blocked.reason) == ("small", "hysteresis")
    assert (clear.tier, clear.reason) == ("mid", "difficulty")


def test_intent_class_change_bypasses_hysteresis() -> None:
    policy = TierPolicy(RoutingConfig())

    decision = policy.decide(
        inputs(
            cls(difficulty=1.8),
            current_tier="mid",
            previous_intent="file_edit",  # intent differs this turn
        )
    )

    assert (decision.tier, decision.reason) == ("small", "difficulty")


def test_binary_rule_changes_bypass_hysteresis_band() -> None:
    """needs_tools-driven upgrades must not deadlock inside the delta band."""
    policy = TierPolicy(RoutingConfig())

    decision = policy.decide(
        inputs(
            cls(difficulty=1.0, needs_tools=True),
            current_tier="small",
            previous_intent="chat_qa",  # same intent: only the rule driver changed
        )
    )

    assert (decision.tier, decision.reason) == ("mid", "needs_tools")


def test_pinned_mode_is_advisory_only() -> None:
    policy = TierPolicy(RoutingConfig())

    decision = policy.decide(
        inputs(cls(difficulty=3.0), current_tier="small", pinned=True)
    )
    escalated = policy.decide(
        inputs(
            cls(), current_tier="small", consecutive_tool_errors=5, pinned=True
        )
    )

    assert (decision.tier, decision.reason) == ("small", "pinned")
    assert decision.escalated is False
    assert escalated.tier == "small"  # pin also beats escalation


def test_phase_updates_after_two_consecutive_tool_calls() -> None:
    policy = TierPolicy(RoutingConfig())

    def phase(prior: Phase, tools: tuple[str, ...]) -> Phase:
        return policy.decide(
            inputs(cls(), prior_phase=prior, recent_tools=tools)
        ).phase

    assert phase("none", ("read", "read")) == "explore"
    assert phase("implement", ("bash", "bash")) == "verify"
    assert phase("implement", ("write", "write")) == "implement"
    # alternating or incomplete history keeps the prior phase
    assert phase("implement", ("read", "write")) == "implement"
    assert phase("implement", ("read",)) == "implement"
    assert phase("explore", ("mystery", "mystery")) == "explore"
