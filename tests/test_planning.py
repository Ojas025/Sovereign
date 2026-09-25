"""Plan phase: structured plan generation, parsing, and the approval callback."""

import pytest

from workbench.agent.planning import (
    Approval,
    PlanParseError,
    auto_approver,
    parse_plan,
    plan_prompt,
)
from workbench.core.session import Plan, PlanStep


def test_parse_plain_json_object() -> None:
    plan = parse_plan('{"steps": ["read the parser", "fix the bug", "run tests"]}')

    assert [step.text for step in plan.steps] == [
        "read the parser",
        "fix the bug",
        "run tests",
    ]
    assert [step.index for step in plan.steps] == [1, 2, 3]
    assert all(step.status == "pending" for step in plan.steps)


def test_parse_fenced_block_with_surrounding_prose() -> None:
    plan = parse_plan(
        "Here is my plan:\n```json\n"
        '{"steps": ["first", "second"]}\n'
        "```\nLet me know!"
    )

    assert [step.text for step in plan.steps] == ["first", "second"]


def test_parse_picks_object_with_steps_over_earlier_json() -> None:
    plan = parse_plan('noise {"other": 1} then {"steps": ["only"]}')

    assert [step.text for step in plan.steps] == ["only"]


def test_parse_rejects_missing_steps_key() -> None:
    with pytest.raises(PlanParseError):
        parse_plan('{"todo": ["a"]}')


def test_parse_rejects_empty_steps() -> None:
    with pytest.raises(PlanParseError):
        parse_plan('{"steps": []}')


def test_parse_rejects_non_string_steps() -> None:
    with pytest.raises(PlanParseError):
        parse_plan('{"steps": [1, 2]}')


def test_parse_rejects_blank_step_text() -> None:
    with pytest.raises(PlanParseError):
        parse_plan('{"steps": ["   "]}')

def test_parse_rejects_free_text() -> None:
    with pytest.raises(PlanParseError):
        parse_plan("Sure — I would start by looking around and then fix things.")


def test_first_attempt_prompt_is_a_json_contract() -> None:
    prompt = plan_prompt(attempt=1)

    assert "steps" in prompt
    assert "JSON" in prompt
    assert "feedback" not in prompt.lower()


def test_retry_prompt_carries_rejection_feedback() -> None:
    prompt = plan_prompt(attempt=2, feedback="too vague, skip the refactor")

    assert "too vague, skip the refactor" in prompt
    assert str(2) in prompt


async def test_auto_approver_approves_every_plan() -> None:
    approval = await auto_approver(Plan(steps=[PlanStep(index=1, text="x")]), 1)

    assert approval.decision == "approved"


def test_approval_feedback_defaults_to_empty() -> None:
    assert Approval(decision="rejected").feedback == ""
