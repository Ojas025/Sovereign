"""Plan phase: structured plan generation, parsing, and the approval callback.

The plan is the contract the user reviews before autonomous execution starts
(plan §5.1): the model is asked for strict JSON steps, ``parse_plan`` extracts
them from whatever prose it answered with, and an async ``Approver`` decides —
interactive (TUI, M6) or headless (print mode).
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from workbench.core.session import Plan, PlanStep

ApprovalDecision = Literal["approved", "rejected", "cancelled"]


class PlanParseError(Exception):
    """Model output contained no usable ``{"steps": [...]}`` plan."""


@dataclass(frozen=True, slots=True)
class Approval:
    decision: ApprovalDecision
    feedback: str = ""  # free-text reason on rejection, fed into the next attempt


# Async so a TUI approver can prompt without blocking the loop.
Approver = Callable[[Plan, int], Awaitable[Approval]]


def plan_prompt(*, attempt: int, feedback: str = "") -> str:
    """Build the plan request; retries embed the user's rejection feedback verbatim."""
    lines = [
        "Before acting, produce a plan for the user's request.",
        'Reply with ONLY a JSON object of the form {"steps": ["...", "..."]}: an ordered',
        "list of 2-6 short imperative steps (optionally inside a ```json fence).",
    ]
    if feedback:
        lines += [
            "",
            f"The user rejected your previous plan (attempt {attempt}):",
            feedback,
            "Revise the plan to address that feedback.",
        ]
    return "\n".join(lines)


def parse_plan(text: str) -> Plan:
    """Extract the first JSON object carrying a non-empty list of strings as steps."""
    payload = _extract_steps_object(text)
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanParseError("'steps' must be a non-empty list")
    if not all(isinstance(step, str) and step.strip() for step in steps):
        raise PlanParseError("every step must be a non-empty string")
    return Plan(
        steps=[
            PlanStep(index=index, text=str(step).strip())
            for index, step in enumerate(steps, start=1)
        ]
    )


def _extract_steps_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            candidate, _end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "steps" in candidate:
            return candidate
    raise PlanParseError("no JSON object with a 'steps' key in model output")


async def auto_approver(plan: Plan, attempt: int) -> Approval:
    """Headless policy: print mode IS the user's request, so approve unconditionally.

    Interactive y/N review belongs to the TUI (M6); a declined plan there can
    still be regenerated with feedback.
    """
    return Approval(decision="approved")
