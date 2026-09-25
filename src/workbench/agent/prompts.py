"""Prompt fragments the loop sends outside the user's own words.

Pure string builders, so tests (and later the TUI) can show exactly what the
model was asked: identity, the approved-plan handoff, the reflection self-check,
and the nudge that puts an off-track turn back on the plan.
"""

from __future__ import annotations

from workbench.core.protocols import ToolProtocol
from workbench.core.session import Plan

_BASE = (
    "You are workbench, a local AI assistant running fully offline in a "
    "sandboxed workspace. Be direct and concrete; use tools when they help "
    "and answer directly when they don't."
)

_JSON_PROTOCOL = """
To call a tool, output a fenced block exactly like this:
```tool
{"name": "read", "arguments": {...}}
```
One block per tool call; its result comes back as a message. Outside blocks,
reply normally."""


def system_prompt(protocol: ToolProtocol) -> str:
    """Identity line plus the tool contract for the active protocol."""
    if protocol == "native":
        return _BASE + " Call the provided functions to act; results are returned to you."
    if protocol == "json":
        return _BASE + _JSON_PROTOCOL
    return _BASE


def approved_plan_message(plan: Plan) -> str:
    """The approved plan enters the conversation as user context for execution."""
    steps = "\n".join(f"{step.index}. {step.text}" for step in plan.steps)
    return f"Plan approved — work through it step by step:\n{steps}"


def reflect_prompt(plan: Plan) -> str:
    """Bounded self-check at plan boundaries; verdict line drives done/nudge/revise."""
    steps = "\n".join(f"{step.index}. [{step.status}] {step.text}" for step in plan.steps)
    return (
        f"Compare your progress against this plan:\n{steps}\n\n"
        "Reply with exactly two lines:\n"
        "completed: <comma-separated numbers of fully finished steps>\n"
        "verdict: DONE | CONTINUE | REVISE <short reason>\n"
        "Use DONE only when every plan step is finished and the request is satisfied."
    )


def nudge_prompt(*, verdict: str, reason: str, plan: Plan) -> str:
    """Correction injected after reflection: keep the loop moving or ask again."""
    pending = plan.first_pending()
    if verdict == "revise":
        base = f"Correction: {reason}."
    elif verdict == "done":
        unfinished = ", ".join(str(step.index) for step in plan.steps if step.status != "done")
        base = f"You reported the task complete, but plan steps are unfinished: {unfinished}."
    else:
        base = "Stay on track with the plan."
    if pending is None:
        return f"{base} All steps are done — give your final answer."
    return f"{base} Continue with step {pending.index}: {pending.text}."
