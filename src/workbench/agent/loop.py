"""Plan → approve → execute → reflect: one turn of autonomous work.

State machine (plan §5.1): classify and route the message, maybe generate and
approve a plan, then stream model rounds whose tool calls run through the
sandboxed registry — reflecting at plan boundaries until the model reports the
plan complete or a budget trips. Every budget breach (max_rounds, wall clock,
per-turn tokens, reflection nudges) is a graceful stop with a state summary,
never an exception escaping the turn.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import uuid4

from workbench.agent.json_calls import JsonToolAction, JsonToolError, parse_json_tool_calls
from workbench.agent.planning import Approver, PlanParseError, parse_plan, plan_prompt
from workbench.agent.prompts import (
    approved_plan_message,
    nudge_prompt,
    reflect_prompt,
    system_prompt,
)
from workbench.config import Config
from workbench.core.events import Event, EventBus
from workbench.core.protocols import (
    ChatMessage,
    ChatRequest,
    Classification,
    LLMClient,
    Router,
    StreamEnd,
    TextDelta,
    Tool,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    ToolContext,
    ToolProtocol,
    ToolResult,
    ToolSpec,
    Usage,
    UsageReport,
)
from workbench.core.session import Plan, Session
from workbench.llm.client import LLMError
from workbench.logging_setup import get_logger, set_turn_id
from workbench.routing.policy import PolicyInputs, RouteDecision, TierPolicy

logger = get_logger("agent")

OutcomeKind = Literal["completed", "cancelled", "budget", "error"]

_REASON_REVISION_FEEDBACK = (
    "Your previous reply was not a JSON object with a \"steps\" list."
)
_REASON_NO_FEEDBACK = "Make the steps more concrete and actionable."
# Events are the TUI's display channel (and NDJSON's): cap a tool result so a
# 64KB file read never floods the stream — full content lives in the session JSONL.
_TOOL_OUTPUT_EVENT_CHARS = 4000


def _display_output(content: str) -> str:
    """Size-cap a tool result for the ``tool_execution_end`` event payload."""
    if len(content) <= _TOOL_OUTPUT_EVENT_CHARS:
        return content
    return content[:_TOOL_OUTPUT_EVENT_CHARS] + "…"


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """What the user gets back from one turn: final text, how it ended, model calls."""

    text: str
    outcome: OutcomeKind
    rounds: int


@dataclass(frozen=True, slots=True)
class Reflection:
    kind: Literal["done", "continue", "revise"]
    completed: tuple[int, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class _Round:
    """Accumulated result of one streamed model call."""

    text: str
    calls: tuple[tuple[str, str, str], ...]  # (call_id, name, arguments-json)


class _Budget(Exception):
    """Internal control flow: a configured budget is exhausted for this turn."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def resolve_model(config: Config, tier: str) -> str:
    """Tier → profile → llama-server id: filename stem when the profile has a path

    (M2 finding: router model ids are stems), else the profile/tier name.
    """
    profile_name = config.routing.tiers.get(tier, tier)
    profile = config.models.profiles.get(profile_name)
    if profile is not None and profile.path:
        return Path(profile.path).stem
    return profile_name


def resolve_protocol(config: Config, tier: str) -> ToolProtocol:
    """Profile's tool_calling mode; ``auto`` acts as native until GGUF probing (M8)."""
    profile_name = config.routing.tiers.get(tier, tier)
    profile = config.models.profiles.get(profile_name)
    if profile is not None and profile.tool_calling == "json":
        return "json"
    if profile is not None and profile.tool_calling == "none":
        return "none"
    return "native"


class AgentLoop:
    """Drives plan/execute/reflect for one session; owns no server or TUI."""

    def __init__(
        self,
        *,
        client: LLMClient,
        router: Router,
        policy: TierPolicy,
        tools: Mapping[str, Tool],
        session: Session,
        bus: EventBus,
        config: Config,
        context: ToolContext,
        approver: Approver,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._client = client
        self._router = router
        self._policy = policy
        self._tools = dict(tools)
        self._session = session
        self._bus = bus
        self._config = config
        self._context = context
        self._approver = approver
        self._clock = clock
        self._rounds = 0
        self._tokens = 0
        self._started = 0.0

    # --- turn ---------------------------------------------------------------

    async def run_turn(self, user_message: str) -> TurnOutcome:
        """Run one user message to completion, cancellation, or a budget stop."""
        turn_id = uuid4().hex[:8]
        set_turn_id(turn_id)
        self._rounds = 0
        self._tokens = 0
        self._started = self._clock()
        self._bus.emit(Event("turn_start", {"turn_id": turn_id, "message": user_message}))
        self._session.append_message(ChatMessage(role="user", content=user_message))
        classification: Classification | None = None
        try:
            route_started = self._clock()
            classification = await self._router.classify(user_message)
            decision = self._decide(
                classification,
                announce=True,
                route_duration_s=self._clock() - route_started,
            )
            outcome: OutcomeKind
            text: str
            plan_hinted = (
                decision.suggest_plan
                or classification.difficulty >= self._config.routing.policy.high_threshold
            )
            if plan_hinted and self._session.plan is None:
                model = resolve_model(self._config, decision.tier)
                cancel_reason = await self._plan_phase(model=model)
                if cancel_reason is not None:
                    return self._finish(
                        outcome="cancelled",
                        text=cancel_reason,
                        user_message=user_message,
                        classification=classification,
                    )
            outcome, text = await self._execute(classification)
        except _Budget as budget:
            self._bus.emit(
                Event(
                    "budget_exceeded",
                    {
                        "reason": budget.reason,
                        "rounds": self._rounds,
                        "elapsed_s": self._clock() - self._started,
                    },
                )
            )
            outcome = "budget"
            text = self._summary(budget.reason)
            self._session.append_message(ChatMessage(role="assistant", content=text))
        except LLMError as error:
            self._bus.emit(Event("error", {"stage": "llm", "message": str(error)}))
            outcome = "error"
            text = f"LLM error: {error}"
            self._session.append_message(ChatMessage(role="assistant", content=text))
        return self._finish(
            outcome=outcome,
            text=text,
            user_message=user_message,
            classification=classification,
        )

    def _finish(
        self,
        *,
        outcome: OutcomeKind,
        text: str,
        user_message: str,
        classification: Classification | None,
    ) -> TurnOutcome:
        duration_s = self._clock() - self._started
        self._bus.emit(
            Event(
                "turn_end",
                {"outcome": outcome, "rounds": self._rounds, "duration_s": duration_s},
            )
        )
        self._session.record_turn(
            user_message=user_message,
            outcome=outcome,
            rounds=self._rounds,
            duration_s=duration_s,
            intent=classification.intent if classification is not None else "unknown",
        )
        set_turn_id(None)
        return TurnOutcome(text=text, outcome=outcome, rounds=self._rounds)

    # --- routing ------------------------------------------------------------

    def _decide(
        self,
        classification: Classification,
        *,
        announce: bool,
        route_duration_s: float = 0.0,
    ) -> RouteDecision:
        session = self._session
        if session.pinned_tier is not None:
            # /model pin: the policy is advisory until the pin clears (M6.7)
            session.tier = session.pinned_tier
        decision = self._policy.decide(
            PolicyInputs(
                classification=classification,
                current_tier=session.tier,
                previous_intent=session.previous_intent,
                prior_phase=session.phase,
                recent_tools=session.recent_tools,
                consecutive_tool_errors=session.consecutive_tool_errors,
                escalated=session.escalated,
                pinned=session.pinned_tier is not None,
            )
        )
        previous_tier = session.tier
        session.tier = decision.tier
        session.phase = decision.phase
        if decision.escalated:
            session.escalated = True
            self._bus.emit(
                Event(
                    "route_escalated",
                    {
                        "from_tier": previous_tier,
                        "to_tier": decision.tier,
                        "failures": session.consecutive_tool_errors,
                    },
                )
            )
        if announce:
            self._bus.emit(
                Event(
                    "route_decided",
                    {
                        "tier": decision.tier,
                        "reason": decision.reason,
                        "intent": classification.intent,
                        "difficulty": classification.difficulty,
                        "confidence": classification.confidence,
                        "backend": classification.backend,
                        "model": resolve_model(self._config, decision.tier),
                        "suggest_plan": decision.suggest_plan,
                        "duration_s": route_duration_s,  # classify+decide (M7.2)
                    },
                )
            )
        return decision

    def _round_route(self, classification: Classification) -> tuple[str, ToolProtocol]:
        """Round-top decide: refreshes phase/tier and fires the escalation hook."""
        decision = self._decide(classification, announce=False)
        return (
            resolve_model(self._config, decision.tier),
            resolve_protocol(self._config, decision.tier),
        )

    # --- plan phase -----------------------------------------------------------

    async def _plan_phase(self, *, model: str) -> str | None:
        """Generate → approve → (regenerate | cancel); None means approved."""
        attempts = 1 + self._config.agent.plan_max_retries
        feedback = ""
        for attempt in range(1, attempts + 1):
            self._check_budget()
            request = self._request(
                model=model,
                messages=(
                    *self._session.messages,
                    ChatMessage(
                        role="user", content=plan_prompt(attempt=attempt, feedback=feedback)
                    ),
                ),
                protocol="none",
            )
            result = await self._ask(request)
            self._check_tokens()
            try:
                plan = parse_plan(result.text)
            except PlanParseError:
                self._bus.emit(
                    Event(
                        "plan_rejected",
                        {"decision": "unparseable", "attempt": attempt, "feedback": ""},
                    )
                )
                feedback = _REASON_REVISION_FEEDBACK
                continue
            self._bus.emit(
                Event(
                    "plan_proposed",
                    {"attempt": attempt, "steps": [step.text for step in plan.steps]},
                )
            )
            approval = await self._approver(plan, attempt)
            if approval.decision == "approved":
                self._bus.emit(Event("plan_approved", {"attempt": attempt}))
                self._session.set_plan(plan)
                self._session.append_message(
                    ChatMessage(role="user", content=approved_plan_message(plan))
                )
                return None
            self._bus.emit(
                Event(
                    "plan_rejected",
                    {
                        "decision": approval.decision,
                        "attempt": attempt,
                        "feedback": approval.feedback,
                    },
                )
            )
            if approval.decision == "cancelled":
                return "Plan cancelled."
            feedback = approval.feedback or _REASON_NO_FEEDBACK
        self._bus.emit(
            Event("plan_rejected", {"decision": "exhausted", "attempt": attempts, "feedback": ""})
        )
        return f"Plan generation failed after {attempts} attempts."

    # --- execute / reflect -----------------------------------------------------

    async def _execute(self, classification: Classification) -> tuple[OutcomeKind, str]:
        nudges = 0
        while True:
            self._check_budget()
            model, protocol = self._round_route(classification)
            result = await self._ask(
                self._request(model=model, messages=self._session.messages, protocol=protocol)
            )
            self._check_tokens()  # a blown token budget stops before the round's work runs
            if result.calls:
                await self._run_tools(result)
                continue
            actions = parse_json_tool_calls(result.text) if protocol == "json" else ()
            if actions:
                await self._run_json_actions(result, actions)
                continue
            self._session.append_message(ChatMessage(role="assistant", content=result.text))
            plan = self._session.plan
            if plan is None:
                return "completed", result.text
            reflection = await self._reflect(plan, classification)
            if reflection.kind == "done" and plan.all_done():
                return "completed", result.text
            if nudges >= self._config.agent.reflection_nudge_cap:
                raise _Budget("reflection")
            nudges += 1
            self._session.append_message(
                ChatMessage(
                    role="user",
                    content=nudge_prompt(
                        verdict=reflection.kind, reason=reflection.reason, plan=plan
                    ),
                )
            )

    async def _reflect(self, plan: Plan, classification: Classification) -> Reflection:
        """Ad-hoc self-check call: prompt and verdict stay out of the transcript."""
        model, _protocol = self._round_route(classification)
        request = self._request(
            model=model,
            messages=(
                *self._session.messages,
                ChatMessage(role="user", content=reflect_prompt(plan)),
            ),
            protocol="none",
        )
        result = await self._ask(request)
        reflection = _parse_reflection(result.text)
        completed = set(reflection.completed)
        for step in plan.steps:
            if step.index in completed and step.status != "done":
                step.status = "done"
        pending = plan.first_pending()
        if pending is not None and pending.status == "pending":
            pending.status = "active"
        self._session.set_plan(plan)
        self._bus.emit(
            Event(
                "reflection",
                {
                    "verdict": reflection.kind,
                    "completed": [step.index for step in plan.steps if step.index in completed],
                },
            )
        )
        return reflection

    async def _run_tools(self, result: _Round) -> None:
        """Execute the round's tool calls in order; every result feeds back as a message."""
        self._session.append_message(
            ChatMessage(
                role="assistant",
                content=result.text,
                tool_calls=tuple(
                    ToolCall(id=call_id, name=name, arguments=arguments)
                    for call_id, name, arguments in result.calls
                ),
            )
        )
        for call_id, name, arguments in result.calls:
            self._bus.emit(
                Event(
                    "tool_execution_start",
                    {"call_id": call_id, "tool": name, "arguments": arguments},
                )
            )
            started = self._clock()
            tool_result = await self._call_tool(name, arguments)
            self._bus.emit(
                Event(
                    "tool_execution_end",
                    {
                        "call_id": call_id,
                        "tool": name,
                        "status": "error" if tool_result.is_error else "ok",
                        "duration_s": self._clock() - started,
                        "output": _display_output(tool_result.content),
                        "blocked_reason": tool_result.blocked_reason,  # M7.5
                    },
                )
            )
            self._session.note_tool_result(tool=name, is_error=tool_result.is_error)
            self._session.append_message(
                ChatMessage(
                    role="tool",
                    tool_call_id=call_id,
                    name=name,
                    content=tool_result.content,
                )
            )

    async def _call_tool(self, name: str, arguments: str) -> ToolResult:
        """Native path: decode the streamed arguments JSON, then run the tool."""
        try:
            payload = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return ToolResult(
                f"invalid JSON arguments for {name}: {arguments[:200]}", is_error=True
            )
        if not isinstance(payload, dict):
            return ToolResult(f"arguments for {name} must be a JSON object", is_error=True)
        return await self._execute_tool(name, payload)

    async def _run_json_actions(self, result: _Round, actions: tuple[JsonToolAction, ...]) -> None:
        """json protocol: run parsed blocks; results go back as plain user messages

        because templates without tool support have no ``tool`` role either.
        """
        self._session.append_message(ChatMessage(role="assistant", content=result.text))
        for index, action in enumerate(actions):
            call_id = f"json_{index}"
            if isinstance(action, JsonToolError):
                reason = f"Tool call block malformed: {action.message}"
                self._bus.emit(
                    Event(
                        "tool_execution_start",
                        {"call_id": call_id, "tool": "parse", "arguments": ""},
                    )
                )
                self._session.note_tool_result(tool="parse", is_error=True)
                self._bus.emit(
                    Event(
                        "tool_execution_end",
                        {
                            "call_id": call_id,
                            "tool": "parse",
                            "status": "error",
                            "duration_s": 0.0,
                            "output": _display_output(reason),
                            "blocked_reason": None,  # uniform key set (M7.2)
                        },
                    )
                )
                self._session.append_message(
                    ChatMessage(
                        role="user",
                        content=reason,
                    )
                )
                continue
            self._bus.emit(
                Event(
                    "tool_execution_start",
                    {
                        "call_id": call_id,
                        "tool": action.name,
                        "arguments": json.dumps(action.arguments),
                    },
                )
            )
            started = self._clock()
            tool_result = await self._execute_tool(action.name, action.arguments)
            self._bus.emit(
                Event(
                    "tool_execution_end",
                    {
                        "call_id": call_id,
                        "tool": action.name,
                        "status": "error" if tool_result.is_error else "ok",
                        "duration_s": self._clock() - started,
                        "output": _display_output(tool_result.content),
                        "blocked_reason": tool_result.blocked_reason,  # M7.5
                    },
                )
            )
            self._session.note_tool_result(tool=action.name, is_error=tool_result.is_error)
            self._session.append_message(
                ChatMessage(
                    role="user",
                    content=f"Tool '{action.name}' returned:\n{tool_result.content}",
                )
            )

    async def _execute_tool(self, name: str, payload: dict[str, object]) -> ToolResult:
        """Validate then run one call; tool bugs become model-visible errors, not crashes."""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools))
            return ToolResult(f"unknown tool {name!r}; available: {available}", is_error=True)
        try:
            return await tool.execute(payload, self._context)
        except Exception as exc:
            logger.exception("tool %s crashed", name)
            self._bus.emit(
                Event("error", {"stage": "tool", "message": f"tool {name!r} failed: {exc}"})
            )
            return ToolResult(f"tool {name!r} failed: {exc}", is_error=True)

    # --- plumbing ---------------------------------------------------------------

    def _request(
        self, *, model: str, messages: Sequence[ChatMessage], protocol: ToolProtocol
    ) -> ChatRequest:
        specs: tuple[ToolSpec, ...] = ()
        if protocol == "native":  # json/none models never see native tool specs
            specs = tuple(
                ToolSpec(tool.name, tool.description, tool.parameters)
                for tool in self._tools.values()
            )
        return ChatRequest(
            messages=(
                ChatMessage(role="system", content=system_prompt(protocol)),
                *messages,
            ),
            model=model,
            tools=specs,
        )

    async def _ask(self, request: ChatRequest) -> _Round:
        """Stream one model call: message events out, text/calls/usage accumulated."""
        self._rounds += 1
        self._bus.emit(Event("message_start", {"model": request.model, "role": "assistant"}))
        stream_started = self._clock()
        text_parts: list[str] = []
        calls: dict[str, tuple[str, list[str]]] = {}
        usage: Usage | None = None
        ttft_s: float | None = None
        stop_reason = "stop"
        async for event in self._client.stream(request):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
                self._bus.emit(Event("message_update", {"text": event.text}))
            elif isinstance(event, ToolCallStart):
                calls[event.call_id] = (event.name, [])
            elif isinstance(event, ToolCallDelta):
                entry = calls.get(event.call_id)
                if entry is not None:
                    entry[1].append(event.arguments_delta)
            elif isinstance(event, UsageReport):
                usage = event.usage
            elif isinstance(event, StreamEnd):
                ttft_s = event.ttft_s
                stop_reason = event.stop_reason
        if usage is not None:
            self._tokens += usage.prompt_tokens + usage.completion_tokens
        self._bus.emit(
            Event(
                "message_end",
                {
                    "stop_reason": "tool_calls" if calls else stop_reason,
                    "ttft_s": ttft_s,
                    "duration_s": self._clock() - stream_started,  # M7.2
                    "usage": None
                    if usage is None
                    else {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                    },
                },
            )
        )
        return _Round(
            text="".join(text_parts),
            calls=tuple(
                (call_id, name, "".join(chunks))
                for call_id, (name, chunks) in calls.items()
            ),
        )

    def _check_budget(self) -> None:
        """Loop-top guard: rounds and wall clock are checked before spending a model call."""
        agent = self._config.agent
        if self._rounds >= agent.max_rounds:
            raise _Budget("rounds")
        if self._clock() - self._started >= agent.wall_clock_s:
            raise _Budget("wall_clock")
        self._check_tokens()

    def _check_tokens(self) -> None:
        if self._tokens > self._config.agent.token_cap_per_turn:
            raise _Budget("tokens")

    def _summary(self, reason: str) -> str:
        """State summary on a budget stop: what ran, plan progress, what's left."""
        elapsed = self._clock() - self._started
        lines = [
            f"Stopped early: {reason} budget exceeded after {self._rounds} rounds "
            f"({elapsed:.0f}s)."
        ]
        plan = self._session.plan
        if plan is not None:
            done = sum(1 for step in plan.steps if step.status == "done")
            lines.append(f"Plan progress: {done}/{len(plan.steps)} steps done.")
            pending = plan.first_pending()
            if pending is not None:
                lines.append(f"Next: step {pending.index} — {pending.text}")
        return "\n".join(lines)


def _parse_reflection(text: str) -> Reflection:
    """Two-line verdict; anything unparseable reads as CONTINUE (nudge-capped)."""
    kind: Literal["done", "continue", "revise"] = "continue"
    reason = ""
    completed: tuple[int, ...] = ()
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("completed:"):
            completed = tuple(
                int(token) for token in re.findall(r"\d+", stripped.split(":", 1)[1])
            )
        elif lowered.startswith("verdict:"):
            body = stripped.split(":", 1)[1].strip()
            upper = body.upper()
            if upper.startswith("DONE"):
                kind = "done"
            elif upper.startswith("REVISE"):
                kind = "revise"
                reason = body[len("REVISE") :].strip()
    return Reflection(kind=kind, completed=completed, reason=reason)
