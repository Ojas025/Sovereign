"""Agent loop: plan approval, execute/reflect, budgets, escalation, event stream."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

from workbench.agent.loop import AgentLoop, TurnOutcome
from workbench.agent.planning import Approval, auto_approver
from workbench.config import Config, load_config
from workbench.core.events import Event, EventBus
from workbench.core.protocols import (
    ChatRequest,
    Classification,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
    ToolContext,
    ToolResult,
    Usage,
    UsageReport,
)
from workbench.core.session import Plan, PlanStep, Session, SessionStore
from workbench.llm.client import LLMError
from workbench.routing.policy import TierPolicy

DEFAULT_CLASSIFICATION = Classification(
    intent="code_gen",
    difficulty=1.0,
    confidence=0.9,
    needs_tools=False,
    backend="fake",
)


# --- fakes -----------------------------------------------------------------


class ScriptedClient:
    """One scripted stream per call; raises loudly when the script runs dry."""

    def __init__(self, scripts: list[list[StreamEvent] | BaseException]) -> None:
        self._scripts = list(scripts)
        self.requests: list[ChatRequest] = []

    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        if not self._scripts:
            raise AssertionError("unexpected LLM call: script exhausted")
        script = self._scripts.pop(0)
        if isinstance(script, BaseException):
            raise script

        async def generate() -> AsyncIterator[StreamEvent]:
            for event in script:
                yield event

        return generate()


class FakeTool:
    name = "echo"
    description = "Echo the arguments back"
    parameters = {"type": "object", "properties": {}}

    def __init__(
        self, *, reply: str = "ok", is_error: bool = False, crash: bool = False
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._reply = reply
        self._is_error = is_error
        self._crash = crash

    async def execute(
        self, arguments: dict[str, object], context: ToolContext
    ) -> ToolResult:
        self.calls.append(arguments)
        if self._crash:
            raise RuntimeError("tool exploded")
        return ToolResult(self._reply, is_error=self._is_error)


class FakeRouter:
    def __init__(self, classification: Classification = DEFAULT_CLASSIFICATION) -> None:
        self.classification = classification

    async def classify(self, message: str) -> Classification:
        return self.classification


class EventRecorder:
    def __init__(self, bus: EventBus) -> None:
        self.events: list[Event] = []
        bus.subscribe(None, self.events.append)

    def of(self, kind: str) -> list[Event]:
        return [event for event in self.events if event.kind == kind]

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]


async def _approve_all(prompt: str) -> bool:
    return True


def text_events(text: str) -> list[StreamEvent]:
    return [
        TextDelta(text),
        UsageReport(Usage(prompt_tokens=10, completion_tokens=5)),
        StreamEnd(stop_reason="stop", ttft_s=0.01),
    ]


def tool_events(call_id: str, tool: str, arguments: str) -> list[StreamEvent]:
    return [
        ToolCallStart(call_id=call_id, name=tool),
        ToolCallDelta(call_id=call_id, arguments_delta=arguments),
        UsageReport(Usage(prompt_tokens=10, completion_tokens=5)),
        StreamEnd(stop_reason="tool_calls"),
    ]


def make_config(tmp_path: Path, toml_text: str = "") -> Config:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    if toml_text:
        (project / ".workbench.toml").write_text(toml_text, encoding="utf-8")
    return load_config(home=tmp_path / "home", project_dir=project)


def build(
    tmp_path: Path,
    scripts: list[list[StreamEvent] | BaseException],
    **overrides: object,
) -> SimpleNamespace:
    config = make_config(tmp_path, str(overrides.pop("config_toml", "")))
    bus = EventBus()
    session = Session.create(
        store=SessionStore(tmp_path / "sessions"), workspace=str(tmp_path / "ws")
    )
    context = ToolContext(workspace_root=tmp_path / "ws", confirm=_approve_all)
    loop = AgentLoop(
        client=ScriptedClient(scripts),
        router=overrides.pop(
            "router", FakeRouter(overrides.pop("classification", DEFAULT_CLASSIFICATION))
        ),  # type: ignore[arg-type]
        policy=TierPolicy(config.routing),
        tools=overrides.pop("tools", {"echo": FakeTool()}),  # type: ignore[arg-type]
        session=session,
        bus=bus,
        config=config,
        context=context,
        approver=overrides.pop("approver", auto_approver),  # type: ignore[arg-type]
    )
    return SimpleNamespace(
        loop=loop,
        client=loop._client,  # noqa: SLF001 - test inspects what it wired
        recorder=EventRecorder(bus),
        session=session,
        config=config,
    )


# --- plain turns -------------------------------------------------------------


async def test_text_turn_completes_and_persists(tmp_path: Path) -> None:
    rig = build(tmp_path, [text_events("Answer")])

    outcome = await rig.loop.run_turn("hi")

    assert outcome == TurnOutcome(text="Answer", outcome="completed", rounds=1)
    assert [m.role for m in rig.session.messages] == ["user", "assistant"]
    assert rig.recorder.kinds()[0] == "turn_start"
    assert rig.recorder.kinds()[-1] == "turn_end"
    assert rig.recorder.of("route_decided") and rig.recorder.of("message_end")
    assert rig.recorder.of("turn_end")[0].data["outcome"] == "completed"
    assert rig.session.turn_count == 1
    # routed to the default small tier: no profile configured → tier name as model
    assert rig.client.requests[0].model == "small"
    assert rig.client.requests[0].messages[0].role == "system"


async def test_tool_round_appends_result_into_history(tmp_path: Path) -> None:
    tool = FakeTool(reply="echoed")
    rig = build(
        tmp_path,
        [tool_events("c1", "echo", '{"msg": "hi"}'), text_events("done")],
        tools={"echo": tool},
    )

    outcome = await rig.loop.run_turn("use the tool")

    assert outcome.outcome == "completed"
    assert tool.calls == [{"msg": "hi"}]
    assert [m.role for m in rig.session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_message = rig.session.messages[2]
    assert tool_message.content == "echoed"
    assert tool_message.tool_call_id == "c1"
    assert tool_message.name == "echo"
    start = rig.recorder.of("tool_execution_start")[0]
    assert start.data["tool"] == "echo"
    end = rig.recorder.of("tool_execution_end")[0]
    assert end.data["status"] == "ok"
    # second request carries system + user + assistant(tool_calls) + tool result
    roles = [m.role for m in rig.client.requests[1].messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert rig.client.requests[1].tools and rig.client.requests[1].tools[0].name == "echo"


async def test_unknown_tool_returns_feedback_and_counts_error(tmp_path: Path) -> None:
    rig = build(
        tmp_path, [tool_events("c1", "nope", "{}"), text_events("recovered")]
    )

    outcome = await rig.loop.run_turn("call something")

    assert outcome.outcome == "completed"
    tool_message = rig.session.messages[2]
    assert "unknown tool" in tool_message.content
    assert rig.recorder.of("tool_execution_end")[0].data["status"] == "error"
    assert rig.session.consecutive_tool_errors == 1


async def test_malformed_tool_arguments_never_reach_the_tool(tmp_path: Path) -> None:
    tool = FakeTool()
    rig = build(
        tmp_path,
        [tool_events("c1", "echo", "{not json"), text_events("ok")],
        tools={"echo": tool},
    )

    outcome = await rig.loop.run_turn("break it")

    assert outcome.outcome == "completed"
    assert tool.calls == []
    assert "invalid JSON" in rig.session.messages[2].content
    assert rig.session.consecutive_tool_errors == 1


async def test_crashing_tool_becomes_error_feedback_not_a_crash(tmp_path: Path) -> None:
    rig = build(
        tmp_path,
        [tool_events("c1", "echo", "{}"), text_events("recovered")],
        tools={"echo": FakeTool(crash=True)},
    )

    outcome = await rig.loop.run_turn("crash please")

    assert outcome.outcome == "completed"
    assert "tool 'echo' failed" in rig.session.messages[2].content
    error_events = rig.recorder.of("error")
    assert error_events and error_events[0].data["stage"] == "tool"


async def test_llm_failure_emits_error_and_fails_gracefully(tmp_path: Path) -> None:
    rig = build(tmp_path, [LLMError("connection refused")])

    outcome = await rig.loop.run_turn("hi")

    assert outcome.outcome == "error"
    assert "connection refused" in outcome.text
    assert rig.recorder.of("error")[0].data["stage"] == "llm"
    assert rig.recorder.of("turn_end")[0].data["outcome"] == "error"
    assert rig.session.turn_count == 1


# --- budgets -----------------------------------------------------------------


async def test_max_rounds_budget_is_graceful(tmp_path: Path) -> None:
    tool = FakeTool()
    scripts = [tool_events(f"c{i}", "echo", "{}") for i in range(5)]
    rig = build(
        tmp_path, scripts, tools={"echo": tool}, config_toml="[agent]\nmax_rounds = 3\n"
    )

    outcome = await rig.loop.run_turn("never stop")

    assert outcome.outcome == "budget"
    assert outcome.rounds == 3
    assert len(tool.calls) == 3
    budget = rig.recorder.of("budget_exceeded")[0]
    assert budget.data["reason"] == "rounds"
    assert "round" in outcome.text  # state summary, not a crash
    assert rig.recorder.of("turn_end")[0].data["outcome"] == "budget"


async def test_wall_clock_budget(tmp_path: Path) -> None:
    rig = build(tmp_path, [], config_toml="[agent]\nwall_clock_s = 0.0\n")

    outcome = await rig.loop.run_turn("slow")

    assert outcome.outcome == "budget"
    assert rig.recorder.of("budget_exceeded")[0].data["reason"] == "wall_clock"
    assert rig.client.requests == []  # checked before spending a round


async def test_token_cap_budget_stops_before_more_work(tmp_path: Path) -> None:
    tool = FakeTool()
    rig = build(
        tmp_path,
        [tool_events("c1", "echo", "{}"), tool_events("c2", "echo", "{}")],
        tools={"echo": tool},
        config_toml="[agent]\ntoken_cap_per_turn = 20\n",
    )

    outcome = await rig.loop.run_turn("token burner")

    assert outcome.outcome == "budget"
    assert rig.recorder.of("budget_exceeded")[0].data["reason"] == "tokens"
    assert len(tool.calls) == 1  # round two streamed (15+15 > 20) but its tools never ran
    assert len(rig.client.requests) == 2


# --- routing & escalation ------------------------------------------------------


async def test_route_decided_payload(tmp_path: Path) -> None:
    rig = build(tmp_path, [text_events("ok")])

    await rig.loop.run_turn("write a parser")

    decision = rig.recorder.of("route_decided")[0].data
    assert decision["tier"] == "small"
    assert decision["reason"] == "difficulty"
    assert decision["intent"] == "code_gen"
    assert decision["confidence"] == 0.9
    assert decision["model"] == "small"
    assert len(rig.recorder.of("route_decided")) == 1


async def test_consecutive_failures_escalate_once_and_switch_model(
    tmp_path: Path,
) -> None:
    failing = FakeTool(reply="boom", is_error=True)
    rig = build(
        tmp_path,
        [
            tool_events("c1", "echo", "{}"),
            tool_events("c2", "echo", "{}"),
            text_events("recovered"),
        ],
        tools={"echo": failing},
        config_toml=(
            "[models.profiles.small]\npath = '~/models/Tiny-Small.gguf'\n"
            "[models.profiles.mid]\npath = '~/models/Mid-Model.gguf'\n"
        ),
    )

    outcome = await rig.loop.run_turn("keep trying")

    assert outcome.outcome == "completed"
    assert [request.model for request in rig.client.requests] == [
        "Tiny-Small",
        "Tiny-Small",
        "Mid-Model",
    ]
    escalations = rig.recorder.of("route_escalated")
    assert len(escalations) == 1
    assert escalations[0].data == {
        "from_tier": "small",
        "to_tier": "mid",
        "failures": 2,
    }
    assert rig.session.escalated is True
    assert rig.session.tier == "mid"
    assert len(rig.recorder.of("route_decided")) == 1  # mid-turn escalation stays quiet


# --- plan phase -----------------------------------------------------------------


def _plan_events(steps: list[str]) -> list[StreamEvent]:
    return text_events(json.dumps({"steps": steps}))


async def test_plan_phase_happy_path(tmp_path: Path) -> None:
    rig = build(
        tmp_path,
        [
            _plan_events(["alpha", "beta"]),
            tool_events("c1", "echo", "{}"),
            text_events("All done."),
            text_events("completed: 1, 2\nverdict: DONE"),
        ],
        classification=Classification(
            intent="planning", difficulty=1.0, confidence=0.9,
            needs_tools=False, backend="fake",
        ),
    )

    outcome = await rig.loop.run_turn("plan something")

    assert outcome.outcome == "completed"
    assert outcome.text == "All done."
    proposals = rig.recorder.of("plan_proposed")
    assert proposals[0].data["steps"] == ["alpha", "beta"]
    assert rig.recorder.of("plan_approved")[0].data["attempt"] == 1
    assert rig.session.plan is not None
    assert [step.status for step in rig.session.plan.steps] == ["done", "done"]
    approved = [m for m in rig.session.messages if "Plan approved" in m.content]
    assert approved and "1. alpha" in approved[0].content
    # the plan prompt was an ad-hoc call: it never entered the persisted transcript
    assert not any("JSON object" in m.content for m in rig.session.messages)


async def test_plan_rejection_feeds_back_and_regenerates(tmp_path: Path) -> None:
    approvals = iter(
        [
            Approval(decision="rejected", feedback="be specific"),
            Approval(decision="approved"),
        ]
    )

    async def scripted_approver(plan: object, attempt: int) -> Approval:
        return next(approvals)

    rig = build(
        tmp_path,
        [
            _plan_events(["vague step"]),
            _plan_events(["open parser.py", "run pytest"]),
            text_events("done now"),
            text_events("completed: 1, 2\nverdict: DONE"),
        ],
        approver=scripted_approver,
        classification=Classification(
            intent="planning", difficulty=1.0, confidence=0.9,
            needs_tools=False, backend="fake",
        ),
    )

    outcome = await rig.loop.run_turn("do the thing")

    assert outcome.outcome == "completed"
    retry_messages = rig.client.requests[1].messages
    assert any("be specific" in m.content for m in retry_messages)
    rejected = rig.recorder.of("plan_rejected")[0]
    assert rejected.data["decision"] == "rejected"
    assert rig.recorder.of("plan_approved")[0].data["attempt"] == 2
    assert rig.session.plan is not None
    assert rig.session.plan.steps[0].text == "open parser.py"


async def test_plan_cancellation_skips_execution(tmp_path: Path) -> None:
    async def cancel(plan: object, attempt: int) -> Approval:
        return Approval(decision="cancelled")

    rig = build(
        tmp_path,
        [_plan_events(["a"])],
        approver=cancel,
        classification=Classification(
            intent="planning", difficulty=1.0, confidence=0.9,
            needs_tools=False, backend="fake",
        ),
    )

    outcome = await rig.loop.run_turn("no thanks")

    assert outcome.outcome == "cancelled"
    assert "cancel" in outcome.text.lower()
    assert len(rig.client.requests) == 1  # plan call only
    assert rig.recorder.of("plan_rejected")[0].data["decision"] == "cancelled"
    assert rig.recorder.of("turn_end")[0].data["outcome"] == "cancelled"


async def test_plan_retries_exhausted_cancels(tmp_path: Path) -> None:
    async def always_reject(plan: object, attempt: int) -> Approval:
        return Approval(decision="rejected", feedback="no")

    rig = build(
        tmp_path,
        [_plan_events(["a"]), _plan_events(["b"])],
        approver=always_reject,
        config_toml="[agent]\nplan_max_retries = 1\n",
        classification=Classification(
            intent="planning", difficulty=1.0, confidence=0.9,
            needs_tools=False, backend="fake",
        ),
    )

    outcome = await rig.loop.run_turn("try hard")

    assert outcome.outcome == "cancelled"
    assert len(rig.client.requests) == 2
    exhausted = rig.recorder.of("plan_rejected")[-1]
    assert exhausted.data["decision"] == "exhausted"
    assert exhausted.data["attempt"] == 2


async def test_unparseable_plan_spends_an_attempt(tmp_path: Path) -> None:
    rig = build(
        tmp_path,
        [
            text_events("I would just wing it"),
            _plan_events(["real step"]),
            text_events("finished"),
            text_events("completed: 1\nverdict: DONE"),
        ],
        config_toml="[agent]\nplan_max_retries = 1\n",
        classification=Classification(
            intent="planning", difficulty=1.0, confidence=0.9,
            needs_tools=False, backend="fake",
        ),
    )

    outcome = await rig.loop.run_turn("make a plan")

    assert outcome.outcome == "completed"
    assert rig.recorder.of("plan_rejected")[0].data["decision"] == "unparseable"
    retry = rig.client.requests[1].messages
    assert any("JSON" in m.content for m in retry)
    assert rig.session.plan is not None


async def test_high_difficulty_triggers_plan_without_planning_intent(
    tmp_path: Path,
) -> None:
    rig = build(
        tmp_path,
        [
            _plan_events(["step one"]),
            text_events("phew, done"),
            text_events("completed: 1\nverdict: DONE"),
        ],
        classification=Classification(
            intent="search_analysis", difficulty=2.5, confidence=0.9,
            needs_tools=False, backend="fake",
        ),
    )

    outcome = await rig.loop.run_turn("hard problem")

    assert outcome.outcome == "completed"
    assert rig.recorder.of("plan_proposed")
    assert rig.recorder.of("route_decided")[0].data["tier"] == "mid"
    assert rig.recorder.of("route_decided")[0].data["reason"] == "difficulty"


# --- reflection -------------------------------------------------------------------


async def _set_plan(rig: SimpleNamespace, *texts: str) -> None:
    rig.session.set_plan(
        Plan(steps=[PlanStep(index=i, text=t) for i, t in enumerate(texts, start=1)])
    )


async def test_reflection_done_completes_and_keeps_final_answer(
    tmp_path: Path,
) -> None:
    rig = build(
        tmp_path,
        [
            text_events("All finished — results below."),
            text_events("completed: 1, 2\nverdict: DONE"),
        ],
    )
    await _set_plan(rig, "first", "second")

    outcome = await rig.loop.run_turn("do it")

    assert outcome.outcome == "completed"
    assert outcome.text == "All finished — results below."
    reflections = rig.recorder.of("reflection")
    assert reflections[0].data["verdict"] == "done"
    assert reflections[0].data["completed"] == [1, 2]
    assert [step.status for step in rig.session.plan.steps] == ["done", "done"]  # type: ignore[union-attr]
    # the reflect exchange is scaffolding: never persisted
    assert not any("completed:" in m.content for m in rig.session.messages)
    assert [m.role for m in rig.session.messages] == ["user", "assistant"]


async def test_reflection_continue_nudges_until_cap(tmp_path: Path) -> None:
    scripts: list[list[StreamEvent] | BaseException] = []
    for _ in range(4):
        scripts += [text_events("stopped early"), text_events("completed: \nverdict: CONTINUE")]
    rig = build(
        tmp_path,
        scripts,
        config_toml="[agent]\nreflection_nudge_cap = 2\n",
    )
    await _set_plan(rig, "only step")

    outcome = await rig.loop.run_turn("nudge me")

    assert outcome.outcome == "budget"
    assert rig.recorder.of("budget_exceeded")[0].data["reason"] == "reflection"
    nudges = [
        m
        for m in rig.session.messages
        if m.role == "user" and "step 1" in m.content
    ]
    assert len(nudges) == 2
    assert all("Continue" in m.content for m in nudges)


async def test_done_with_unfinished_steps_downgrades_to_nudge(tmp_path: Path) -> None:
    rig = build(
        tmp_path,
        [
            text_events("half way there"),
            text_events("completed: 1\nverdict: DONE"),
            text_events("now really done"),
            text_events("completed: 1, 2\nverdict: DONE"),
        ],
    )
    await _set_plan(rig, "first", "second")

    outcome = await rig.loop.run_turn("two steps")

    assert outcome.outcome == "completed"
    assert outcome.text == "now really done"
    reflections = rig.recorder.of("reflection")
    assert reflections[0].data["verdict"] == "done"
    assert reflections[0].data["completed"] == [1]
    unfinished_nudge = [
        m for m in rig.session.messages if m.role == "user" and "unfinished" in m.content
    ]
    assert len(unfinished_nudge) == 1
    assert "2" in unfinished_nudge[0].content


async def test_reflection_revise_injects_correction(tmp_path: Path) -> None:
    rig = build(
        tmp_path,
        [
            text_events("draft attempt"),
            text_events("completed: \nverdict: REVISE wrong approach, add tests"),
            text_events("fixed properly"),
            text_events("completed: 1\nverdict: DONE"),
        ],
    )
    await _set_plan(rig, "the task")

    outcome = await rig.loop.run_turn("revise away")

    assert outcome.outcome == "completed"
    verdicts = [e.data["verdict"] for e in rig.recorder.of("reflection")]
    assert verdicts == ["revise", "done"]
    corrections = [
        m for m in rig.session.messages if m.role == "user" and "Correction" in m.content
    ]
    assert corrections and "wrong approach" in corrections[0].content


async def test_no_plan_means_no_reflection_calls(tmp_path: Path) -> None:
    rig = build(tmp_path, [text_events("simple answer")])

    await rig.loop.run_turn("quick question")

    assert len(rig.client.requests) == 1
    assert rig.recorder.of("reflection") == []


# --- json tool protocol ---------------------------------------------------------


_JSON_PROFILE_TOML = """
[routing.tiers]
small = "tiny"
mid = "tiny"

[models.profiles.tiny]
path = "~/models/Tiny-Chat.gguf"
tool_calling = "json"
"""


def _tool_block(name: str, arguments: str) -> str:
    return f'Calling now:\n```tool\n{{"name": "{name}", "arguments": {arguments}}}\n```'


async def test_json_profile_sends_no_native_tools(tmp_path: Path) -> None:
    rig = build(tmp_path, [text_events("hi")], config_toml=_JSON_PROFILE_TOML)

    await rig.loop.run_turn("hello")

    assert rig.client.requests[0].tools == ()
    assert "```tool" in rig.client.requests[0].messages[0].content


async def test_json_block_executes_tool_and_returns_via_user_message(
    tmp_path: Path,
) -> None:
    tool = FakeTool(reply="file content")
    rig = build(
        tmp_path,
        [text_events(_tool_block("read", '{"path": "a.txt"}')), text_events("Got it.")],
        tools={"echo": tool},
        config_toml=_JSON_PROFILE_TOML,
    )

    outcome = await rig.loop.run_turn("read a file")

    assert outcome.outcome == "completed"
    # FakeTool is named echo — the block said "read", so it is an unknown tool error
    assert rig.session.messages[1].role == "assistant"
    result_message = rig.session.messages[2]
    assert result_message.role == "user"  # json protocol: results ride as plain text
    assert "Tool 'read' returned:" in result_message.content
    assert "unknown tool" in result_message.content
    assert rig.recorder.of("tool_execution_start")[0].data["tool"] == "read"


async def test_json_block_executes_registered_tool(tmp_path: Path) -> None:
    tool = FakeTool(reply="content of a.txt")
    rig = build(
        tmp_path,
        [text_events(_tool_block("echo", '{"path": "a.txt"}')), text_events("Done.")],
        tools={"echo": tool},
        config_toml=_JSON_PROFILE_TOML,
    )

    await rig.loop.run_turn("read via echo")

    assert tool.calls == [{"path": "a.txt"}]
    result_message = rig.session.messages[2]
    assert result_message.role == "user"
    assert "content of a.txt" in result_message.content


async def test_malformed_json_block_feeds_error_back(tmp_path: Path) -> None:
    rig = build(
        tmp_path,
        [
            text_events('```tool\n{"name": "echo", "arguments"\n```'),
            text_events("retrying, got it"),
        ],
        config_toml=_JSON_PROFILE_TOML,
    )

    outcome = await rig.loop.run_turn("break the protocol")

    assert outcome.outcome == "completed"
    feedback = rig.session.messages[2]
    assert feedback.role == "user"
    assert "malformed" in feedback.content
    assert rig.recorder.of("tool_execution_end")[0].data["status"] == "error"
    assert rig.session.consecutive_tool_errors == 1


async def test_json_text_without_blocks_completes_as_plain_answer(
    tmp_path: Path,
) -> None:
    rig = build(tmp_path, [text_events("no tools needed")], config_toml=_JSON_PROFILE_TOML)

    outcome = await rig.loop.run_turn("just answer")

    assert outcome.outcome == "completed"
    assert outcome.text == "no tools needed"
    assert rig.recorder.of("tool_execution_start") == []


# --- persistence handoff -----------------------------------------------------------


async def test_completed_turn_is_replayable_from_store(tmp_path: Path) -> None:
    rig = build(tmp_path, [text_events("persisted answer")])

    await rig.loop.run_turn("remember me")

    store = rig.session._store  # noqa: SLF001 - replay what the loop wrote
    loaded = store.load(rig.session.id)
    assert loaded.turn_count == 1
    assert loaded.tier == "small"
    assert loaded.previous_intent == "code_gen"
    assert [m.content for m in loaded.messages] == ["remember me", "persisted answer"]


# --- tool output in events (M6 display) ------------------------------------------


async def test_tool_execution_end_carries_output_for_display(tmp_path: Path) -> None:
    """The TUI renders what a tool returned: the event carries the result content."""
    tool = FakeTool(reply="wrote hello.txt")
    rig = build(
        tmp_path, [tool_events("c1", "echo", "{}"), text_events("done")], tools={"echo": tool}
    )

    await rig.loop.run_turn("use the tool")

    end = rig.recorder.of("tool_execution_end")[0]
    assert end.data["output"] == "wrote hello.txt"


async def test_tool_execution_end_output_is_truncated_for_events(tmp_path: Path) -> None:
    """A huge result must not flood the NDJSON stream: events cap the display size."""
    tool = FakeTool(reply="x" * 5000)
    rig = build(
        tmp_path, [tool_events("c1", "echo", "{}"), text_events("done")], tools={"echo": tool}
    )

    await rig.loop.run_turn("use the tool")

    output = rig.recorder.of("tool_execution_end")[0].data["output"]
    assert isinstance(output, str)
    assert output.startswith("x" * 100)
    assert output.endswith("…")
    assert len(output) <= 4001  # capped display size + the ellipsis


async def test_malformed_json_block_event_reports_the_parse_error(tmp_path: Path) -> None:
    """Even the parse-failure pseudo-call explains itself through the event output."""
    rig = build(
        tmp_path,
        [
            text_events('```tool\n{"name": "echo", "arguments"\n```'),
            text_events("retrying"),
        ],
        config_toml=_JSON_PROFILE_TOML,
    )

    await rig.loop.run_turn("break the protocol")

    end = rig.recorder.of("tool_execution_end")[0]
    assert end.data["status"] == "error"
    assert "malformed" in str(end.data["output"])


# --- /model pin (M6) ----------------------------------------------------------------


async def test_pinned_tier_bypasses_policy_and_reports_pinned(tmp_path: Path) -> None:
    """`/model mid` pins the session: routing stays advisory until unpinned."""
    rig = build(tmp_path, [text_events("pinned answer")])
    rig.session.pinned_tier = "mid"
    rig.session.tier = "mid"

    await rig.loop.run_turn("hello")

    decided = rig.recorder.of("route_decided")[0]
    assert decided.data["tier"] == "mid"
    assert decided.data["reason"] == "pinned"
    assert rig.session.tier == "mid"


async def test_plan_asking_user_yields_turn(tmp_path: Path) -> None:
    question = "Please specify the specific functionality or deployment you need."
    rig = build(
        tmp_path,
        [
            text_events(question),
            text_events("completed: 1\nverdict: CONTINUE"),
        ],
    )
    await _set_plan(rig, "step 1", "step 2")

    outcome = await rig.loop.run_turn("start k8s setup")

    assert outcome.outcome == "completed"
    assert outcome.text == question
    nudges = [m for m in rig.session.messages if m.role == "user" and "Continue" in m.content]
    assert len(nudges) == 0


async def test_plan_wait_user_verdict_yields_turn(tmp_path: Path) -> None:
    rig = build(
        tmp_path,
        [
            text_events("Waiting for clarification on cluster credentials."),
            text_events("completed: 1\nverdict: WAIT_USER needing credentials"),
        ],
    )
    await _set_plan(rig, "step 1", "step 2")

    outcome = await rig.loop.run_turn("run deployment")

    assert outcome.outcome == "completed"
    assert "Waiting for clarification" in outcome.text
    reflections = rig.recorder.of("reflection")
    assert reflections[0].data["verdict"] == "wait_user"
    nudges = [m for m in rig.session.messages if m.role == "user" and "Continue" in m.content]
    assert len(nudges) == 0


async def test_reflection_tokens_do_not_leak_to_transcript_stream(tmp_path: Path) -> None:
    rig = build(
        tmp_path,
        [
            text_events("Finished step one."),
            text_events("completed: 1, 2\nverdict: DONE"),
        ],
    )
    await _set_plan(rig, "step 1", "step 2")

    await rig.loop.run_turn("proceed")

    message_updates = rig.recorder.of("message_update")
    for ev in message_updates:
        assert "verdict:" not in ev.data.get("text", "")
        assert "completed:" not in ev.data.get("text", "")


async def test_user_prompt_with_pii_is_redacted_and_emits_event(tmp_path: Path) -> None:
    from workbench.routing.heuristic import HeuristicRouter

    rig = build(
        tmp_path,
        [text_events("Acknowledged.")],
        router=HeuristicRouter(),
    )

    await rig.loop.run_turn("My email is secret@example.com and phone is +1-555-123-4567")

    pii_events = rig.recorder.of("pii_redacted")
    assert len(pii_events) == 1
    assert pii_events[0].data["counts"]["email"] == 1
    assert pii_events[0].data["counts"]["phone"] == 1
    user_msg = rig.session.messages[0]
    assert user_msg.role == "user"
    assert "secret@example.com" not in user_msg.content
    assert "[EMAIL]" in user_msg.content
    assert "[PHONE]" in user_msg.content

