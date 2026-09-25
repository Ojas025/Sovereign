"""Session model: messages, plan, and routing state with JSONL persistence."""

from workbench.config import load_config
from workbench.core.protocols import ChatMessage, ToolCall
from workbench.core.session import Plan, PlanStep, Session, SessionStore

# --- state defaults ------------------------------------------------------


def test_new_session_is_empty() -> None:
    session = Session(id="abc", created_at=1.0)

    assert session.messages == []
    assert session.plan is None
    assert session.tier is None
    assert session.previous_intent is None
    assert session.phase == "none"
    assert session.recent_tools == ()
    assert session.consecutive_tool_errors == 0
    assert session.escalated is False
    assert session.turn_count == 0


def test_recent_tools_keeps_two_newest_first() -> None:
    session = Session(id="abc", created_at=1.0)

    for tool in ("read", "read", "bash", "write"):
        session.note_tool_call(tool)

    assert session.recent_tools == ("write", "bash")


def test_tool_errors_reset_on_success_and_cap_escalation_state() -> None:
    session = Session(id="abc", created_at=1.0)

    session.note_tool_result(tool="bash", is_error=True)
    session.note_tool_result(tool="bash", is_error=True)
    assert session.consecutive_tool_errors == 2

    session.note_tool_result(tool="read", is_error=False)
    assert session.consecutive_tool_errors == 0


def test_plan_tracks_step_statuses() -> None:
    plan = Plan(steps=[PlanStep(index=1, text="first"), PlanStep(index=2, text="second")])

    assert plan.all_done() is False
    assert plan.first_pending() is plan.steps[0]

    plan.steps[0].status = "done"
    assert plan.first_pending() is plan.steps[1]

    plan.steps[1].status = "done"
    assert plan.all_done() is True
    assert plan.first_pending() is None


# --- persistence ---------------------------------------------------------


def test_create_writes_meta_record(tmp_path) -> None:
    store = SessionStore(tmp_path)

    Session.create(store=store, workspace="/ws")

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert '"type": "meta"' in files[0].read_text(encoding="utf-8")


def test_append_message_persists_immediately(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(store=store, workspace="/ws")

    session.append_message(ChatMessage(role="user", content="hello"))

    lines = (store.path(session.id)).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # meta + message
    assert '"content": "hello"' in lines[1]


def test_tool_calls_survive_serialization(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(store=store, workspace="/ws")
    session.append_message(
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(ToolCall(id="c1", name="bash", arguments='{"command": "ls"}'),),
        )
    )

    loaded = store.load(session.id)

    assert loaded.messages[-1].tool_calls[0].name == "bash"
    assert loaded.messages[-1].tool_calls[0].arguments == '{"command": "ls"}'


def test_plan_snapshot_persists_status_changes(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(store=store, workspace="/ws")
    session.set_plan(Plan(steps=[PlanStep(index=1, text="do the thing")]))
    session.set_plan(session.plan)  # status change re-snapshot

    loaded = store.load(session.id)

    assert loaded.plan is not None
    assert [step.text for step in loaded.plan.steps] == ["do the thing"]
    assert loaded.plan.steps[0].status == "pending"


def test_turn_record_restores_routing_state(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(store=store, workspace="/ws")
    session.tier = "mid"
    session.previous_intent = "code_gen"
    session.phase = "implement"
    session.recent_tools = ("write", "read")
    session.consecutive_tool_errors = 1
    session.escalated = True
    session.record_turn(
        user_message="build it",
        outcome="completed",
        rounds=4,
        duration_s=12.5,
        intent="code_gen",
    )

    loaded = store.load(session.id)

    assert loaded.turn_count == 1
    assert loaded.tier == "mid"
    assert loaded.previous_intent == "code_gen"
    assert loaded.phase == "implement"
    assert loaded.recent_tools == ("write", "read")
    assert loaded.consecutive_tool_errors == 1
    assert loaded.escalated is True


def test_load_roundtrip_preserves_message_order(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(store=store, workspace="/ws")
    session.append_message(ChatMessage(role="user", content="one"))
    session.append_message(ChatMessage(role="assistant", content="two"))
    session.append_message(ChatMessage(role="user", content="three"))

    loaded = store.load(session.id)

    assert [message.content for message in loaded.messages] == ["one", "two", "three"]
    assert [message.role for message in loaded.messages] == ["user", "assistant", "user"]


def test_load_tolerates_torn_final_line(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(store=store, workspace="/ws")
    session.append_message(ChatMessage(role="user", content="fine"))
    with store.path(session.id).open("a", encoding="utf-8") as handle:
        handle.write('{"type": "message", "role": "assist')  # torn append

    loaded = store.load(session.id)

    assert [message.content for message in loaded.messages] == ["fine"]


def test_load_ignores_unknown_record_types(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(store=store, workspace="/ws")
    with store.path(session.id).open("a", encoding="utf-8") as handle:
        handle.write('{"type": "future_kind", "payload": 1}\n')

    loaded = store.load(session.id)  # must not raise

    assert loaded.id == session.id


def test_list_ids_returns_persisted_sessions(tmp_path) -> None:
    store = SessionStore(tmp_path)
    first = Session.create(store=store, workspace="/ws")
    second = Session.create(store=store, workspace="/ws")

    assert sorted(store.list_ids()) == sorted([first.id, second.id])


def test_session_dir_defaults_and_overrides(tmp_path) -> None:
    config = load_config(home=tmp_path / "home", project_dir=tmp_path / "project")
    assert config.agent.session_dir == "~/.local/state/workbench/sessions"

    project = tmp_path / "project"
    project.mkdir()
    (project / ".workbench.toml").write_text(
        '[agent]\nsession_dir = "/tmp/sessions"\n', encoding="utf-8"
    )
    reloaded = load_config(home=tmp_path / "home", project_dir=project)
    assert reloaded.agent.session_dir == "/tmp/sessions"
