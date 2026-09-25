"""Session state and JSONL persistence.

A session is one task's conversation: its messages, its plan, and the routing
state (tier, phase, tool history, escalation flag) that carries across turns.
Every mutation appends a record to ``<session_dir>/<id>.jsonl`` so a session can
be resumed or inspected later; the loader skips torn or unknown lines so a
crash mid-append never bricks resume.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from workbench.core.protocols import ChatMessage, Phase, ToolCall
from workbench.logging_setup import get_logger

logger = get_logger("session")

PlanStepStatus = Literal["pending", "active", "done"]  # set by reflection, shown by the TUI

_PHASE_VALUES: tuple[str, ...] = ("explore", "implement", "verify", "none")


@dataclass
class PlanStep:
    index: int  # 1-based, matching how steps are numbered for the model
    text: str
    status: PlanStepStatus = "pending"


@dataclass
class Plan:
    steps: list[PlanStep]

    def all_done(self) -> bool:
        return all(step.status == "done" for step in self.steps)

    def first_pending(self) -> PlanStep | None:
        return next((step for step in self.steps if step.status != "done"), None)


@dataclass
class Session:
    """One task's state; mutators persist through the store they were created with."""

    id: str
    created_at: float = 0.0
    workspace: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    plan: Plan | None = None
    # Routing state carried across turns (the policy's per-task inputs).
    tier: str | None = None
    previous_intent: str | None = None
    phase: Phase = "none"
    recent_tools: tuple[str, ...] = ()  # two most recent, newest first
    consecutive_tool_errors: int = 0
    escalated: bool = False
    # /model pin: in-memory only (ruling M6.7) — the tier itself persists through
    # turn records; a resumed session re-pins explicitly.
    pinned_tier: str | None = None
    turn_count: int = 0
    _store: SessionStore | None = field(default=None, repr=False, compare=False)

    # --- lifecycle -----------------------------------------------------

    @classmethod
    def create(cls, *, store: SessionStore, workspace: str) -> Session:
        session = cls(
            id=uuid.uuid4().hex[:12],
            created_at=time.time(),
            workspace=workspace,
            _store=store,
        )
        store.append(
            session.id,
            {
                "type": "meta",
                "session_id": session.id,
                "created_at": session.created_at,
                "workspace": workspace,
            },
        )
        return session

    # --- mutations (persist immediately) -------------------------------

    def append_message(self, message: ChatMessage) -> None:
        self.messages.append(message)
        self._persist(
            {
                "type": "message",
                "role": message.role,
                "content": message.content,
                "tool_call_id": message.tool_call_id,
                "name": message.name,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in message.tool_calls
                ],
            }
        )

    def set_plan(self, plan: Plan | None) -> None:
        self.plan = plan
        steps: list[dict[str, object]] | None = None
        if plan is not None:
            steps = [
                {"index": step.index, "text": step.text, "status": step.status}
                for step in plan.steps
            ]
        self._persist({"type": "plan", "steps": steps})

    def note_tool_call(self, name: str) -> None:
        """Feed one executed tool name into the two-slot phase history (newest first)."""
        self.recent_tools = (name, *self.recent_tools)[:2]

    def note_tool_result(self, *, tool: str, is_error: bool) -> None:
        self.note_tool_call(tool)
        self.consecutive_tool_errors = 0 if not is_error else self.consecutive_tool_errors + 1

    def record_turn(
        self,
        *,
        user_message: str,
        outcome: str,
        rounds: int,
        duration_s: float,
        intent: str,
    ) -> None:
        """Close a turn: apply final routing state and snapshot it for replay."""
        self.previous_intent = intent
        self.turn_count += 1
        self._persist(
            {
                "type": "turn",
                "user_message": user_message,
                "outcome": outcome,
                "rounds": rounds,
                "duration_s": duration_s,
                "intent": intent,
                "tier": self.tier,
                "phase": self.phase,
                "recent_tools": list(self.recent_tools),
                "consecutive_tool_errors": self.consecutive_tool_errors,
                "escalated": self.escalated,
            }
        )

    def _persist(self, record: Mapping[str, object]) -> None:
        if self._store is not None:
            self._store.append(self.id, record)


class SessionStore:
    """Append-only JSONL store: one file per session, replayed on load."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory).expanduser()

    def path(self, session_id: str) -> Path:
        return self._directory / f"{session_id}.jsonl"

    def append(self, session_id: str, record: Mapping[str, object]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with self.path(session_id).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def load(self, session_id: str) -> Session:
        """Replay a session file; malformed and unknown lines are skipped, not fatal."""
        session = Session(id=session_id, created_at=0.0, _store=self)
        for raw_line in self.path(session_id).read_text(encoding="utf-8").splitlines():
            record = self._parse(raw_line)
            if record is None:
                continue
            self._apply(session, record)
        return session

    def list_ids(self) -> list[str]:
        if not self._directory.is_dir():
            return []
        return sorted(path.stem for path in self._directory.glob("*.jsonl"))

    @staticmethod
    def _parse(raw_line: str) -> dict[str, Any] | None:
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            logger.warning("skipping malformed session line: %.80s", raw_line)
            return None
        if not isinstance(record, dict):
            logger.warning("skipping non-object session line: %.80s", raw_line)
            return None
        return record

    @staticmethod
    def _apply(session: Session, record: dict[str, Any]) -> None:
        kind = record.get("type")
        if kind == "meta":
            session.created_at = float(record.get("created_at", session.created_at))
            session.workspace = str(record.get("workspace", session.workspace))
        elif kind == "message":
            session.messages.append(SessionStore._message(record))
        elif kind == "plan":
            session.plan = SessionStore._plan(record.get("steps"))
        elif kind == "turn":
            session.previous_intent = str(record.get("intent", ""))
            session.tier = record.get("tier")
            phase = record.get("phase", "none")
            session.phase = cast(Phase, phase) if phase in _PHASE_VALUES else "none"
            session.recent_tools = tuple(str(name) for name in record.get("recent_tools", ()))
            session.consecutive_tool_errors = int(record.get("consecutive_tool_errors", 0))
            session.escalated = bool(record.get("escalated", False))
            session.turn_count += 1
        else:
            logger.warning("skipping unknown session record type: %r", kind)

    @staticmethod
    def _message(record: dict[str, Any]) -> ChatMessage:
        return ChatMessage(
            role=record["role"],
            content=str(record.get("content", "")),
            tool_call_id=record.get("tool_call_id"),
            name=record.get("name"),
            tool_calls=tuple(
                ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"])
                for call in record.get("tool_calls", ())
            ),
        )

    @staticmethod
    def _plan(steps: Any) -> Plan | None:
        if not isinstance(steps, list):
            return None
        return Plan(
            steps=[
                PlanStep(
                    index=int(step["index"]), text=str(step["text"]), status=step["status"]
                )
                for step in steps
            ]
        )
