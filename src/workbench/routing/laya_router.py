"""LayaRouter: one batched laya forward pass per turn, resident checkpoint.

Falls back to the deterministic router when the checkpoint cannot load, when
inference errors, or when the weakest question's confidence drops below the
gate (plan §4.2). Offline is forced before laya/torch is imported (constraint 5).

The single inference call is synchronous (~50ms warm, once per turn at a
message boundary) — the event loop is not serving frames at that moment.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from workbench.config import RoutingConfig
from workbench.core.protocols import Classification, Router
from workbench.logging_setup import get_logger
from workbench.routing.questions import INTENT_CRITERIA, WORKBENCH_QUESTIONS

logger = get_logger("router")

_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_CHECKPOINTS: dict[str, tuple[str, str | None]] = {
    "laya": ("convaiinnovations/laya", None),
    "laya-multilingual": ("convaiinnovations/laya", "multilingual"),
    "laya-typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
}
_QUESTIONS = ("intent", "difficulty", "needs_tools")
_NEEDS_TOOLS_THRESHOLD = 0.5


def _force_offline() -> None:
    # huggingface_hub reads these at import time; set before laya is imported
    for name in _OFFLINE_ENV:
        os.environ[name] = "1"


def _load_checkpoint(checkpoint: str) -> Any:
    import laya  # heavy (torch): imported lazily, after the offline env is set

    repo, subfolder = _CHECKPOINTS.get(checkpoint, (checkpoint, None))
    return laya.load(repo, subfolder=subfolder)


class LayaRouter:
    """``Router`` backed by the laya checkpoint (lazy load, resident, offline)."""

    def __init__(
        self,
        fallback: Router,
        routing: RoutingConfig,
        *,
        loader: Callable[[], Any] | None = None,
    ) -> None:
        self._fallback = fallback
        self._gate = routing.policy.confidence_gate
        self._loader: Callable[[], Any] = loader or (lambda: _load_checkpoint(routing.checkpoint))
        self._agent: Any | None = None
        self._load_failed = False

    async def classify(self, message: str) -> Classification:
        agent = self._ensure_agent()
        if agent is None:
            return await self._fallback.classify(message)
        try:
            answers = agent.predict(message, WORKBENCH_QUESTIONS)["answers"]
            intent = answers["intent"]["choice"]
            if intent not in INTENT_CRITERIA:
                raise ValueError(f"unexpected intent {intent!r}")
            difficulty = float(answers["difficulty"]["score"])
            needs_tools = float(answers["needs_tools"]["noul"]) >= _NEEDS_TOOLS_THRESHOLD
            # weakest-link: trust the classification only if every question is
            # confident — a mushy difficulty must not drive the tier threshold
            confidence = min(
                float(answers[question]["answer_confidence"]) for question in _QUESTIONS
            )
        except Exception as exc:
            logger.warning("laya inference failed, using heuristic: %s", exc)
            return await self._fallback.classify(message)

        if confidence < self._gate:
            logger.debug(
                "laya confidence %.2f < gate %.2f, using heuristic",
                confidence,
                self._gate,
            )
            return await self._fallback.classify(message)

        return Classification(
            intent=intent,
            difficulty=difficulty,
            confidence=confidence,
            needs_tools=needs_tools,
            backend="laya",
        )

    def _ensure_agent(self) -> Any | None:
        if self._agent is not None:
            return self._agent
        if self._load_failed:
            return None  # permanent for this session: loading costs seconds
        _force_offline()
        try:
            self._agent = self._loader()
        except Exception as exc:
            self._load_failed = True
            logger.error("cannot load laya checkpoint, degraded to heuristic: %s", exc)
            return None
        return self._agent
