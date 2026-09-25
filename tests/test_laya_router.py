"""LayaRouter: resident checkpoint, weakest-link confidence gate, heuristic fallback."""

import os

import pytest

from workbench.config import RoutingConfig
from workbench.core.protocols import Classification
from workbench.routing.laya_router import LayaRouter
from workbench.routing.questions import INTENT_CRITERIA, WORKBENCH_QUESTIONS


def answers(
    intent: str = "file_edit",
    difficulty: float = 1.5,
    needs_tools_p: float = 0.9,
    intent_conf: float = 0.9,
    difficulty_conf: float = 0.9,
) -> dict:
    """Canned predict() answers mirroring laya's real response shape."""
    return {
        "intent": {
            "type": "choice",
            "choice": intent,
            "probabilities": {intent: intent_conf},
            "confidence": intent_conf,
            "answer_confidence": intent_conf,
        },
        "difficulty": {
            "type": "score",
            "score": difficulty,
            "probabilities": {"1": difficulty_conf},
            "confidence": difficulty_conf,
            "answer_confidence": difficulty_conf,
        },
        "needs_tools": {
            "type": "noul",
            "noul": needs_tools_p,
            "confidence": 0.95,
            "answer_confidence": 0.95,
        },
    }


class FakeAgent:
    """Stands in for laya's Agent.predict — canned payload or a raised error."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self.payload = payload if payload is not None else {"answers": answers()}
        self.error = error
        self.calls = 0

    def predict(self, state: str, questions: dict) -> dict:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.payload


class FakeFallback:
    def __init__(self) -> None:
        self.calls = 0

    async def classify(self, message: str) -> Classification:
        self.calls += 1
        return Classification(
            intent="chat_qa", difficulty=0.0, confidence=0.7, needs_tools=False,
            backend="heuristic",
        )


def router_with(
    agent: FakeAgent, routing: RoutingConfig | None = None
) -> tuple[LayaRouter, FakeFallback]:
    fallback = FakeFallback()
    router = LayaRouter(fallback, routing or RoutingConfig(), loader=lambda: agent)
    return router, fallback


async def test_confident_answers_map_to_laya_classification() -> None:
    router, fallback = router_with(FakeAgent())

    result = await router.classify("edit utils.py")

    assert result == Classification(
        intent="file_edit",
        difficulty=1.5,
        confidence=0.9,
        needs_tools=True,
        backend="laya",
    )
    assert fallback.calls == 0


async def test_weakest_question_confidence_gates_to_fallback() -> None:
    """Gate reads min(answer_confidence) — a mushy difficulty must not route."""
    agent = FakeAgent(payload={"answers": answers(difficulty_conf=0.3)})
    router, fallback = router_with(agent)

    result = await router.classify("do something")

    assert result.backend == "heuristic"
    assert fallback.calls == 1


async def test_inference_error_falls_back_every_turn() -> None:
    agent = FakeAgent(error=RuntimeError("cuda oom"))
    router, fallback = router_with(agent)

    await router.classify("first")
    await router.classify("second")

    assert agent.calls == 2  # inference errors are retried, not cached
    assert fallback.calls == 2


async def test_load_failure_is_cached_and_falls_back() -> None:
    loads = {"count": 0}

    def broken_loader() -> FakeAgent:
        loads["count"] += 1
        raise RuntimeError("no checkpoint on disk")

    fallback = FakeFallback()
    router = LayaRouter(fallback, RoutingConfig(), loader=broken_loader)

    await router.classify("first")
    await router.classify("second")

    assert loads["count"] == 1  # loading costs seconds — never retried
    assert fallback.calls == 2


async def test_offline_env_is_forced_before_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    router, _ = router_with(FakeAgent())

    await router.classify("warm the load path")

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


async def test_unexpected_intent_falls_back() -> None:
    payload = {"answers": answers(intent="warp_drive")}
    router, fallback = router_with(FakeAgent(payload=payload))

    result = await router.classify("anything")

    assert result.backend == "heuristic"
    assert fallback.calls == 1


async def test_needs_tools_follows_noul_probability() -> None:
    low, _ = router_with(FakeAgent(payload={"answers": answers(needs_tools_p=0.4)}))
    high, _ = router_with(FakeAgent(payload={"answers": answers(needs_tools_p=0.6)}))

    assert (await low.classify("x")).needs_tools is False
    assert (await high.classify("x")).needs_tools is True


def test_workbench_questions_shape() -> None:
    assert set(WORKBENCH_QUESTIONS) == {"intent", "difficulty", "needs_tools"}
    assert {name: q["type"] for name, q in WORKBENCH_QUESTIONS.items()} == {
        "intent": "choice",
        "difficulty": "score",
        "needs_tools": "noul",
    }
    assert set(INTENT_CRITERIA) == {
        "chat_qa",
        "file_edit",
        "code_gen",
        "shell_task",
        "search_analysis",
        "planning",
        "meta",
    }
    assert len(WORKBENCH_QUESTIONS["difficulty"]["criteria"]) == 4  # ordinal 0-3


def test_load_checkpoint_filters_temperature_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types
    import warnings

    from workbench.routing import laya_router

    class FakeLayaModule(types.ModuleType):
        def load(self, repo: str, subfolder: str | None = None) -> object:
            warnings.warn(
                "laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]; "
                "using choice:11 +=0.10058280825614929 -> 0.5. Treat confidence from the affected "
                "entries as uncalibrated.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "loaded_agent"

    monkeypatch.setitem(sys.modules, "laya", FakeLayaModule("laya"))
    with warnings.catch_warnings(record=True) as recorded:
        agent = laya_router._load_checkpoint("convaiinnovations/laya")
        assert agent == "loaded_agent"
        matching = [w for w in recorded if "invalid temperatures" in str(w.message)]
        assert len(matching) == 0
