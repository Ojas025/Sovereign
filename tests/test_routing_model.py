"""Real laya checkpoint routing — local-only with GPU; CI runs `-m "not model"`.

Assertions are backend-agnostic: laya is primary, but the confidence gate may
legitimately hand a turn to the heuristic, so the *final* classification must
be sensible either way.
"""

import pytest

from workbench.config import RoutingConfig
from workbench.routing.heuristic import HeuristicRouter
from workbench.routing.laya_router import LayaRouter

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def real_router() -> LayaRouter:
    return LayaRouter(HeuristicRouter(), RoutingConfig())


async def test_edit_and_chat_classify_sensibly(real_router: LayaRouter) -> None:
    edit = await real_router.classify("edit utils.py so parse() returns None on empty input")
    assert edit.intent == "file_edit"
    assert edit.needs_tools is True
    assert 0.0 <= edit.difficulty < 2.0
    assert edit.backend in {"laya", "heuristic"}

    chat = await real_router.classify("hi there, how are you?")
    assert chat.intent == "chat_qa"
    assert chat.difficulty < 2.0
    assert 0.0 < chat.confidence <= 1.0


async def test_complex_refactor_scores_mid_difficulty(real_router: LayaRouter) -> None:
    result = await real_router.classify(
        "Refactor the entire authentication module into async, update all call "
        "sites across the codebase, then write a migration plan for every consumer"
    )

    assert result.difficulty >= 2.0
    assert result.intent in {"planning", "code_gen"}
