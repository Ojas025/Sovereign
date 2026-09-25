"""HeuristicRouter: deterministic length/keyword/code-fence fallback backend."""

from workbench.core.protocols import Classification
from workbench.routing.heuristic import HeuristicRouter


async def classify(message: str) -> Classification:
    return await HeuristicRouter().classify(message)


async def test_greeting_is_confident_low_difficulty_chat() -> None:
    result = await classify("hi there, how are you?")

    assert result.intent == "chat_qa"
    assert result.difficulty == 0.0
    assert result.needs_tools is False
    assert result.confidence >= 0.6
    assert result.backend == "heuristic"


async def test_edit_request_maps_to_file_edit_needing_tools() -> None:
    result = await classify("edit utils.py so parse() returns None on empty input")

    assert result.intent == "file_edit"
    assert result.needs_tools is True
    assert result.difficulty == 0.0


async def test_writing_script_maps_to_code_gen() -> None:
    result = await classify("write a python script that downloads and processes the CSV")

    assert result.intent == "code_gen"
    assert result.needs_tools is True
    assert result.difficulty == 0.0


async def test_commands_map_to_shell_task() -> None:
    assert (await classify("run git status and show me the output")).intent == "shell_task"
    assert (await classify("install the dependencies with uv")).intent == "shell_task"


async def test_explain_question_maps_to_search_analysis() -> None:
    result = await classify("explain why the tests fail on main")

    assert result.intent == "search_analysis"
    assert result.needs_tools is False
    assert result.difficulty == 0.0


async def test_plan_request_maps_to_planning() -> None:
    result = await classify("outline a plan for migrating the build to uv")

    assert result.intent == "planning"
    assert result.difficulty == 1.0  # complexity keyword "migrating"


async def test_meta_question_maps_to_meta() -> None:
    result = await classify("who are you and which model are you running?")

    assert result.intent == "meta"
    assert result.needs_tools is False


async def test_bare_file_path_maps_to_file_edit_needing_tools() -> None:
    result = await classify("src/parser.py has an error?")

    assert result.intent == "file_edit"
    assert result.needs_tools is True


async def test_long_multistep_fenced_request_scores_max_difficulty() -> None:
    message = (
        "Refactor the entire authentication module into async and update all "
        "call sites across the codebase. Then apply the migration below and "
        "verify every caller compiles cleanly before finishing the change:\n\n"
        "```python\nasync def login(user): ...\n```\n"
        "run the full suite afterwards too please"
    )

    result = await classify(message)

    assert result.difficulty == 3.0  # long + multi-step + complexity marker


async def test_short_vague_request_falls_below_confidence_gate() -> None:
    result = await classify("do the thing")

    assert result.intent == "chat_qa"
    assert result.confidence < 0.6  # policy will default-to-larger on this
    assert result.needs_tools is False
