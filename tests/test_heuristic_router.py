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


async def test_bug_fix_maps_to_file_edit_needing_tools() -> None:
    r1 = await classify(
        "fix the bug in src/workbench/tui/patch.py where trailing newlines duplicate"
    )
    assert r1.intent == "file_edit"
    assert r1.needs_tools is True

    r2 = await classify("fix the typo in README.md")
    assert r2.intent == "file_edit"
    assert r2.needs_tools is True

    r3 = await classify("fix the crash caused by null pointer exception")
    assert r3.intent == "file_edit"
    assert r3.needs_tools is True


async def test_modify_and_update_map_to_file_edit_needing_tools() -> None:
    r1 = await classify("modify the status bar to show model name")
    assert r1.intent == "file_edit"
    assert r1.needs_tools is True

    r2 = await classify("update .env with database url")
    assert r2.intent == "file_edit"
    assert r2.needs_tools is True

    r3 = await classify("tweak the prompt template in config.py")
    assert r3.intent == "file_edit"
    assert r3.needs_tools is True

    r4 = await classify("clean up unused imports")
    assert r4.intent == "file_edit"
    assert r4.needs_tools is True


async def test_document_and_pdf_generation_maps_to_code_gen_needing_tools() -> None:
    r1 = await classify("generate a pdf report summarizing quarterly sales")
    assert r1.intent == "code_gen"
    assert r1.needs_tools is True

    r2 = await classify("create a docx document for project roadmap")
    assert r2.intent == "code_gen"
    assert r2.needs_tools is True

    r3 = await classify("create a pdf invoice for client Acme")
    assert r3.intent == "code_gen"
    assert r3.needs_tools is True


async def test_write_code_prompts_map_to_code_gen() -> None:
    r1 = await classify("write code to calculate fibonacci numbers")
    assert r1.intent == "code_gen"
    assert r1.needs_tools is True

    r2 = await classify("build an express app with auth")
    assert r2.intent == "code_gen"
    assert r2.needs_tools is True

    r3 = await classify("develop a web scraper in python")
    assert r3.intent == "code_gen"
    assert r3.needs_tools is True

    r4 = await classify("implement binary search algorithm")
    assert r4.intent == "code_gen"
    assert r4.needs_tools is True


async def test_shell_commands_map_to_shell_task() -> None:
    r1 = await classify("run pytest tests/test_heuristic_router.py")
    assert r1.intent == "shell_task"
    assert r1.needs_tools is True

    r2 = await classify("git checkout -b feature/router-fix")
    assert r2.intent == "shell_task"
    assert r2.needs_tools is True

    r3 = await classify("docker restart redis")
    assert r3.intent == "shell_task"
    assert r3.needs_tools is True

    r4 = await classify("uv run mypy src tests")
    assert r4.intent == "shell_task"
    assert r4.needs_tools is True


async def test_search_and_analysis_queries() -> None:
    r1 = await classify("check the logs in /tmp")
    assert r1.intent == "search_analysis"

    r2 = await classify("inspect the database schema")
    assert r2.intent == "search_analysis"

    r3 = await classify("find all occurrences of route_decided")
    assert r3.intent == "search_analysis"

    r4 = await classify("show me where config is initialized")
    assert r4.intent == "search_analysis"


async def test_meta_inquiries() -> None:
    r1 = await classify("what is your name?")
    assert r1.intent == "meta"
    assert r1.needs_tools is False

    r2 = await classify("what can you do?")
    assert r2.intent == "meta"
    assert r2.needs_tools is False

    r3 = await classify("tell me about yourself")
    assert r3.intent == "meta"
    assert r3.needs_tools is False


async def test_difficulty_scaling_by_complexity_and_steps() -> None:
    # simple chat is trivial
    r_simple = await classify("what is the capital of France?")
    assert r_simple.intent == "chat_qa"
    assert r_simple.difficulty == 0.0

    # complexity keywords raise difficulty
    r_complex = await classify(
        "implement jwt auth with async concurrency and redis token blacklist"
    )
    assert r_complex.difficulty >= 1.0

    # numbered steps raise difficulty
    r_steps = await classify("1. update schema\n2. run migration\n3. verify tests")
    assert r_steps.difficulty >= 1.0


async def test_language_specific_code_generation() -> None:
    r1 = await classify("write a script in python to clean up csv data")
    assert r1.intent == "code_gen"
    assert r1.needs_tools is True

    r2 = await classify("code a simple web server using go")
    assert r2.intent == "code_gen"
    assert r2.needs_tools is True


async def test_planning_breakdown_queries() -> None:
    r1 = await classify("break this down into architectural phases")
    assert r1.intent == "planning"

    r2 = await classify("how should we structure the database migration?")
    assert r2.intent == "planning"


async def test_additional_meta_queries() -> None:
    r1 = await classify("which model is currently running?")
    assert r1.intent == "meta"
    assert r1.needs_tools is False

    r2 = await classify("what are your instructions?")
    assert r2.intent == "meta"
    assert r2.needs_tools is False

    r3 = await classify("are you an ai assistant?")
    assert r3.intent == "meta"
    assert r3.needs_tools is False


