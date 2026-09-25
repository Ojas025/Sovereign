"""Central logging: JSONL output with correlation ids and level filtering."""

import json
import logging
from pathlib import Path

import pytest

from workbench.logging_setup import configure_logging, get_logger, set_turn_id


def read_log_lines(log_dir: Path) -> list[dict[str, object]]:
    files = sorted(log_dir.glob("run-*.jsonl"))
    assert files, f"no log files written under {log_dir}"
    return [json.loads(line) for line in files[-1].read_text().splitlines() if line.strip()]


def test_jsonl_line_carries_run_id_logger_and_message(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path, level="INFO")

    get_logger("config").info("config loaded", extra={"layers": 4})

    records = read_log_lines(tmp_path)
    assert records[-1]["msg"] == "config loaded"
    assert records[-1]["logger"] == "workbench.config"
    assert records[-1]["level"] == "INFO"
    assert records[-1]["run_id"]
    assert records[-1]["layers"] == 4


def test_turn_id_appears_on_records_after_set_turn_id(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path, level="INFO")

    set_turn_id("turn-7")
    get_logger("agent").info("reflecting")

    assert read_log_lines(tmp_path)[-1]["turn_id"] == "turn-7"


def test_level_filtering_suppresses_lower_severity(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path, level="WARNING")

    logger = get_logger("server")
    logger.info("warm")
    logger.warning("hot")

    records = read_log_lines(tmp_path)
    assert [r["level"] for r in records] == ["WARNING"]


def test_reconfigure_does_not_duplicate_handlers(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path, level="INFO")
    configure_logging(log_dir=tmp_path, level="INFO")

    get_logger("agent").info("once")

    lines = [line for f in tmp_path.glob("run-*.jsonl") for line in f.read_text().splitlines()]
    assert len([line for line in lines if "once" in line]) == 1


def test_console_logging_is_opt_in(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    configure_logging(log_dir=tmp_path, level="INFO", console=False)

    with caplog.at_level(logging.INFO, logger="workbench.agent"):
        get_logger("agent").info("quiet")

    assert not [r for r in caplog.records if r.name == "workbench.agent"]
