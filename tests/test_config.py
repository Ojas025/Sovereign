"""Central config: layering precedence, defaults, validation, and type coercion."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from workbench.config import Config, ConfigError, load_config


def write_toml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


class TestDefaults:
    def test_no_files_yields_documented_defaults(self) -> None:
        config = load_config(home=Path("/nonexistent"), project_dir=Path("/nonexistent"))

        assert config.routing.backend == "laya"
        assert config.routing.policy.high_threshold == 2.0
        assert config.routing.policy.confidence_gate == 0.6
        assert config.routing.policy.default_tier == "mid"
        assert config.agent.max_rounds == 15
        assert config.agent.reflection_nudge_cap == 3
        assert config.server.idle_timeout_s == 300.0
        assert config.sandbox.backend == "bwrap"
        assert config.sandbox.bash_timeout_s == 60.0
        assert config.observability.metrics_port == 9600
        assert config.routing.tiers == {"small": "small", "mid": "mid"}


class TestLayering:
    def test_project_file_overrides_global_file(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        project = tmp_path / "project"
        write_toml(home / ".config/workbench/config.toml", "[agent]\nmax_rounds = 9\n")
        write_toml(project / ".workbench.toml", "[agent]\nmax_rounds = 21\n")

        config = load_config(home=home, project_dir=project)

        assert config.agent.max_rounds == 21

    def test_env_overrides_project_file(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        write_toml(project / ".workbench.toml", "[observability]\nmetrics_port = 9700\n")

        config = load_config(
            home=tmp_path / "home",
            project_dir=project,
            env={"WB_METRICS_PORT": "9801"},
        )

        assert config.observability.metrics_port == 9801

    def test_cli_overrides_env(self, tmp_path: Path) -> None:
        config = load_config(
            home=tmp_path / "home",
            project_dir=tmp_path / "project",
            env={"WB_LOG_LEVEL": "DEBUG"},
            cli_overrides={"observability.log_level": "ERROR"},
        )

        assert config.observability.log_level == "ERROR"

    def test_explicit_config_path_replaces_global_file(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.toml"
        write_toml(custom, "[server]\nbinary = '/opt/llama-serve'\n")

        config = load_config(
            home=tmp_path / "home",
            project_dir=tmp_path / "project",
            config_path=custom,
        )

        assert config.server.binary == "/opt/llama-serve"


class TestTypedSections:
    def test_profiles_and_tiers_roundtrip(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        write_toml(
            project / ".workbench.toml",
            """
            [models.profiles.coder7b]
            path = "~/models/Qwen2.5-Coder-7B.gguf"
            ctx_len = 16384
            tool_calling = "json"

            [routing.tiers]
            small = "coder7b"
            mid = "coder7b"
            """,
        )

        config = load_config(home=tmp_path / "home", project_dir=project)
        profile = config.models.profiles["coder7b"]

        assert profile.path == "~/models/Qwen2.5-Coder-7B.gguf"
        assert profile.ctx_len == 16384
        assert profile.tool_calling == "json"
        assert config.routing.tiers == {"small": "coder7b", "mid": "coder7b"}

    def test_env_values_are_coerced_to_default_types(self, tmp_path: Path) -> None:
        config = load_config(
            home=tmp_path / "home",
            project_dir=tmp_path / "project",
            env={"WB_IDLE_TIMEOUT_S": "45.5", "WB_VERBOSE": "true"},
        )

        assert config.server.idle_timeout_s == 45.5
        assert config.runtime.verbose is True


class TestValidation:
    def test_unknown_key_raises_config_error(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        write_toml(project / ".workbench.toml", "[agent]\nmax_rouds = 15\n")

        with pytest.raises(ConfigError, match="max_rouds"):
            load_config(home=tmp_path / "home", project_dir=project)

    def test_invalid_type_raises_config_error(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        write_toml(project / ".workbench.toml", "[agent]\nmax_rounds = 'many'\n")

        with pytest.raises(ConfigError, match="max_rounds"):
            load_config(home=tmp_path / "home", project_dir=project)

    def test_unreadable_toml_raises_config_error(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        write_toml(project / ".workbench.toml", "not [ valid toml")

        with pytest.raises(ConfigError, match=".workbench.toml"):
            load_config(home=tmp_path / "home", project_dir=project)


def test_loaded_config_is_frozen(tmp_path: Path) -> None:
    config = load_config(home=tmp_path / "home", project_dir=tmp_path / "project")

    assert isinstance(config, Config)
    with pytest.raises(FrozenInstanceError):
        config.agent.max_rounds = 1  # type: ignore[misc]
