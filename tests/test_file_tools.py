"""File tools: read/write/edit semantics, caps, and jail enforcement."""

from pathlib import Path

from workbench.config import ToolsConfig
from workbench.core.protocols import ToolContext
from workbench.tools.files import EditTool, ReadTool, WriteTool


async def _yes(prompt: str) -> bool:
    return True


def ctx(workspace_root: Path) -> ToolContext:
    return ToolContext(workspace_root=workspace_root, confirm=_yes)


class TestRead:
    async def test_reads_line_numbered_content(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import os\nprint('hi')\n", encoding="utf-8")
        tool = ReadTool(ToolsConfig())

        result = await tool.execute({"path": "app.py"}, ctx(tmp_path))

        assert result.is_error is False
        assert result.content == "    1: import os\n    2: print('hi')"

    async def test_offset_starts_at_given_line(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")

        result = await ReadTool(ToolsConfig()).execute(
            {"path": "f.txt", "offset": 3}, ctx(tmp_path)
        )

        assert result.content == "    3: c\n    4: d"

    async def test_limit_caps_returned_lines(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")

        result = await ReadTool(ToolsConfig()).execute({"path": "f.txt", "limit": 2}, ctx(tmp_path))

        assert result.content == "    1: a\n    2: b"

    async def test_missing_file_reports_error(self, tmp_path: Path) -> None:
        result = await ReadTool(ToolsConfig()).execute({"path": "nope.txt"}, ctx(tmp_path))

        assert result.is_error is True
        assert "file not found" in result.content

    async def test_directory_reports_error(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()

        result = await ReadTool(ToolsConfig()).execute({"path": "sub"}, ctx(tmp_path))

        assert result.is_error is True
        assert "directory" in result.content

    async def test_binary_file_reports_error(self, tmp_path: Path) -> None:
        (tmp_path / "blob.bin").write_bytes(b"\x00\xff\xfe")

        result = await ReadTool(ToolsConfig()).execute({"path": "blob.bin"}, ctx(tmp_path))

        assert result.is_error is True
        assert "UTF-8" in result.content

    async def test_file_over_byte_cap_reports_error(self, tmp_path: Path) -> None:
        (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")

        result = await ReadTool(ToolsConfig(read_max_bytes=50)).execute(
            {"path": "big.txt"}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "too large" in result.content

    async def test_offset_beyond_end_of_file_reports_error(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("only line\n", encoding="utf-8")

        result = await ReadTool(ToolsConfig()).execute(
            {"path": "f.txt", "offset": 9}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "offset" in result.content

    async def test_empty_file_returns_empty_content(self, tmp_path: Path) -> None:
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")

        result = await ReadTool(ToolsConfig()).execute({"path": "empty.txt"}, ctx(tmp_path))

        assert result.is_error is False
        assert result.content == ""

    async def test_escape_attempt_is_blocked(self, tmp_path: Path) -> None:
        result = await ReadTool(ToolsConfig()).execute({"path": "../secrets.txt"}, ctx(tmp_path))

        assert result.is_error is True
        assert "blocked" in result.content

    async def test_non_string_path_reports_error(self, tmp_path: Path) -> None:
        result = await ReadTool(ToolsConfig()).execute({"path": 42}, ctx(tmp_path))

        assert result.is_error is True
        assert "path" in result.content

    async def test_bool_offset_reports_error(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\n", encoding="utf-8")

        result = await ReadTool(ToolsConfig()).execute(
            {"path": "f.txt", "offset": True}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "offset" in result.content

    async def test_read_root_files_are_readable(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        shared = tmp_path / "shared"
        shared.mkdir()
        (shared / "data.txt").write_text("shared data", encoding="utf-8")

        result = await ReadTool(ToolsConfig(read_roots=(str(shared),))).execute(
            {"path": str(shared / "data.txt")}, ctx(workspace)
        )

        assert result.is_error is False
        assert "shared data" in result.content


class TestWrite:
    async def test_creates_new_file(self, tmp_path: Path) -> None:
        tool = WriteTool(ToolsConfig())

        result = await tool.execute({"path": "new.txt", "content": "hello"}, ctx(tmp_path))

        assert result.is_error is False
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello"
        assert "5 bytes" in result.content

    async def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("old", encoding="utf-8")

        result = await WriteTool(ToolsConfig()).execute(
            {"path": "f.txt", "content": "new"}, ctx(tmp_path)
        )

        assert result.is_error is False
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "new"

    async def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        result = await WriteTool(ToolsConfig()).execute(
            {"path": "deep/nested/dir/f.txt", "content": "x"}, ctx(tmp_path)
        )

        assert result.is_error is False
        assert (tmp_path / "deep/nested/dir/f.txt").read_text(encoding="utf-8") == "x"

    async def test_empty_content_is_allowed(self, tmp_path: Path) -> None:
        result = await WriteTool(ToolsConfig()).execute(
            {"path": "e.txt", "content": ""}, ctx(tmp_path)
        )

        assert result.is_error is False
        assert (tmp_path / "e.txt").exists()

    async def test_content_over_cap_is_rejected_without_writing(self, tmp_path: Path) -> None:
        result = await WriteTool(ToolsConfig(write_max_bytes=4)).execute(
            {"path": "big.txt", "content": "12345"}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "too large" in result.content
        assert not (tmp_path / "big.txt").exists()

    async def test_escape_attempt_is_blocked_and_no_file_written(self, tmp_path: Path) -> None:
        result = await WriteTool(ToolsConfig()).execute(
            {"path": "../evil.txt", "content": "x"}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "blocked" in result.content
        assert not (tmp_path.parent / "evil.txt").exists()

    async def test_missing_content_reports_error(self, tmp_path: Path) -> None:
        result = await WriteTool(ToolsConfig()).execute({"path": "f.txt"}, ctx(tmp_path))

        assert result.is_error is True
        assert "content" in result.content

    async def test_missing_path_reports_error(self, tmp_path: Path) -> None:
        result = await WriteTool(ToolsConfig()).execute({"content": "x"}, ctx(tmp_path))

        assert result.is_error is True
        assert "path" in result.content


class TestEdit:
    async def test_replaces_unique_occurrence(self, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        tool = EditTool(ToolsConfig())

        result = await tool.execute(
            {"path": "f.py", "old_string": "x = 1", "new_string": "x = 42"}, ctx(tmp_path)
        )

        assert result.is_error is False
        assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 42\ny = 2\n"
        assert "1 occurrence" in result.content

    async def test_missing_old_string_reports_error(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("hello", encoding="utf-8")

        result = await EditTool(ToolsConfig()).execute(
            {"path": "f.txt", "old_string": "bye", "new_string": "x"}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "not found" in result.content
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "hello"

    async def test_ambiguous_match_without_replace_all_reports_error(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "f.txt").write_text("ab ab ab", encoding="utf-8")

        result = await EditTool(ToolsConfig()).execute(
            {"path": "f.txt", "old_string": "ab", "new_string": "z"}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "3 matches" in result.content
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "ab ab ab"

    async def test_replace_all_replaces_every_occurrence(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("ab ab ab", encoding="utf-8")

        result = await EditTool(ToolsConfig()).execute(
            {"path": "f.txt", "old_string": "ab", "new_string": "z", "replace_all": True},
            ctx(tmp_path),
        )

        assert result.is_error is False
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "z z z"

    async def test_empty_old_string_reports_error(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("content", encoding="utf-8")

        result = await EditTool(ToolsConfig()).execute(
            {"path": "f.txt", "old_string": "", "new_string": "x"}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "old_string" in result.content

    async def test_missing_file_reports_error(self, tmp_path: Path) -> None:
        result = await EditTool(ToolsConfig()).execute(
            {"path": "ghost.txt", "old_string": "a", "new_string": "b"}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "file not found" in result.content

    async def test_escape_attempt_is_blocked(self, tmp_path: Path) -> None:
        result = await EditTool(ToolsConfig()).execute(
            {"path": "../outside.txt", "old_string": "a", "new_string": "b"}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "blocked" in result.content

    async def test_edit_preserves_crlf_line_endings(self, tmp_path: Path) -> None:
        (tmp_path / "win.txt").write_bytes(b"a\r\nb\r\n")

        await EditTool(ToolsConfig()).execute(
            {"path": "win.txt", "old_string": "a", "new_string": "c"}, ctx(tmp_path)
        )

        assert (tmp_path / "win.txt").read_bytes() == b"c\r\nb\r\n"

    async def test_file_over_edit_cap_reports_error(self, tmp_path: Path) -> None:
        (tmp_path / "big.txt").write_text("y" * 100, encoding="utf-8")

        result = await EditTool(ToolsConfig(read_max_bytes=10)).execute(
            {"path": "big.txt", "old_string": "y", "new_string": "z"}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "too large" in result.content

    async def test_non_string_old_string_reports_error(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("x", encoding="utf-8")

        result = await EditTool(ToolsConfig()).execute(
            {"path": "f.txt", "old_string": 1, "new_string": "b"}, ctx(tmp_path)
        )

        assert result.is_error is True
        assert "old_string" in result.content
