"""Path jail: file tools resolve inside the workspace; every escape is rejected."""

from pathlib import Path

import pytest

from workbench.tools.jail import PathJail, PathJailError


def jail(tmp_path: Path, *read_roots: Path) -> PathJail:
    return PathJail(tmp_path, read_roots)


class TestInsideWorkspace:
    def test_relative_path_resolves_inside(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()

        resolved = jail(tmp_path).resolve_read("src/app.py")

        assert resolved == tmp_path / "src" / "app.py"

    def test_absolute_path_inside_workspace_allowed(self, tmp_path: Path) -> None:
        resolved = jail(tmp_path).resolve_read(str(tmp_path / "notes.txt"))

        assert resolved == tmp_path / "notes.txt"

    def test_symlink_pointing_inside_workspace_resolves(self, tmp_path: Path) -> None:
        (tmp_path / "real.txt").write_text("hi")
        (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")

        resolved = jail(tmp_path).resolve_read("link.txt")

        assert resolved == tmp_path / "real.txt"

    def test_workspace_root_itself_allowed(self, tmp_path: Path) -> None:
        assert jail(tmp_path).resolve_read(".") == tmp_path


class TestEscapesRejected:
    def test_parent_traversal_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PathJailError, match="outside"):
            jail(tmp_path).resolve_read("../outside.txt")

    def test_deep_traversal_to_etc_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PathJailError):
            jail(tmp_path).resolve_read("../../etc/passwd")

    def test_absolute_path_outside_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PathJailError, match="outside"):
            jail(tmp_path).resolve_read("/etc/passwd")

    def test_tilde_expanding_outside_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PathJailError, match="outside"):
            jail(tmp_path).resolve_read("~/.ssh/id_rsa")

    def test_symlink_file_pointing_outside_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "escape").symlink_to("/etc/passwd")

        with pytest.raises(PathJailError, match="outside"):
            jail(tmp_path).resolve_read("escape")

    def test_symlinked_directory_prefix_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "dirlink").symlink_to("/etc")

        with pytest.raises(PathJailError, match="outside"):
            jail(tmp_path).resolve_read("dirlink/passwd")

    def test_write_through_outside_symlink_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "escape").symlink_to("/tmp")

        with pytest.raises(PathJailError, match="outside"):
            jail(tmp_path).resolve_write("escape/payload.txt")

    def test_write_parent_traversal_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PathJailError, match="outside"):
            jail(tmp_path).resolve_write("../evil.txt")

    def test_write_absolute_outside_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PathJailError, match="outside"):
            jail(tmp_path).resolve_write("/etc/cron.d/evil")

    def test_null_byte_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PathJailError):
            jail(tmp_path).resolve_read("ok.txt\x00../../etc/passwd")

    def test_empty_path_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PathJailError):
            jail(tmp_path).resolve_read("")

    def test_write_to_read_root_rejected(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        read_jail = PathJail(tmp_path / "workspace", [shared])

        with pytest.raises(PathJailError, match="outside"):
            read_jail.resolve_write(shared / "file.txt")


class TestExtraReadRoots:
    def test_declared_read_root_allows_reads(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        shared = tmp_path / "shared"
        shared.mkdir()
        (shared / "data.txt").write_text("ok")

        resolved = PathJail(workspace, [shared]).resolve_read(str(shared / "data.txt"))

        assert resolved == shared / "data.txt"

    def test_paths_outside_all_roots_still_rejected(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        shared = tmp_path / "shared"
        shared.mkdir()

        with pytest.raises(PathJailError, match="outside"):
            PathJail(workspace, [shared]).resolve_read("/etc/shadow")
