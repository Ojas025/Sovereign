"""T8.1 — local model store: search-path scan + index.json registry (PLAN §6.1)."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from workbench.models.store import Store, StoreEntry

_SHA_ABC = hashlib.sha256(b"abc").hexdigest()


def test_scan_finds_ggufs_across_search_paths_and_skips_missing_dirs(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "Tiny-7B-Q4_K_M.gguf").write_bytes(b"12345")
    (second / "Orphan-Q8_0.gguf").write_bytes(b"12")
    (first / "notes.txt").write_text("not a model")

    store = Store([str(first), str(tmp_path / "missing"), str(second), str(first)])
    entries = store.scan()

    assert [entry.name for entry in entries] == ["Orphan-Q8_0", "Tiny-7B-Q4_K_M"]
    by_name = {entry.name: entry for entry in entries}
    assert by_name["Tiny-7B-Q4_K_M"].size == 5
    assert by_name["Orphan-Q8_0"].size == 2
    assert by_name["Tiny-7B-Q4_K_M"].path == (first / "Tiny-7B-Q4_K_M.gguf")


def test_quant_and_family_parsed_from_the_filename(tmp_path: Path) -> None:
    cases = {
        "MiniCPM5-2B-Q4_K_M": ("MiniCPM5-2B", "Q4_K_M"),
        "Qwen2.5-Coder-7B-Instruct-Q4_K_M": ("Qwen2.5-Coder-7B-Instruct", "Q4_K_M"),
        "some-model-IQ2_XXS": ("some-model", "IQ2_XXS"),
        "llama-7b": ("llama-7b", ""),
    }
    for stem, (family, quant) in cases.items():
        entry = StoreEntry.for_path(tmp_path / f"{stem}.gguf")
        assert (entry.family, entry.quant) == (family, quant), stem


def test_register_computes_sha_and_persists_it_for_the_next_scan(tmp_path: Path) -> None:
    model = tmp_path / "Tiny-7B-Q4_K_M.gguf"
    model.write_bytes(b"abc")

    entry = Store([str(tmp_path)]).register(model)

    assert entry.sha == _SHA_ABC
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["entries"]["Tiny-7B-Q4_K_M.gguf"]["sha"] == _SHA_ABC

    rescanned = Store([str(tmp_path)]).scan()
    assert rescanned[0].sha == _SHA_ABC


def test_scan_ignores_a_stale_index_sha_when_the_file_changed(tmp_path: Path) -> None:
    model = tmp_path / "Tiny-7B-Q4_K_M.gguf"
    model.write_bytes(b"abc")
    store = Store([str(tmp_path)])
    store.register(model)

    model.write_bytes(b"abcabc")  # replaced after the checksum was recorded

    assert store.scan()[0].sha is None


def test_scan_survives_a_corrupt_index(tmp_path: Path) -> None:
    (tmp_path / "Tiny-7B-Q4_K_M.gguf").write_bytes(b"1")
    (tmp_path / "index.json").write_text("{not json", encoding="utf-8")

    entries = Store([str(tmp_path)]).scan()

    assert len(entries) == 1
    assert entries[0].sha is None


def test_models_list_joins_profiles_tiers_and_store_entries(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    (store_dir / "Tiny-7B-Q4_K_M.gguf").write_bytes(b"12345")
    (store_dir / "Orphan-Q8_0.gguf").write_bytes(b"12")
    tiny_path = store_dir / "Tiny-7B-Q4_K_M.gguf"
    toml = (
        f'[models]\nsearch_paths = ["{store_dir}"]\n\n'
        f"[models.profiles.tiny]\npath = \"{tiny_path}\"\n\n"
        f"[models.profiles.ghost]\npath = \"{tmp_path / "Ghost-Q4_K_M.gguf"}\"\n\n"
        "[routing.tiers]\nsmall = \"tiny\"\n"
    )
    (tmp_path / ".workbench.toml").write_text(toml, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "workbench", "models", "list"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    tiny_line = next(line for line in lines if line.startswith("tiny:"))
    assert "tier small" in tiny_line
    assert "Tiny-7B" in tiny_line and "Q4_K_M" in tiny_line
    ghost_line = next(line for line in lines if line.startswith("ghost:"))
    assert "tier -" in ghost_line and "not in store" in ghost_line
    orphan_line = next(line for line in lines if line.startswith("store:"))
    assert "Orphan-Q8_0.gguf" in orphan_line and "Q8_0" in orphan_line


def test_models_list_without_profiles_keeps_the_configure_hint(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "workbench", "models", "list"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "no model profiles configured" in result.stdout
