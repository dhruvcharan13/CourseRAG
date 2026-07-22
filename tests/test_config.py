"""Config loads with defaults (no file) and honors config.toml overrides."""

from __future__ import annotations

from pathlib import Path

from course_kb.config import Config, load_config


def test_defaults_with_no_file(tmp_path, monkeypatch):
    # Run in an empty directory so no config.toml is discovered.
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg == Config()
    assert cfg.root == Path("course-kb")
    assert cfg.embedder == "dummy"
    assert cfg.courses_dir == Path("course-kb") / "courses"
    assert cfg.archive_dir == Path("course-kb") / "archive"


def test_explicit_path_overrides(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('root = "my-kb"\nembedder = "dummy"\nchunk_size = 42\n', encoding="utf-8")
    cfg = load_config(toml)
    assert cfg.root == Path("my-kb")
    assert cfg.chunk_size == 42


def test_unknown_keys_ignored(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('root = "x"\nfuture_option = true\n', encoding="utf-8")
    cfg = load_config(toml)
    assert cfg.root == Path("x")
