"""Config loads with defaults (no file) and honors config.toml overrides."""

from __future__ import annotations

from pathlib import Path

from course_kb.config import ROOT_ENV_VAR, Config, load_config


def test_defaults_with_no_file(tmp_path, monkeypatch):
    # Run in an empty directory so no config.toml is discovered.
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg == Config()
    assert cfg.root == Path("course-kb")
    assert cfg.embedder == "auto"  # real local model if installed, else the dummy
    assert cfg.courses_dir == Path("course-kb") / "courses"
    assert cfg.archive_dir == Path("course-kb") / "archive"
    assert cfg.models_dir == Path("course-kb") / "models"


def test_explicit_path_overrides(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('root = "my-kb"\nembedder = "dummy"\nchunk_size = 42\n', encoding="utf-8")
    cfg = load_config(toml)
    assert cfg.root == Path("my-kb")
    assert cfg.embedder == "dummy"
    assert cfg.chunk_size == 42
    assert cfg.models_dir == Path("my-kb") / "models"  # follows root


def test_cache_dir_override_is_a_path(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text(
        'embedder = "minilm"\ncache_dir = "/tmp/kb-models"\nembed_batch_size = 8\n',
        encoding="utf-8",
    )
    cfg = load_config(toml)
    assert cfg.cache_dir == Path("/tmp/kb-models")
    assert cfg.models_dir == Path("/tmp/kb-models")  # explicit override wins over root
    assert cfg.embed_batch_size == 8


def test_unknown_keys_ignored(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('root = "x"\nfuture_option = true\n', encoding="utf-8")
    cfg = load_config(toml)
    assert cfg.root == Path("x")


# --------------------------------------------------------------------------- #
# COURSE_KB_ROOT
# --------------------------------------------------------------------------- #


def test_env_root_wins_over_the_config_file(tmp_path, monkeypatch):
    """The environment is the only handle a *caller* has on the root.

    A process started by something else — the MCP server, launched by an editor —
    inherits that program's working directory, so it can neither rely on the relative
    default nor count on a config.toml being next to it.
    """
    toml = tmp_path / "config.toml"
    toml.write_text('root = "from-file"\nembedder = "dummy"\n', encoding="utf-8")
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path / "from-env"))

    cfg = load_config(toml)

    assert cfg.root == tmp_path / "from-env"
    assert cfg.models_dir == tmp_path / "from-env" / "models"  # follows root
    assert cfg.embedder == "dummy"  # everything else still comes from the file


def test_env_root_is_resolved_to_an_absolute_path(tmp_path, monkeypatch):
    """A relative override would reintroduce the exact bug this variable exists to fix."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ROOT_ENV_VAR, "relative-kb")

    assert load_config().root == (tmp_path / "relative-kb").resolve()


def test_env_root_works_with_no_config_file_at_all(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path / "kb"))

    cfg = load_config()
    assert cfg.root == tmp_path / "kb"
    assert cfg.embedder == "auto"  # untouched default


def test_an_empty_env_root_is_ignored(tmp_path, monkeypatch):
    """An unset-looking variable must not resolve to the cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ROOT_ENV_VAR, "")

    assert load_config().root == Path("course-kb")
