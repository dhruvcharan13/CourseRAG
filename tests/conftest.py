"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def dummy_config(tmp_path):
    """Pin the embedder to the dummy for CLI tests.

    The default is ``embedder = "auto"``, which selects the real local model when the
    ``[local]`` extra is installed. Tests that assert on dummy dims/model ids must not
    depend on whether it is installed, so this writes a ``config.toml`` into
    ``tmp_path`` — the directory those tests chdir into — before the test body runs.
    """
    path = tmp_path / "config.toml"
    path.write_text('embedder = "dummy"\n', encoding="utf-8")
    return path
