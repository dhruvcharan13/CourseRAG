"""The query/passage asymmetry: bge queries get a prefix, passages never do.

bge was trained so that queries carry a retrieval instruction and passages do not.
Reproducing that at search time is not cosmetic — skipping the prefix puts queries in a
subtly different place than the passages they should match. These tests assert both
halves: the prefix reaches queries, and it never reaches passages.
"""

from __future__ import annotations

import json

import pytest

from courserag.cli import main
from courserag.embedding.dummy import DummyEmbedder
from courserag.embedding.sentence_transformer import (
    DEFAULT_MODEL_ID,
    MINILM_MODEL_ID,
    SentenceTransformerEmbedder,
    is_available,
)
from courserag.retrieval import embed_query

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


# --------------------------------------------------------------------------- #
# Offline: which code path calls which method
# --------------------------------------------------------------------------- #


class _RecordingEmbedder:
    """A DummyEmbedder that logs whether each call came in as a passage or a query.

    Exposes ``embed_query`` (which DummyEmbedder does not), so it can prove that ingest
    takes the passage path and search takes the query path — the routing, not the model.
    """

    def __init__(self) -> None:
        self._inner = DummyEmbedder()
        self.model_id = self._inner.model_id
        self.dims = self._inner.dims
        self.query_prefix = "QUERY: "
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed", tuple(texts)))
        return self._inner.embed(texts)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed_query", tuple(texts)))
        return self._inner.embed([self.query_prefix + t for t in texts])

    def methods_used(self) -> set[str]:
        return {name for name, _ in self.calls}


def test_shim_prefers_embed_query_when_present():
    embedder = _RecordingEmbedder()

    vector = embed_query(embedder, "what is a skip list")

    assert embedder.methods_used() == {"embed_query"}
    # The prefix actually changed the vector — it was not silently dropped.
    assert vector == embedder._inner.embed(["QUERY: what is a skip list"])[0]


def test_shim_falls_back_to_embed_for_a_prefix_free_embedder():
    """DummyEmbedder has no ``embed_query``; discovery by getattr degrades cleanly."""
    embedder = DummyEmbedder()
    assert not hasattr(embedder, "embed_query")

    assert embed_query(embedder, "what is a skip list") == embedder.embed(["what is a skip list"])[0]


def test_ingest_uses_the_passage_path_and_search_uses_the_query_path(
    tmp_path, monkeypatch, dummy_config
):
    """The asymmetry holds end to end through the CLI, offline, with no model loaded."""
    monkeypatch.chdir(tmp_path)
    recorder = _RecordingEmbedder()
    monkeypatch.setattr("courserag.cli.get_embedder", lambda name, cfg: recorder)

    (tmp_path / "notes.txt").write_text("Skip lists use coin flips to pick tower height.\n")
    assert main(["init-course", "C"]) == 0
    assert main(["ingest", "C", "notes.txt", "--category", "lecture"]) == 0

    # Ingest embedded passages, and did so without ever taking the query path.
    assert recorder.methods_used() == {"embed"}
    assert all(not t.startswith("QUERY: ") for _, texts in recorder.calls for t in texts)

    recorder.calls.clear()
    assert main(["search", "C", "how tall is a tower", "--json"]) == 0

    assert recorder.methods_used() == {"embed_query"}


# --------------------------------------------------------------------------- #
# The real models
# --------------------------------------------------------------------------- #

pytestmark_local = pytest.mark.skipif(
    not is_available(), reason='requires the [local] extra: pip install -e ".[local]"'
)


@pytestmark_local
def test_bge_declares_the_documented_prefix():
    assert SentenceTransformerEmbedder(DEFAULT_MODEL_ID).query_prefix == BGE_PREFIX


@pytestmark_local
def test_symmetric_models_get_no_prefix():
    """MiniLM was not trained with an instruction, so prefixing it would only hurt."""
    assert SentenceTransformerEmbedder(MINILM_MODEL_ID).query_prefix == ""


@pytestmark_local
def test_bge_embed_query_is_exactly_the_prefixed_passage_embedding():
    """The behavioural assertion: the prefix is applied, not merely declared."""
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL_ID)
    query = "how is the tower height chosen in a skip list"

    queried = embedder.embed_query([query])[0]

    assert queried == embedder.embed([BGE_PREFIX + query])[0]
    assert queried != embedder.embed([query])[0]


@pytestmark_local
def test_embed_query_is_a_no_op_for_a_model_without_a_prefix():
    embedder = SentenceTransformerEmbedder(MINILM_MODEL_ID)
    assert embedder.embed_query(["a query"]) == embedder.embed(["a query"])
