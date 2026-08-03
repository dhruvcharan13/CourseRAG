"""DummyEmbedder determinism, dimensions, and factory wiring."""

from __future__ import annotations

import pytest

from courserag.config import Config
from courserag.embedding import Embedder, get_embedder, resolve_embedder_name
from courserag.embedding import sentence_transformer as st
from courserag.embedding.dummy import DummyEmbedder


def test_default_dims_and_model_id():
    emb = DummyEmbedder()
    assert emb.dims == 64
    assert emb.model_id == "dummy-hash-64"


def test_embed_shape():
    emb = DummyEmbedder()
    vectors = emb.embed(["a", "bb", "ccc"])
    assert len(vectors) == 3
    assert all(len(v) == 64 for v in vectors)


def test_deterministic_across_instances():
    a = DummyEmbedder().embed(["hello world"])
    b = DummyEmbedder().embed(["hello world"])
    assert a == b


def test_distinct_texts_differ():
    (v1,) = DummyEmbedder().embed(["hello"])
    (v2,) = DummyEmbedder().embed(["world"])
    assert v1 != v2


def test_factory_returns_dummy_and_satisfies_protocol():
    emb = get_embedder("dummy", Config())
    assert isinstance(emb, DummyEmbedder)
    assert isinstance(emb, Embedder)  # runtime_checkable protocol


def test_factory_rejects_unknown_embedder():
    with pytest.raises(NotImplementedError):
        get_embedder("openai", Config())


# --------------------------------------------------------------------------- #
# "auto" resolution
# --------------------------------------------------------------------------- #


def test_auto_prefers_local_model_when_installed(monkeypatch):
    monkeypatch.setattr(st, "is_available", lambda: True)
    assert resolve_embedder_name("auto") == "local"


def test_auto_falls_back_to_dummy_without_the_extra(monkeypatch):
    monkeypatch.setattr(st, "is_available", lambda: False)
    assert resolve_embedder_name("auto") == "dummy"
    assert isinstance(get_embedder("auto", Config()), DummyEmbedder)


@pytest.mark.parametrize(
    "name", ["dummy", "minilm", "local", "bge", "sentence-transformers/all-MiniLM-L6-v2"]
)
def test_explicit_names_pass_through_resolution(name):
    assert resolve_embedder_name(name) == name


@pytest.mark.parametrize("name", ["local", "bge", "minilm"])
def test_requesting_a_real_model_without_the_extra_raises_importerror(monkeypatch, name):
    # A missing extra must fail before anything is created, with an install hint.
    monkeypatch.setattr(st, "is_available", lambda: False)
    with pytest.raises(ImportError, match=r"\[local\]"):
        get_embedder(name, Config())
