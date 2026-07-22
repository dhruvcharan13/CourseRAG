"""DummyEmbedder determinism, dimensions, and factory wiring."""

from __future__ import annotations

import pytest

from course_kb.config import Config
from course_kb.embedding import Embedder, get_embedder
from course_kb.embedding.dummy import DummyEmbedder


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
