"""Real local embeddings: dimensions, unit norm, and cosine sanity.

Skipped unless the ``[local]`` extra is installed; the first run downloads ~90MB into
the shared HuggingFace cache. These assert on the embedder directly — no retrieval is
involved (that is Phase 3).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from course_kb.cli import main
from course_kb.embedding.sentence_transformer import (
    DEFAULT_MODEL_ID,
    SentenceTransformerEmbedder,
    is_available,
)
from course_kb.manifest import read_manifest
from course_kb.store import CourseStore

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not is_available(), reason='requires the [local] extra: pip install -e ".[local]"'
)

# Two ways of saying the same thing, and one unrelated sentence.
SIMILAR_A = "Quicksort has an average-case running time of O(n log n)."
SIMILAR_B = "The average-case runtime of quicksort is O(n log n)."
UNRELATED = "Let the ribeye rest for ten minutes before slicing it against the grain."


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    """One model load shared by every test in this module."""
    return SentenceTransformerEmbedder()


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (_norm(a) * _norm(b))


def _write_local_config(tmp_path: Path, monkeypatch) -> None:
    """Chdir into a scratch dir configured for the real embedder.

    ``cache_dir`` points at the shared HuggingFace cache so tests reuse an already
    downloaded model instead of fetching one per run.
    """
    from huggingface_hub import constants

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        f'embedder = "minilm"\ncache_dir = {json.dumps(constants.HF_HUB_CACHE)}\n',
        encoding="utf-8",
    )


def test_model_id_and_dims(embedder):
    assert embedder.model_id == DEFAULT_MODEL_ID
    assert embedder.dims == 384


def test_dims_are_known_before_the_model_loads():
    # init-course needs the vector width to size the table; paying a model load
    # for it would make every CLI invocation slow.
    fresh = SentenceTransformerEmbedder()
    assert fresh.dims == 384
    assert fresh._model is None


def test_embed_returns_finite_unit_vectors(embedder):
    vectors = embedder.embed([SIMILAR_A, SIMILAR_B, UNRELATED])

    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == 384
        assert all(math.isfinite(x) for x in vector)
        assert abs(_norm(vector) - 1.0) < 1e-3  # normalized, so cosine == dot product


def test_similar_texts_score_high_and_unrelated_low(embedder):
    a, b, c = embedder.embed([SIMILAR_A, SIMILAR_B, UNRELATED])

    similar = _cosine(a, b)
    unrelated = _cosine(a, c)

    assert similar > 0.8, f"near-identical texts scored only {similar:.3f}"
    assert unrelated < 0.4, f"unrelated texts scored {unrelated:.3f}"
    assert similar > unrelated


def test_empty_input_short_circuits_without_loading_the_model():
    fresh = SentenceTransformerEmbedder()
    assert fresh.embed([]) == []
    assert fresh._model is None  # a no-op re-ingest never pays for a model load


def test_batching_does_not_change_vectors(embedder):
    texts = [SIMILAR_A, SIMILAR_B, UNRELATED]
    batched = embedder.embed(texts)
    one_at_a_time = [embedder.embed([text])[0] for text in texts]

    for got, want in zip(batched, one_at_a_time):
        assert max(abs(g - w) for g, w in zip(got, want)) < 1e-5


def test_batch_size_is_configurable_and_order_preserving(embedder):
    texts = [SIMILAR_A, SIMILAR_B, UNRELATED]
    small = SentenceTransformerEmbedder(batch_size=1).embed(texts)

    for got, want in zip(small, embedder.embed(texts)):
        assert max(abs(g - w) for g, w in zip(got, want)) < 1e-5


# --------------------------------------------------------------------------- #
# Token budget / truncation
# --------------------------------------------------------------------------- #

# Notation-dense text: ~1 word-piece per character, where chars/4 badly under-counts.
MATH_HEAVY = "Let A = { n ∈ N : n ≡ 1 ( mod 3 ) } and B = { n ∈ N : n ≡ 2 ( mod 3 ) }. " * 12


def test_max_input_tokens_is_the_models_budget(embedder):
    assert embedder.max_input_tokens == 256


def test_count_tokens_matches_the_tokenizer(embedder):
    counts = embedder.count_tokens([SIMILAR_A, MATH_HEAVY])
    assert counts[0] < 32  # one short sentence
    assert counts[1] > embedder.max_input_tokens  # this one gets truncated
    assert embedder.count_tokens([]) == []


def test_char_estimate_undercounts_notation_dense_text(embedder):
    # The reason ingest counts real tokens instead of trusting chars/4: this text is
    # under the character-based threshold but well over the model's real budget.
    from course_kb.chunker import estimate_tokens

    (real,) = embedder.count_tokens([MATH_HEAVY])
    assert estimate_tokens(MATH_HEAVY) < 256 < real


def test_oversize_text_embeds_only_its_head(embedder):
    """Truncation is silent and total: the tail contributes nothing to the vector."""
    tokenizer = embedder._ensure_model().tokenizer
    limit = embedder.max_input_tokens
    ids = tokenizer(MATH_HEAVY, add_special_tokens=False, truncation=False, verbose=False)[
        "input_ids"
    ]
    assert len(ids) > limit
    head = tokenizer.decode(ids[: limit - 2])

    full_vector, head_vector = embedder.embed([MATH_HEAVY, head])
    assert max(abs(a - b) for a, b in zip(full_vector, head_vector)) == 0.0


def test_cli_ingest_warns_with_real_token_counts(tmp_path, monkeypatch, capsys):
    _write_local_config(tmp_path, monkeypatch)
    assert main(["init-course", "C"]) == 0

    listing = tmp_path / "proof.txt"
    listing.write_text("```\n" + MATH_HEAVY + "\n```", encoding="utf-8")  # atomic block
    assert main(["ingest", "C", str(listing), "--category", "notes"]) == 0

    err = capsys.readouterr().err
    assert "exceed the model's 256-token limit" in err
    assert "not searchable" in err

    # The stored text is still whole — only the vector is head-only.
    records = CourseStore.open_or_create(
        tmp_path / "course-kb" / "courses" / "C", 384
    ).get_all()
    assert any(MATH_HEAVY.strip() in r.text for r in records)


def test_cli_ingest_stores_384_dim_unit_vectors(tmp_path, monkeypatch):
    """End to end: init-course records the real model, ingest stores real vectors."""
    _write_local_config(tmp_path, monkeypatch)

    assert main(["init-course", "C"]) == 0
    course_dir = tmp_path / "course-kb" / "courses" / "C"
    manifest = read_manifest(course_dir)
    assert manifest.embedding_model == DEFAULT_MODEL_ID
    assert manifest.dims == 384

    assert main(["ingest", "C", str(FIXTURES / "notes.pdf"), "--category", "notes"]) == 0

    records = CourseStore.open_or_create(course_dir, manifest.dims).get_all()
    assert records
    for record in records:
        assert record.vector is not None
        assert len(record.vector) == 384
        assert all(math.isfinite(x) for x in record.vector)
        assert abs(_norm(record.vector) - 1.0) < 1e-3
