"""``kb search`` and ``kb eval``: output shape, and the guards that refuse to guess."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from course_kb.cli import main
from course_kb.store import CourseStore

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def course(tmp_path, monkeypatch, dummy_config):
    """A dummy-embedded course over slides.pdf: three chunks on three distinct pages.

    A real PDF rather than a text file, because page numbers are what eval labels match
    on and the text parser has none. Returns the stored chunks so tests can query with a
    chunk's exact text: dummy vectors are seeded by text, so an exact query embeds to
    that chunk's own vector and must come back at rank 1 with similarity 1.
    """
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "CS240"]) == 0
    assert main(["ingest", "CS240", str(FIXTURES / "slides.pdf"), "--category", "lecture"]) == 0

    store = CourseStore.open_or_create(tmp_path / "course-kb" / "courses" / "CS240", 64)
    chunks = sorted(store.get_all(), key=lambda c: c.page)
    assert [c.page for c in chunks] == [1, 2, 3]
    return chunks


def test_search_prints_ranked_results_with_score_and_citation(course, capsys):
    target = course[1]

    assert main(["search", "CS240", target.text]) == 0

    out = capsys.readouterr().out
    assert out.startswith("3 result(s)")
    # Rank, score, and a full citation on the first line.
    assert "1. [1.000] slides.pdf p2" in out
    assert target.title in out
    assert "Left and right rotations" in out


def test_k_limits_the_number_of_results(course, capsys):
    assert main(["search", "CS240", "rotations", "-k", "2"]) == 0
    assert capsys.readouterr().out.startswith("2 result(s)")


def test_json_mode_emits_a_pipeable_payload_on_stdout(course, capsys):
    assert main(["search", "CS240", course[0].text, "--json", "-k", "2"]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert [row["rank"] for row in payload] == [1, 2]
    assert payload[0]["score"] >= payload[1]["score"]
    assert (payload[0]["source_file"], payload[0]["page"]) == ("slides.pdf", 1)
    assert set(payload[0]) == {
        "rank", "score", "rerank_score", "id", "text",
        "source_file", "page", "title", "module", "category",
    }
    # Dense-only: the rerank column exists in the schema but is unset.
    assert payload[0]["rerank_score"] is None
    # Human commentary goes to stderr so stdout stays valid JSON, as `kb chunks` does.
    assert "result(s)" in captured.err


def test_searching_a_dummy_course_warns_that_rankings_are_noise(course, capsys):
    assert main(["search", "CS240", "anything"]) == 0
    assert "no semantic meaning" in capsys.readouterr().err


def test_searching_an_uninitialized_course_fails(tmp_path, monkeypatch, dummy_config, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["search", "NOPE", "anything"]) == 1
    assert "not initialized" in capsys.readouterr().err


def test_searching_an_empty_course_says_so(tmp_path, monkeypatch, dummy_config, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "EMPTY"]) == 0
    capsys.readouterr()

    assert main(["search", "EMPTY", "anything"]) == 1
    assert "no chunks yet" in capsys.readouterr().err


def test_query_model_mismatch_is_refused_loudly(course, tmp_path, capsys):
    """A query embedded by a different model lands in a different space.

    The scores would still look like numbers, which is exactly the danger — so search
    refuses rather than returning results from a space the passages do not live in.
    """
    (tmp_path / "config.toml").write_text('embedder = "dummy"\n', encoding="utf-8")
    # Same 'dummy' family, different width -> a different model_id, so a mismatch.
    import course_kb.cli as cli
    from course_kb.embedding.dummy import DummyEmbedder

    original = cli.get_embedder
    cli.get_embedder = lambda name, cfg: DummyEmbedder(dims=32)
    try:
        capsys.readouterr()
        assert main(["search", "CS240", "skip lists"]) == 1
    finally:
        cli.get_embedder = original

    captured = capsys.readouterr()
    assert "embedder mismatch" in captured.err
    assert "refusing to search" in captured.err
    assert "dummy-hash-64" in captured.err and "dummy-hash-32" in captured.err
    # Nothing was ranked: no results leaked out of the wrong vector space.
    assert captured.out == ""


# --------------------------------------------------------------------------- #
# Two-stage retrieval (--rerank)
# --------------------------------------------------------------------------- #


@pytest.fixture
def rerank_course(tmp_path, monkeypatch, dummy_rerank_config):
    """The same slides.pdf course, with both dummies pinned in config."""
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "CS240"]) == 0
    assert main(["ingest", "CS240", str(FIXTURES / "slides.pdf"), "--category", "lecture"]) == 0
    store = CourseStore.open_or_create(tmp_path / "course-kb" / "courses" / "CS240", 64)
    return sorted(store.get_all(), key=lambda c: c.page)


def test_default_search_is_unchanged_by_the_rerank_feature(rerank_course, capsys):
    """Without the flag the pipeline must behave exactly as it did in Phase 3."""
    assert main(["search", "CS240", rerank_course[1].text]) == 0
    out = capsys.readouterr().out
    assert "1. [1.000] slides.pdf p2" in out
    # Bare cosine in the score bracket — no cross-encoder column when dense-only.
    # Matched on the rank lines only; chunk text can contain anything.
    for line in out.splitlines():
        if re.match(r"^\d+\. \[", line):
            assert re.match(r"^\d+\. \[-?\d\.\d{3}\] ", line), line


def test_rerank_reorders_and_shows_both_scores(rerank_course, capsys):
    """The dummy reverses stage one, so the last dense candidate must come back first."""
    assert main(["search", "CS240", rerank_course[1].text, "--rerank", "-k", "3"]) == 0

    out = capsys.readouterr().out
    assert "ce " in out and "cos " in out  # both scales are shown
    first = next(ln for ln in out.splitlines() if ln.startswith("1. "))
    # Dense would rank p2 first (exact text match); reversed, it must not be first.
    assert "slides.pdf p2" not in first


def test_rerank_json_carries_both_scores_and_citations(rerank_course, capsys):
    assert main(["search", "CS240", rerank_course[0].text, "--rerank", "--json", "-k", "2"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [row["rank"] for row in payload] == [1, 2]
    assert all(row["rerank_score"] is not None for row in payload)
    assert all(row["score"] is not None for row in payload)
    assert payload[0]["source_file"] == "slides.pdf"
    assert payload[0]["page"] in (1, 2, 3)
    # Reranked order is by cross-encoder score, not cosine.
    assert payload[0]["rerank_score"] >= payload[1]["rerank_score"]


def test_candidates_below_k_is_clamped_not_silently_truncating(rerank_course, capsys):
    """`-k 3 --candidates 1` must still return 3 results, not 1."""
    assert main(["search", "CS240", "rotations", "--rerank", "-k", "3", "--candidates", "1"]) == 0
    assert capsys.readouterr().out.startswith("3 result(s)")


def test_rerank_without_the_local_extra_fails_with_a_hint(rerank_course, tmp_path, capsys):
    (tmp_path / "config.toml").write_text(
        'embedder = "dummy"\nreranker = "local"\n', encoding="utf-8"
    )
    import course_kb.reranking.cross_encoder as ce

    original = ce.is_available
    ce.is_available = lambda: False
    try:
        capsys.readouterr()
        assert main(["search", "CS240", "rotations", "--rerank"]) == 1
    finally:
        ce.is_available = original

    err = capsys.readouterr().err
    assert "sentence-transformers" in err and "--rerank" in err


# --------------------------------------------------------------------------- #
# kb eval
# --------------------------------------------------------------------------- #


def _write_eval_set(tmp_path, query, *, course_id="CS240", pages=(2,)):
    path = tmp_path / "myset.json"
    path.write_text(
        json.dumps(
            {
                "course": course_id,
                "queries": [
                    {
                        "id": "q1",
                        "query": query,
                        "expected": [{"source_file": "slides.pdf", "pages": list(pages)}],
                        "note": "the AVL rotations slide",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_eval_reports_recall_and_mrr(course, tmp_path, capsys):
    path = _write_eval_set(tmp_path, course[1].text)

    assert main(["eval", "CS240", "--set", str(path)]) == 0

    out = capsys.readouterr().out
    assert "recall@1" in out and "MRR@" in out
    assert "100.0%" in out
    assert "No misses." in out
    # Scores are surfaced for threshold calibration, and labelled as not being a cutoff.
    assert "not thresholded" in out


def test_eval_prints_misses_with_expected_and_actual(course, tmp_path, capsys):
    """A miss has to be legible: the query, the label, and what actually came back."""
    path = _write_eval_set(tmp_path, "something entirely unrelated", pages=(99,))

    assert main(["eval", "CS240", "--set", str(path)]) == 0

    out = capsys.readouterr().out
    assert "Misses (1/1)" in out
    assert "expected: slides.pdf p99" in out
    assert "the AVL rotations slide" in out
    assert "got:" in out
    # The actual ranking is printed, so a failure is diagnosable and not just a number.
    assert "1. [" in out and "slides.pdf p" in out


def test_eval_refuses_a_set_written_for_another_course(course, tmp_path, capsys):
    path = _write_eval_set(tmp_path, "anything", course_id="CS348")

    assert main(["eval", "CS240", "--set", str(path)]) == 1
    assert "is an eval set for 'CS348'" in capsys.readouterr().err


def test_eval_without_a_set_explains_where_to_put_one(course, capsys):
    assert main(["eval", "CS240"]) == 1
    err = capsys.readouterr().err
    assert "No eval set at" in err and "evals/CS240.json" in err


def test_eval_rerank_prints_a_paired_ab_with_a_net_headline(rerank_course, tmp_path, capsys):
    """The A/B must lead with NET and account for every query in the matrix."""
    path = _write_eval_set(tmp_path, rerank_course[1].text)

    assert main(["eval", "CS240", "--set", str(path), "--rerank"]) == 0

    out = capsys.readouterr().out
    assert "NET rank-1 change:" in out
    assert "movement matrix" in out
    assert "<- broke" in out and "^ fixed" in out
    # Both columns present so the comparison cannot be reported one-sided.
    assert "dense" in out and "reranked" in out
    assert "latency/query" in out
    # The dummy reverses stage one, so the exact-text query must be demoted: net < 0.
    assert "NET rank-1 change: -1" in out


# --------------------------------------------------------------------------- #
# Document scoping: kb search --file, kb show
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_doc_course(tmp_path, monkeypatch, dummy_config):
    """Two documents in one course, so scoping has something to exclude."""
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "CS240"]) == 0
    assert main(["ingest", "CS240", str(FIXTURES / "slides.pdf"), "--category", "lecture"]) == 0
    other = tmp_path / "other.txt"
    other.write_text("Amortized analysis uses the banker's method.\n", encoding="utf-8")
    assert main(["ingest", "CS240", str(other), "--category", "notes"]) == 0
    return tmp_path


def test_search_file_restricts_results_to_one_document(two_doc_course, capsys):
    assert main(["search", "CS240", "amortized analysis", "--file", "other.txt"]) == 0

    out = capsys.readouterr().out
    assert "other.txt" in out
    assert "slides.pdf" not in out


def test_search_file_rejects_an_unknown_document_and_lists_the_real_ones(
    two_doc_course, capsys
):
    assert main(["search", "CS240", "anything", "--file", "ghost.pdf"]) == 1

    err = capsys.readouterr().err
    assert "No document 'ghost.pdf'" in err
    assert "slides.pdf" in err


def test_show_prints_a_document_in_page_order(two_doc_course, capsys):
    assert main(["show", "CS240", "slides.pdf"]) == 0

    out = capsys.readouterr().out
    assert "3 passage(s)" in out
    assert out.index("Balanced Search Trees") < out.index("AVL Rotations")
    assert "other.txt" not in out


def test_show_honours_a_page_range(two_doc_course, capsys):
    assert main(["show", "CS240", "slides.pdf", "--pages", "2-3"]) == 0

    out = capsys.readouterr().out
    assert "AVL Rotations" in out
    assert "Balanced Search Trees" not in out


def test_show_rejects_an_unreadable_range(two_doc_course, capsys):
    assert main(["show", "CS240", "slides.pdf", "--pages", "banana"]) == 1
    assert "Could not read" in capsys.readouterr().err


def test_show_on_an_unknown_course_fails(tmp_path, monkeypatch, dummy_config, capsys):
    monkeypatch.chdir(tmp_path)

    assert main(["show", "NOPE", "x.pdf"]) == 1
    assert "No course 'NOPE'" in capsys.readouterr().err
