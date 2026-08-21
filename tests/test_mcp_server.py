"""The MCP server: root resolution from a foreign cwd, stdout hygiene, and the tools.

The first two are the ones that matter. A stdio server inherits the *client's* working
directory, and everything else here is only meaningful if it finds the right data and
does not corrupt the protocol while doing so.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mcp") is None,
    reason='requires the [mcp] extra: pip install -e ".[mcp]"',
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: The developer's own model cache. The real-path stdout test has to load real models,
#: and downloading them would make an offline suite fail, so it reuses what ``kb
#: ingest`` already put here and skips when that is not populated.
REAL_MODELS = Path(__file__).resolve().parent.parent / "course-kb" / "models"
_REAL_MODEL_DIRS = (
    "models--BAAI--bge-small-en-v1.5",
    "models--cross-encoder--ms-marco-MiniLM-L-6-v2",
)

#: Gate for the real-model path. Skipping keeps the no-extra and dummy runs exactly as
#: they were — the point of this variant is coverage, not a new hard dependency.
requires_real_models = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None
    or not all((REAL_MODELS / name).is_dir() for name in _REAL_MODEL_DIRS),
    reason='requires the [local] extra and both models cached under course-kb/models',
)

_REAL_SETTINGS = (
    f'embedder = "local"\nreranker = "local"\ncache_dir = "{REAL_MODELS}"\n'
)


@pytest.fixture
def kb_root(tmp_path, monkeypatch):
    """A dummy-embedded course inside ``tmp_path/kbroot``, built through the CLI.

    Returns the data root, deliberately *not* the cwd of any test that uses it: the
    whole point of ``COURSE_KB_ROOT`` is that those two can differ.
    """
    from courserag.cli import main

    workdir = tmp_path / "build-here"
    workdir.mkdir()
    (workdir / "config.toml").write_text(
        'root = "kbroot"\nembedder = "dummy"\nreranker = "dummy"\n', encoding="utf-8"
    )
    monkeypatch.chdir(workdir)
    assert main(["init-course", "CS240"]) == 0
    assert main(["ingest", "CS240", str(FIXTURES / "slides.pdf"), "--category", "lecture"]) == 0
    return workdir / "kbroot"


@pytest.fixture
def tools(kb_root, monkeypatch):
    """The tool functions, pointed at ``kb_root`` the way a real client points them.

    Sets the environment variable rather than patching a config object, so the tests
    exercise the same resolution path the deployed server uses. The embedder is pinned
    to the dummy through a config.toml *inside the root*, which is also the cwd here —
    the foreign-cwd case is covered by subprocess below.
    """
    (kb_root / "config.toml").write_text(
        'embedder = "dummy"\nreranker = "dummy"\n', encoding="utf-8"
    )
    monkeypatch.chdir(kb_root)
    monkeypatch.setenv("COURSE_KB_ROOT", str(kb_root))
    import courserag.mcp_server as srv

    return srv


# --------------------------------------------------------------------------- #
# Root resolution — the gate
# --------------------------------------------------------------------------- #

# Runs the tools from a directory that is not the data root and is not the repo, which
# is the situation a stdio server is always in. ``emit`` is substituted per call so the
# same body can be run once writing its payload and once writing nothing at all.
_PROBE = """
import json, sys
from courserag.mcp_server import list_courses, search_course
payload = {{
    "courses": list_courses(),
    "search": search_course("CS240", "rotations", k=1),
}}
{emit}
"""


def _foreign_cwd(tmp_path: Path) -> Path:
    """A directory that is neither the data root nor the repo, with the embedder pinned.

    The config.toml is what makes the assertion about *root* clean: without it the
    subprocess would resolve ``embedder = "auto"`` to the real local model and refuse
    the dummy-built course, so a root-resolution failure and an embedder mismatch would
    be indistinguishable. It also shows the two sources composing as documented — the
    environment supplies the root, the file supplies everything else.
    """
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    (elsewhere / "config.toml").write_text(
        'root = "unused-relative-root"\nembedder = "dummy"\nreranker = "dummy"\n',
        encoding="utf-8",
    )
    return elsewhere


def _run(
    cwd: Path, root: Path | None, emit: str, probe: str = _PROBE
) -> subprocess.CompletedProcess:
    # HF_HUB_OFFLINE keeps the real-model variant from reaching the network: the models
    # are already cached, and a suite that quietly needs a Hub round trip is a suite
    # that fails on a plane. mcp_server setdefault()s it too; setting it here covers
    # the window before that import.
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "HF_HUB_OFFLINE": "1"}
    if root is not None:
        env["COURSE_KB_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, "-c", probe.format(emit=emit)], cwd=cwd, env=env,
        capture_output=True, text=True, check=True,
    )


def _probe(cwd: Path, root: Path | None) -> dict:
    """Run the tools in a fresh interpreter from ``cwd``; return their results."""
    result = _run(cwd, root, emit="sys.stdout.write(json.dumps(payload))")
    # json.loads doubles as a stdout-purity check: any log line, banner or stray print
    # concatenated onto the payload makes this raise instead of silently passing.
    return json.loads(result.stdout)


def test_the_env_var_finds_the_courses_from_a_foreign_cwd(kb_root, tmp_path):
    """The failure this prevents: a server that reports an empty knowledge base.

    Note the config.toml in the foreign cwd names a *different*, relative root. The
    environment has to win over it, or a client that happens to launch from a directory
    with its own config would silently query the wrong knowledge base.
    """
    payload = _probe(_foreign_cwd(tmp_path), kb_root)

    assert "CS240" in payload["courses"]
    assert "3 passage(s)" in payload["courses"]
    assert "slides.pdf" in payload["search"]


def test_without_the_env_var_a_foreign_cwd_finds_nothing(kb_root, tmp_path):
    """The negative control.

    Without it the test above passes on any machine that happens to be sitting in the
    right directory, and proves nothing about the variable it claims to test.
    """
    payload = _probe(_foreign_cwd(tmp_path), root=None)

    assert "no courses yet" in payload["courses"]
    assert "No course named 'CS240'" in payload["search"]


def test_the_tools_write_nothing_to_stdout(kb_root, tmp_path):
    """Under stdio, stdout *is* the JSON-RPC channel; one stray print corrupts it.

    Same tools, same successful calls, with only the payload write removed — so
    whatever is left on stdout came from somewhere that had no business writing there.
    The mcp SDK installs a logging handler on import, which is exactly the kind of
    thing this catches if it is ever pointed at the wrong stream.
    """
    result = _run(_foreign_cwd(tmp_path), kb_root, emit="assert payload['courses']")

    assert result.stdout == ""


# --------------------------------------------------------------------------- #
# Stdout hygiene on the *real* model path
# --------------------------------------------------------------------------- #

# The dummy test above cannot see the failure that actually threatens the client. The
# dummy embedder loads no model, so the noisiest code in the process — sentence-
# transformers and huggingface_hub, which print progress bars and log on load — never
# runs. That machinery writes to stderr today, which is why the server works; nothing
# pins it there. A dependency bump that moves one progress bar to stdout would corrupt
# the very first JSON-RPC frame and surface as an unexplained client-side protocol
# error, with the whole dummy-based suite still green.
#
# rerank=True is deliberate: it is the only call that loads the *second* model, so both
# the embedder and the cross-encoder get exercised in one probe.
_REAL_PROBE = """
import json, sys
from courserag.mcp_server import list_courses, search_course
payload = {{
    "courses": list_courses(),
    "search": search_course("CS240", "how does a rotation work?", k=2, rerank=True),
}}
{emit}
"""


@pytest.fixture
def real_kb_root(tmp_path, monkeypatch):
    """A course built with the real bge embedder, so the real path is searchable.

    Cannot reuse ``kb_root``: that course is dummy-embedded, and ``_open`` refuses a
    real-embedder query against it by design. The refusal happens *before* any model
    loads, so a real-path stdout test pointed at it would load nothing and assert
    nothing — passing for the wrong reason.
    """
    from courserag.cli import main

    workdir = tmp_path / "build-real"
    workdir.mkdir()
    (workdir / "config.toml").write_text(
        f'root = "kbroot"\n{_REAL_SETTINGS}', encoding="utf-8"
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(workdir)
    assert main(["init-course", "CS240"]) == 0
    assert main(["ingest", "CS240", str(FIXTURES / "slides.pdf"), "--category", "lecture"]) == 0
    return workdir / "kbroot"


def _foreign_cwd_real(tmp_path: Path) -> Path:
    """``_foreign_cwd``, but pinned to the real models rather than the dummies."""
    elsewhere = tmp_path / "somewhere-else-real"
    elsewhere.mkdir()
    (elsewhere / "config.toml").write_text(
        f'root = "unused-relative-root"\n{_REAL_SETTINGS}', encoding="utf-8"
    )
    return elsewhere


@requires_real_models
def test_the_real_models_write_nothing_to_stdout(real_kb_root, tmp_path):
    """The regression guard the dummy variant cannot provide.

    Loads the real embedder *and* the real cross-encoder inside the tool calls, then
    asserts the same thing: stdout is empty. If a future sentence-transformers or
    huggingface_hub release logs to stdout instead of stderr, this fails here rather
    than in a user's editor.
    """
    result = _run(
        _foreign_cwd_real(tmp_path), real_kb_root,
        emit="assert payload['search']", probe=_REAL_PROBE,
    )

    assert result.stdout == ""


@requires_real_models
def test_the_real_path_probe_can_actually_see_stdout(real_kb_root, tmp_path):
    """Negative control for the test above.

    An assertion that stdout is empty is worthless if the harness could never observe a
    write in the first place — a mis-plumbed subprocess would pass it forever. This
    writes one byte on the same path and requires it to show up.
    """
    result = _run(
        _foreign_cwd_real(tmp_path), real_kb_root,
        emit="sys.stdout.write('x')", probe=_REAL_PROBE,
    )

    assert result.stdout == "x"


# --------------------------------------------------------------------------- #
# list_courses / course_info
# --------------------------------------------------------------------------- #


def test_list_courses_reports_documents_and_passages(tools):
    out = tools.list_courses()
    assert "1 course(s)" in out
    assert "CS240 — 1 document(s), 3 passage(s)" in out
    assert "lecture" in out


def test_list_courses_flags_a_placeholder_course_as_unsearchable(tools):
    """A dummy course ranks by random hashes. Saying so is the difference between an
    agent reporting "no answer in the slides" and inventing one from noise."""
    assert "NOT SEARCHABLE" in tools.list_courses()


def test_list_courses_on_an_empty_root_says_how_to_make_one(tools, monkeypatch, tmp_path):
    empty = tmp_path / "empty-root"
    empty.mkdir()
    monkeypatch.setenv("COURSE_KB_ROOT", str(empty))

    out = tools.list_courses()
    assert "no courses yet" in out and "kb init-course" in out


def test_course_info_lists_the_source_documents(tools):
    out = tools.course_info("CS240")
    assert "Course:      CS240" in out
    assert "Passages:    3" in out
    assert "slides.pdf" in out
    assert "WARNING" in out  # dummy-embedded


def test_course_info_on_an_unknown_course_names_the_real_ones(tools):
    out = tools.course_info("CS999")
    assert "No course named 'CS999'" in out
    assert "Available courses: CS240" in out


# --------------------------------------------------------------------------- #
# search_course
# --------------------------------------------------------------------------- #


def test_search_returns_passages_with_a_citation_and_the_text(tools, kb_root):
    """Dummy vectors are seeded by text, so a chunk's own text retrieves itself."""
    from courserag.manifest import read_manifest
    from courserag.store import CourseStore

    course_dir = kb_root / "courses" / "CS240"
    chunks = sorted(CourseStore.open_or_create(course_dir, read_manifest(course_dir).dims)
                    .get_all(), key=lambda c: c.page)
    target = chunks[1]

    out = tools.search_course("CS240", target.text, k=1)

    assert "[1] slides.pdf p2 (lecture; relevance 1.000)" in out
    assert target.title in out
    assert target.text in out  # the full passage, not a snippet


def test_search_honours_k(tools):
    out = tools.search_course("CS240", "rotations", k=2)
    assert "Top 2 passage(s)" in out
    assert out.count("\n---\n") == 1  # two blocks, one separator


def test_search_clamps_nothing_but_refuses_a_nonsense_k(tools):
    assert "k must be at least 1" in tools.search_course("CS240", "rotations", k=0)


def test_search_with_rerank_shows_both_scores(tools):
    """The dummy reranker reverses stage one, so this also proves the flag is wired."""
    out = tools.search_course("CS240", "rotations", k=2, rerank=True)
    assert "rerank " in out and "dense " in out


def test_search_defaults_to_dense_only(tools):
    """Off by default, matching the CLI: the committed baseline stays the default path."""
    out = tools.search_course("CS240", "rotations", k=1)
    assert "relevance " in out and "rerank " not in out


def test_search_on_an_unknown_course_names_the_real_ones(tools):
    out = tools.search_course("CS999", "anything")
    assert "No course named 'CS999'" in out
    assert "Available courses: CS240" in out


def test_search_warns_when_the_course_is_placeholder_embedded(tools):
    out = tools.search_course("CS240", "rotations", k=1)
    assert out.startswith("WARNING")
    assert "must not be presented as an answer" in out


def test_search_on_an_empty_course_explains_rather_than_returning_nothing(tools, kb_root):
    from courserag.cli import main

    assert main(["init-course", "EMPTY"]) == 0
    out = tools.search_course("EMPTY", "anything")
    assert "no documents indexed yet" in out and "kb ingest EMPTY" in out


def test_search_refuses_a_course_built_by_a_different_model(tools, kb_root):
    """Vectors from two models are not comparable; the scores would still look real."""
    (kb_root / "config.toml").write_text(
        'embedder = "dummy-32"\nreranker = "dummy"\n', encoding="utf-8"
    )
    import courserag.embedding as embedding
    from courserag.embedding.dummy import DummyEmbedder

    original = embedding.get_embedder
    tools.get_embedder = lambda name, cfg: DummyEmbedder(dims=32)
    try:
        out = tools.search_course("CS240", "rotations")
    finally:
        tools.get_embedder = original

    assert "dummy-hash-64" in out and "dummy-hash-32" in out
    assert "not comparable" in out


# --------------------------------------------------------------------------- #
# read_document / document-scoped search
# --------------------------------------------------------------------------- #


def test_read_document_returns_the_whole_file_in_page_order(tools):
    out = tools.read_document("CS240", "slides.pdf")

    assert "3 of 3 passage(s), pages 1-3 of 3" in out
    assert out.index("Balanced Search Trees") < out.index("AVL Rotations") < out.index(
        "Amortized Analysis"
    )
    # A straight read is not a ranking, so it must not display one.
    assert "relevance" not in out


def test_read_document_honours_an_explicit_page_range(tools):
    out = tools.read_document("CS240", "slides.pdf", pages="2")

    assert "AVL Rotations" in out
    assert "Balanced Search Trees" not in out


def test_read_document_accepts_an_open_ended_range(tools):
    out = tools.read_document("CS240", "slides.pdf", pages="2-")

    assert "AVL Rotations" in out and "Amortized Analysis" in out
    assert "Balanced Search Trees" not in out


def test_read_document_matches_a_filename_case_insensitively(tools):
    """Names get retyped out of a citation; capitalization should not be a dead end."""
    assert "Balanced Search Trees" in tools.read_document("CS240", "SLIDES.PDF")


def test_read_document_on_an_unknown_file_names_the_real_ones(tools):
    out = tools.read_document("CS240", "nope.pdf")

    assert "No document named 'nope.pdf'" in out
    assert "slides.pdf" in out


def test_read_document_rejects_an_unreadable_page_range(tools):
    out = tools.read_document("CS240", "slides.pdf", pages="banana")

    assert "Could not read 'banana' as a page range" in out


def test_read_document_reports_an_empty_range_against_the_real_length(tools):
    out = tools.read_document("CS240", "slides.pdf", pages="90-")

    assert "No passages in slides.pdf within pages '90-'" in out
    assert "page 3" in out


def test_an_oversize_document_truncates_visibly_and_says_how_to_continue(tools, kb_root):
    """Truncation must never be silent, and the marker has to name a usable next call."""
    (kb_root / "config.toml").write_text(
        'embedder = "dummy"\nreranker = "dummy"\nmax_document_chars = 150\n', encoding="utf-8"
    )

    out = tools.read_document("CS240", "slides.pdf")

    assert "[truncated at the 150-character budget" in out
    assert 'pages="2-"' in out
    # Cut on a page boundary: page 2 is absent entirely rather than half-shown.
    assert "Balanced Search Trees" in out
    assert "AVL Rotations" not in out


def test_following_the_truncation_markers_walks_the_whole_document(tools, kb_root):
    """Each call hands back the range for the next, and the chain terminates.

    With a budget this small every call truncates, so this also pins that following the
    marker makes progress rather than looping on the same page forever.
    """
    (kb_root / "config.toml").write_text(
        'embedder = "dummy"\nreranker = "dummy"\nmax_document_chars = 150\n', encoding="utf-8"
    )

    seen, pages, guard = [], "", 0
    while guard < 5:
        guard += 1
        out = tools.read_document("CS240", "slides.pdf", pages=pages)
        seen += re.findall(r"^\[p(\d+)\]", out, re.M)
        marker = re.search(r'pages="(\d+-)"', out)
        if not marker:
            break
        pages = marker.group(1)

    assert seen == ["1", "2", "3"], "pages must arrive once each, in order"


def test_search_can_be_scoped_to_one_document(tools):
    out = tools.search_course("CS240", "rotations", source_file="slides.pdf")

    assert "in slides.pdf" in out
    assert "slides.pdf" in out


def test_search_scoped_to_an_unknown_document_names_the_real_ones(tools):
    out = tools.search_course("CS240", "rotations", source_file="ghost.pdf")

    assert "No document named 'ghost.pdf'" in out
    assert "slides.pdf" in out


def test_unscoped_search_output_is_unchanged(tools):
    """The acceptance-tested default path must not shift because a parameter was added."""
    assert tools.search_course("CS240", "rotations") == tools.search_course(
        "CS240", "rotations", source_file=""
    )


# --------------------------------------------------------------------------- #
# Looking at a page
# --------------------------------------------------------------------------- #


@pytest.fixture
def tools_with_sources(kb_root, monkeypatch):
    """Like ``tools``, but with the PDF in the course's raw/ folder.

    ``page_image`` reads the document itself, and ``kb ingest`` indexes a file without
    copying it into the course — only ``kb sync`` puts sources under raw/. So a course
    built by ingest alone can be searched but not looked at, which is exactly what this
    fixture makes explicit by not being the default one.
    """
    from courserag.cli import main

    (kb_root / "config.toml").write_text(
        'embedder = "dummy"\nreranker = "dummy"\n', encoding="utf-8"
    )
    monkeypatch.chdir(kb_root)
    monkeypatch.setenv("COURSE_KB_ROOT", str(kb_root))
    assert main(["sync", "CS240", "--from", str(FIXTURES / "slides.pdf"),
                 "--category", "lecture"]) == 0
    import courserag.mcp_server as srv

    return srv


def test_page_image_returns_a_caption_and_an_image(tools_with_sources):
    blocks = tools_with_sources.page_image("CS240", "slides.pdf", 1)

    assert len(blocks) == 2
    assert isinstance(blocks[0], str) and "slides.pdf page 1" in blocks[0]
    content = blocks[1].to_image_content()
    assert content.mime_type == "image/png"
    assert content.data  # base64 payload, non-empty


def test_page_image_reports_an_out_of_range_page_as_text(tools_with_sources):
    """Errors come back as a message the model can act on, not an exception."""
    blocks = tools_with_sources.page_image("CS240", "slides.pdf", 999)

    assert len(blocks) == 1
    assert "out of range" in blocks[0]


def test_page_image_on_an_unknown_course_names_the_real_ones(tools_with_sources):
    blocks = tools_with_sources.page_image("NOPE", "slides.pdf", 1)
    assert "CS240" in blocks[0]


def test_page_image_on_a_file_that_was_never_copied_into_the_course(tools):
    """Ingested-from-elsewhere means searchable but not viewable; say which."""
    blocks = tools.page_image("CS240", "slides.pdf", 1)

    assert len(blocks) == 1
    assert "no source file" in blocks[0]


def test_page_overview_lists_every_page_without_rendering(tools_with_sources):
    out = tools_with_sources.page_overview("CS240", "slides.pdf")

    assert "pages" in out and "drawings" in out
    assert "Median vector drawings" in out


def test_page_overview_on_an_unknown_file_explains_itself(tools_with_sources):
    assert "no source file" in tools_with_sources.page_overview("CS240", "ghost.pdf")


def test_the_image_tools_are_registered_with_the_server(tools_with_sources):
    names = {t.name for t in tools_with_sources.server._tool_manager.list_tools()}
    assert {"page_image", "page_overview"} <= names


# --------------------------------------------------------------------------- #
# The server's own instructions
# --------------------------------------------------------------------------- #


def test_the_instructions_point_at_the_image_tools():
    """Discovery of page_image rests entirely on this text.

    There is no signal in a search result saying "this page is a diagram" — four
    candidate detectors were measured against real decks and all of them failed, so
    the instructions are the only thing telling a model that reading the extracted
    text of a drawn page is worse than useless. Losing this paragraph would silently
    return the server to answering AVL questions from a run of bare numbers.
    """
    import courserag.mcp_server as srv

    instructions = srv.server.instructions
    assert "page_image" in instructions
    assert "page_overview" in instructions
    # The concrete failure, not just the tool name: the point is *why* to look.
    assert "misleading" in instructions


def test_the_instructions_reach_a_client_over_stdio(kb_root, tmp_path):
    """In-process is not the delivery path; the initialize response is."""
    request = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "0"},
                },
            }
        )
        + "\n"
    )
    import os

    result = subprocess.run(
        [sys.executable, "-m", "courserag.mcp_server"],
        input=request,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        env={**os.environ, "COURSE_KB_ROOT": str(kb_root)},
    )
    line = next(ln for ln in result.stdout.splitlines() if '"result"' in ln)
    instructions = json.loads(line)["result"].get("instructions", "")

    assert "page_image" in instructions, instructions


# --------------------------------------------------------------------------- #
# Per-course memory
# --------------------------------------------------------------------------- #


def test_notes_ride_along_on_every_read_of_a_course(tools):
    """A note nobody sees until they look it up is a file, not memory."""
    tools.remember("CS240", "The prof writes n for input size, never N.")

    for out in (
        tools.search_course("CS240", "skip lists"),
        tools.read_document("CS240", "slides.pdf"),
        tools.course_info("CS240"),
    ):
        assert "never N" in out, out


def test_search_results_lead_with_the_notes(tools):
    tools.remember("CS240", "Module 7 was skipped this term.")

    out = tools.search_course("CS240", "skip lists")

    assert out.index("Module 7 was skipped") < out.index("passage(s) from CS240")


def test_a_course_with_no_notes_reads_exactly_as_before(tools):
    out = tools.search_course("CS240", "skip lists")
    assert out.startswith("Top ") or out.startswith("WARNING")


def test_course_info_says_how_to_record_a_note_when_there_are_none(tools):
    assert "remember()" in tools.course_info("CS240")


def test_remember_stores_and_reports_the_note(tools):
    out = tools.remember("CS240", "A4 is worth 20%, not the 15% on the syllabus.")

    assert out.startswith("Recorded for CS240:")
    assert "A4 is worth 20%" in out


def test_remember_recognises_a_fact_it_already_knows(tools):
    tools.remember("CS240", "The prof writes n, never N.")

    out = tools.remember("CS240", "The prof writes n, never N.")

    assert "Already known" in out


@pytest.mark.parametrize("bad", ["", "   "])
def test_remember_refuses_an_empty_note(tools, bad):
    assert "Not recorded" in tools.remember("CS240", bad)


def test_remember_refuses_a_note_the_size_of_a_document(tools):
    assert "Not recorded" in tools.remember("CS240", "x" * 5000)


def test_remember_on_an_unknown_course_names_the_real_ones(tools):
    out = tools.remember("NOPE", "a fact")
    assert "CS240" in out and "No course named" in out


def test_remember_is_the_only_tool_that_writes(tools):
    """The boundary is load-bearing: an agent may accumulate notes, not alter a course."""
    names = {t.name for t in tools.server._tool_manager.list_tools()}
    assert "remember" in names
    # Nothing here ingests, deletes, or syncs — those stay in the CLI.
    assert not {"ingest", "delete", "sync", "init_course"} & names


def test_notes_written_by_the_cli_are_visible_to_the_server(kb_root, tools):
    """One file, two front-ends; `kb notes` and remember() must not diverge."""
    (kb_root / "courses" / "CS240" / "COURSE.md").write_text(
        "- 2026-03-01: Written by hand.\n", encoding="utf-8"
    )

    assert "Written by hand" in tools.search_course("CS240", "skip lists")
