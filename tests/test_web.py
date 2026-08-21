"""The web UI is a front-end for ``kb sync``; these check it stays one.

Every endpoint here is exercised through the real app against a real (dummy-embedder)
course on disk, because the interesting failures are not in the HTTP layer — they are
in whether a drop, a delete, or a folder upload leaves the folder and the index
agreeing with each other.
"""

from __future__ import annotations

import time

import pytest

fastapi = pytest.importorskip("fastapi", reason='requires the [web] extra: pip install -e ".[web]"')
from fastapi.testclient import TestClient  # noqa: E402

from courserag.cli import main  # noqa: E402
from courserag.config import load_config  # noqa: E402
from courserag.ingest import open_course  # noqa: E402
from courserag.sync import source_root  # noqa: E402
from courserag.web.app import _safe_relpath, create_app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch, dummy_config):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "CS247"]) == 0
    return TestClient(create_app(load_config()))


@pytest.fixture
def cfg(tmp_path, monkeypatch, dummy_config):
    monkeypatch.chdir(tmp_path)
    return load_config()


def wait_for(client: TestClient, job_id: str, timeout: float = 20.0) -> dict:
    """Block until a background job finishes; fail loudly rather than hang."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["done"]:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def upload(client: TestClient, course: str, items: list[tuple[str, str]], **data) -> dict:
    """POST ``(relative_path, text)`` pairs the way the browser does.

    ``data`` has to be a dict of lists, not a list of pairs: httpx silently drops the
    file parts when given the latter alongside ``files``.
    """
    files = [("files", (path.split("/")[-1], text.encode(), "text/plain")) for path, text in items]
    form = {"paths": [path for path, _ in items], **data}
    res = client.post(f"/api/courses/{course}/files", files=files, data=form)
    assert res.status_code == 200, res.text
    return res.json()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def test_the_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "CourseRAG" in res.text


def test_course_list_reports_chunk_counts(client):
    body = client.get("/api/courses").json()
    assert [c["id"] for c in body["courses"]] == ["CS247"]
    assert body["courses"][0]["chunks"] == 0
    assert ".pdf" in body["extensions"] and ".txt" in body["extensions"]


def test_a_missing_course_is_a_404(client):
    assert client.get("/api/courses/NOPE").status_code == 404


def test_creating_a_course_through_the_api(client):
    res = client.post("/api/courses", data={"course_id": "PHIL121"})
    assert res.status_code == 200, res.text
    assert res.json()["dims"] == 64
    assert {c["id"] for c in client.get("/api/courses").json()["courses"]} == {"CS247", "PHIL121"}


@pytest.mark.parametrize("bad", ["", "  ", "../escape", "a/b", "a\\b", ".hidden"])
def test_a_course_id_cannot_be_a_path(client, bad):
    # 400 from the handler's own check, 422 when FastAPI rejects the empty form value
    # before it gets there. Either way nothing is created.
    assert client.post("/api/courses", data={"course_id": bad}).status_code in (400, 422)
    assert [c["id"] for c in client.get("/api/courses").json()["courses"]] == ["CS247"]


def test_creating_a_course_twice_conflicts(client):
    assert client.post("/api/courses", data={"course_id": "CS247"}).status_code == 409


# --------------------------------------------------------------------------- #
# Uploading
# --------------------------------------------------------------------------- #


def test_a_dropped_file_lands_in_raw_and_gets_indexed(client, cfg):
    res = upload(client, "CS247", [("notes.txt", "Skip lists pick tower height by coin flip.\n")])
    job = wait_for(client, res["job"]["id"])

    assert job["state"] == "done", job
    assert (source_root(cfg, "CS247") / "notes.txt").exists()
    _manifest, store = open_course(cfg, "CS247")
    assert store.source_file_counts()["notes.txt"] > 0


def test_a_dropped_folder_keeps_its_shape_and_becomes_a_category(client, cfg):
    res = upload(
        client,
        "CS247",
        [
            ("lecture/05.txt", "Skip lists pick tower height by coin flip.\n"),
            ("lecture/06.txt", "Red-black trees rebalance with rotations.\n"),
        ],
    )
    wait_for(client, res["job"]["id"])

    assert (source_root(cfg, "CS247") / "lecture" / "05.txt").exists()
    detail = client.get("/api/courses/CS247").json()
    assert {f["category"] for f in detail["files"]} == {"lecture"}


def test_an_explicit_category_overrides_the_dropped_folder(client, cfg):
    res = upload(
        client,
        "CS247",
        [("whatever/05.txt", "Skip lists pick tower height.\n")],
        category="exam",
    )
    wait_for(client, res["job"]["id"])

    assert (source_root(cfg, "CS247") / "exam" / "05.txt").exists()


def test_files_with_no_parser_are_reported_not_stored(client, cfg):
    res = upload(client, "CS247", [("archive.zip", "nope\n")])

    assert res["skipped"] == ["archive.zip"]
    assert res["job"] is None
    assert not (source_root(cfg, "CS247") / "archive.zip").exists()


def test_uploading_the_same_file_again_is_a_no_op(client, cfg):
    text = "Skip lists pick tower height by coin flip.\n"
    wait_for(client, upload(client, "CS247", [("notes.txt", text)])["job"]["id"])
    _manifest, store = open_course(cfg, "CS247")
    before = store.count()

    job = wait_for(client, upload(client, "CS247", [("notes.txt", text)])["job"]["id"])

    assert "Already up to date." in job["lines"]
    _manifest, store = open_course(cfg, "CS247")
    assert store.count() == before


def test_replacing_a_file_reindexes_it_instead_of_appending(client, cfg):
    wait_for(client, upload(client, "CS247", [("n.txt", "Skip lists use coin flips.\n")])["job"]["id"])
    wait_for(client, upload(client, "CS247", [("n.txt", "Red-black trees use rotations.\n")])["job"]["id"])

    _manifest, store = open_course(cfg, "CS247")
    stored = " ".join(r.text for r in store.get_all())
    assert "rotations" in stored and "coin flips" not in stored


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("lecture/05.pdf", "lecture/05.pdf"),
        ("../../../etc/passwd", "etc/passwd"),
        ("/absolute/path.pdf", "absolute/path.pdf"),
        ("a/../../b/c.pdf", "a/b/c.pdf"),
        ("..", ""),
        (".ssh/config", "config"),
        ("windows\\style\\x.pdf", "windows/style/x.pdf"),
    ],
)
def test_a_browser_supplied_path_cannot_escape_the_course_folder(raw, expected):
    assert str(_safe_relpath(raw)) == (expected or ".")


def test_a_traversal_upload_stays_inside_raw(client, cfg):
    res = upload(client, "CS247", [("../../../pwned.txt", "Skip lists pick tower height.\n")])
    wait_for(client, res["job"]["id"])

    root = source_root(cfg, "CS247")
    assert (root / "pwned.txt").exists()
    assert not (root.parent.parent.parent / "pwned.txt").exists()


# --------------------------------------------------------------------------- #
# Viewing a source file
# --------------------------------------------------------------------------- #


def test_a_source_file_is_served_back_for_the_viewer(client, cfg):
    text = "Skip lists pick tower height by coin flip.\n"
    wait_for(client, upload(client, "CS247", [("lecture/notes.txt", text)])["job"]["id"])

    res = client.get("/api/courses/CS247/raw/notes.txt")

    assert res.status_code == 200
    assert res.text == text
    assert res.headers["content-type"].startswith("text/plain")
    # inline, so the browser renders it instead of downloading it
    assert res.headers["content-disposition"] == 'inline; filename="notes.txt"'


def test_a_pdf_is_served_with_a_pdf_content_type(client, cfg):
    """The viewer relies on the browser's built-in PDF handling, which needs the type."""
    root = source_root(cfg, "CS247")
    (root / "slides.pdf").write_bytes(b"%PDF-1.4\n% not a real pdf, only the type matters\n")

    res = client.get("/api/courses/CS247/raw/slides.pdf")

    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"


def test_a_file_in_a_category_folder_is_found_by_basename(client, cfg):
    """The table shows basenames, so that is what the viewer asks for."""
    wait_for(client, upload(client, "CS247", [("exam/final.txt", "Question 1.\n")])["job"]["id"])

    assert client.get("/api/courses/CS247/raw/final.txt").status_code == 200


@pytest.mark.parametrize(
    "attack",
    [
        "../../../../etc/passwd",
        "..%2f..%2f..%2fetc%2fpasswd",
        "/etc/passwd",
        "....//....//etc/passwd",
        "manifest.json",  # a real file, but one directory up from raw/
        "sync.json",
    ],
)
def test_the_viewer_cannot_read_anything_outside_raw(client, cfg, attack):
    """Names are looked up in the scan, never joined onto a path, so these match nothing."""
    wait_for(client, upload(client, "CS247", [("n.txt", "Skip lists.\n")])["job"]["id"])

    res = client.get(f"/api/courses/CS247/raw/{attack}")

    assert res.status_code == 404, res.text


def test_viewing_an_unknown_file_is_a_404(client):
    assert client.get("/api/courses/CS247/raw/ghost.pdf").status_code == 404


def test_viewing_in_an_unknown_course_is_a_404(client):
    assert client.get("/api/courses/NOPE/raw/anything.pdf").status_code == 404


def test_an_indexed_file_deleted_from_disk_is_no_longer_viewable(client, cfg):
    """The UI disables its name for exactly this reason; the API agrees."""
    wait_for(client, upload(client, "CS247", [("n.txt", "Skip lists.\n")])["job"]["id"])
    (source_root(cfg, "CS247") / "n.txt").unlink()

    assert client.get("/api/courses/CS247/raw/n.txt").status_code == 404


# --------------------------------------------------------------------------- #
# Deleting
# --------------------------------------------------------------------------- #


def test_deleting_removes_the_chunks_and_the_file(client, cfg):
    wait_for(client, upload(client, "CS247", [("n.txt", "Skip lists use coin flips.\n")])["job"]["id"])

    body = client.request("DELETE", "/api/courses/CS247/files/n.txt").json()

    assert body["chunks_removed"] > 0
    assert body["removed_from_disk"] is True
    assert not (source_root(cfg, "CS247") / "n.txt").exists()
    _manifest, store = open_course(cfg, "CS247")
    assert store.count() == 0


def test_deleting_a_file_that_was_dropped_but_not_yet_indexed(client, cfg):
    """It exists in raw/ and nowhere else; removing it must still work."""
    (source_root(cfg, "CS247") / "stray.txt").write_text("unsynced\n", encoding="utf-8")

    body = client.request("DELETE", "/api/courses/CS247/files/stray.txt").json()

    assert body["chunks_removed"] == 0
    assert not (source_root(cfg, "CS247") / "stray.txt").exists()


def test_deleting_something_that_does_not_exist_is_a_404(client):
    assert client.request("DELETE", "/api/courses/CS247/files/ghost.txt").status_code == 404


def test_a_delete_is_not_undone_by_the_next_sync(client, cfg):
    wait_for(client, upload(client, "CS247", [("n.txt", "Skip lists use coin flips.\n")])["job"]["id"])
    client.request("DELETE", "/api/courses/CS247/files/n.txt")

    job = wait_for(client, client.post("/api/courses/CS247/sync").json()["job"]["id"])

    assert "Already up to date." in job["lines"]
    _manifest, store = open_course(cfg, "CS247")
    assert store.count() == 0


# --------------------------------------------------------------------------- #
# Syncing files changed outside the browser
# --------------------------------------------------------------------------- #


def test_a_file_added_in_finder_shows_as_pending_then_indexes(client, cfg):
    (source_root(cfg, "CS247") / "dropped.txt").write_text(
        "Skip lists pick tower height by coin flip.\n", encoding="utf-8"
    )

    detail = client.get("/api/courses/CS247").json()
    pending = [f for f in detail["files"] if f["on_disk"] and not f["indexed"]]
    assert [f["name"] for f in pending] == ["dropped.txt"]

    wait_for(client, client.post("/api/courses/CS247/sync").json()["job"]["id"])

    detail = client.get("/api/courses/CS247").json()
    assert detail["files"][0]["indexed"] and detail["files"][0]["chunks"] > 0


def test_a_file_deleted_in_finder_shows_as_missing_then_syncs_away(client, cfg):
    wait_for(client, upload(client, "CS247", [
        ("keep.txt", "Skip lists use coin flips.\n"),
        ("drop.txt", "Red-black trees use rotations.\n"),
    ])["job"]["id"])
    (source_root(cfg, "CS247") / "drop.txt").unlink()

    detail = client.get("/api/courses/CS247").json()
    missing = [f["name"] for f in detail["files"] if f["indexed"] and not f["on_disk"]]
    assert missing == ["drop.txt"]

    wait_for(client, client.post("/api/courses/CS247/sync").json()["job"]["id"])

    assert [f["name"] for f in client.get("/api/courses/CS247").json()["files"]] == ["keep.txt"]


def test_the_empty_folder_guard_reaches_the_ui_as_a_failed_job(client, cfg):
    """From a browser this is a normal thing to hit, so it must not 500."""
    wait_for(client, upload(client, "CS247", [("n.txt", "Skip lists use coin flips.\n")])["job"]["id"])
    (source_root(cfg, "CS247") / "n.txt").unlink()

    job = wait_for(client, client.post("/api/courses/CS247/sync").json()["job"]["id"])

    assert job["state"] == "error"
    assert "missing folder" in job["error"]
    _manifest, store = open_course(cfg, "CS247")
    assert store.count() > 0  # refused, so nothing was lost


def test_duplicate_basenames_are_surfaced_to_the_ui(client, cfg):
    root = source_root(cfg, "CS247")
    for sub in ("lecture", "tutorial"):
        (root / sub).mkdir(parents=True)
        (root / sub / "05.txt").write_text("Skip lists.\n", encoding="utf-8")

    detail = client.get("/api/courses/CS247").json()

    assert sorted(detail["duplicates"]["05.txt"]) == ["lecture/05.txt", "tutorial/05.txt"]


def test_an_unknown_job_id_is_a_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404


# --------------------------------------------------------------------------- #
# Exporting a course from the UI
# --------------------------------------------------------------------------- #


def test_the_export_endpoint_returns_a_valid_archive(client, cfg, tmp_path):
    wait_for(client, upload(client, "CS247", [("lecture/n.txt", "Skip lists.\n")])["job"]["id"])

    res = client.get("/api/courses/CS247/export")

    assert res.status_code == 200
    assert res.headers["content-type"] == "application/zip"
    assert 'filename="CS247-courserag.zip"' in res.headers["content-disposition"]

    archive = tmp_path / "got.zip"
    archive.write_bytes(res.content)
    from courserag.transfer import read_metadata

    assert read_metadata(archive)["course"] == "CS247"


def test_exporting_a_course_with_no_files_is_a_400(client):
    assert client.get("/api/courses/CS247/export").status_code == 400


def test_exporting_an_unknown_course_is_a_404(client):
    assert client.get("/api/courses/NOPE/export").status_code == 404


# --------------------------------------------------------------------------- #
# Changing a category from the UI
# --------------------------------------------------------------------------- #


def patch_category(client: TestClient, course: str, name: str, category: str):
    return client.patch(f"/api/courses/{course}/files/{name}", data={"category": category})


def test_changing_a_category_moves_the_file_and_reindexes_it(client, cfg):
    wait_for(client, upload(client, "CS247", [("lecture/n.txt", "Skip lists.\n")])["job"]["id"])

    res = patch_category(client, "CS247", "n.txt", "exam")

    assert res.status_code == 200, res.text
    assert res.json()["category"] == "exam"
    wait_for(client, res.json()["job"]["id"])

    assert (source_root(cfg, "CS247") / "exam" / "n.txt").exists()
    _manifest, store = open_course(cfg, "CS247")
    # The chunks carry the new category, not just the folder.
    assert store.source_categories() == {"n.txt": "exam"}


def test_the_detail_payload_reflects_the_new_category(client, cfg):
    wait_for(client, upload(client, "CS247", [("lecture/n.txt", "Skip lists.\n")])["job"]["id"])

    wait_for(client, patch_category(client, "CS247", "n.txt", "exam").json()["job"]["id"])

    detail = client.get("/api/courses/CS247").json()
    assert [f["category"] for f in detail["files"]] == ["exam"]


def test_moving_to_a_brand_new_category_creates_it(client, cfg):
    wait_for(client, upload(client, "CS247", [("n.txt", "Skip lists.\n")])["job"]["id"])

    wait_for(client, patch_category(client, "CS247", "n.txt", "tutorial").json()["job"]["id"])

    detail = client.get("/api/courses/CS247").json()
    assert "tutorial" in detail["categories"]


@pytest.mark.parametrize("hostile", ["../../../pwned", "/etc", "..", ".hidden"])
def test_a_hostile_category_cannot_place_a_file_outside_raw(client, cfg, hostile):
    wait_for(client, upload(client, "CS247", [("n.txt", "Skip lists.\n")])["job"]["id"])

    res = patch_category(client, "CS247", "n.txt", hostile)
    wait_for(client, res.json()["job"]["id"])

    root = source_root(cfg, "CS247").resolve()
    landed = [p for p in root.rglob("n.txt")]
    assert landed and all(root in p.resolve().parents for p in landed)


def test_recategorizing_a_file_that_is_not_on_disk_is_a_404(client):
    assert patch_category(client, "CS247", "ghost.txt", "exam").status_code == 404


def test_recategorizing_in_an_unknown_course_is_a_404(client):
    assert patch_category(client, "NOPE", "n.txt", "exam").status_code == 404


# --------------------------------------------------------------------------- #
# Renaming from the UI
# --------------------------------------------------------------------------- #


def patch_file(client: TestClient, course: str, name: str, **fields):
    return client.patch(f"/api/courses/{course}/files/{name}", data=fields)


def test_renaming_moves_the_chunks_to_the_new_name(client, cfg):
    wait_for(client, upload(client, "CS247", [("lecture/old.txt", "Skip lists.\n")])["job"]["id"])

    res = patch_file(client, "CS247", "old.txt", new_name="new.txt")

    assert res.status_code == 200, res.text
    assert res.json()["name"] == "new.txt"
    wait_for(client, res.json()["job"]["id"])

    assert (source_root(cfg, "CS247") / "lecture" / "new.txt").exists()
    _manifest, store = open_course(cfg, "CS247")
    assert set(store.source_file_counts()) == {"new.txt"}


def test_renaming_and_recategorizing_in_one_call_costs_one_sync(client, cfg):
    """Two moves, one reindex — syncing between them would index the file twice."""
    wait_for(client, upload(client, "CS247", [("lecture/old.txt", "Skip lists.\n")])["job"]["id"])

    res = patch_file(client, "CS247", "old.txt", new_name="new.txt", category="exam")
    job = wait_for(client, res.json()["job"]["id"])

    assert (source_root(cfg, "CS247") / "exam" / "new.txt").exists()
    _manifest, store = open_course(cfg, "CS247")
    assert store.source_categories() == {"new.txt": "exam"}
    # One add, not two.
    assert sum(1 for line in job["lines"] if line.startswith("ingest:")) == 1


def test_renaming_onto_an_existing_name_is_a_conflict(client, cfg):
    wait_for(client, upload(client, "CS247", [
        ("a.txt", "Skip lists.\n"), ("b.txt", "Red-black trees.\n")])["job"]["id"])

    res = patch_file(client, "CS247", "a.txt", new_name="b.txt")

    assert res.status_code == 409
    assert "already has a file named" in res.json()["detail"]
    _manifest, store = open_course(cfg, "CS247")
    assert set(store.source_file_counts()) == {"a.txt", "b.txt"}


def test_renaming_away_the_extension_is_a_conflict(client, cfg):
    wait_for(client, upload(client, "CS247", [("n.txt", "Skip lists.\n")])["job"]["id"])

    res = patch_file(client, "CS247", "n.txt", new_name="n.zip")

    assert res.status_code == 409
    assert (source_root(cfg, "CS247") / "n.txt").exists()


@pytest.mark.parametrize("attempt", ["../escaped.txt", "/etc/escaped.txt", "sub/dir/x.txt"])
def test_a_rename_cannot_relocate_the_file(client, cfg, attempt):
    wait_for(client, upload(client, "CS247", [("lecture/n.txt", "Skip lists.\n")])["job"]["id"])

    res = patch_file(client, "CS247", "n.txt", new_name=attempt)
    wait_for(client, res.json()["job"]["id"])

    root = source_root(cfg, "CS247").resolve()
    landed = [p for p in root.rglob("*") if p.is_file()]
    assert landed and all(root in p.resolve().parents for p in landed)
    assert all(p.parent.name == "lecture" for p in landed)


def test_a_patch_that_changes_nothing_queues_no_job(client):
    wait_for(client, upload(client, "CS247", [("n.txt", "Skip lists.\n")])["job"]["id"])

    res = patch_file(client, "CS247", "n.txt", new_name="n.txt")

    assert res.status_code == 200
    assert res.json()["job"] is None


def test_renaming_a_file_that_is_not_on_disk_is_a_404(client):
    assert patch_file(client, "CS247", "ghost.txt", new_name="x.txt").status_code == 404
