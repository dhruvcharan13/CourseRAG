"""Parser registry lookup and the reference TextParser."""

from __future__ import annotations

from pathlib import Path

import pytest

from course_kb.parsing import ParsedDocument, get_parser_for
from course_kb.parsing.text import TextParser


@pytest.mark.parametrize("name", ["notes.txt", "readme.md", "UPPER.TXT"])
def test_registry_resolves_text_extensions(name):
    parser = get_parser_for(Path(name))
    assert isinstance(parser, TextParser)


def test_registry_raises_on_unknown_extension():
    with pytest.raises(ValueError):
        get_parser_for(Path("slides.pptx"))


def test_text_parser_whole_file_one_element(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("line one\nline two\n", encoding="utf-8")

    doc = get_parser_for(path).parse(path)
    assert isinstance(doc, ParsedDocument)
    assert doc.source_file == str(path)
    assert len(doc.elements) == 1
    element = doc.elements[0]
    assert element.text == "line one\nline two\n"
    assert element.page is None
    assert element.title == "note"
