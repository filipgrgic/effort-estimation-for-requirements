import pytest
from pipeline.chunker import chunk_text
from config import MAX_CHARS


tenthOfMax = MAX_CHARS // 10


def test_returns_single_chunk_when_text_is_shorter_than_max_chars():

    text = "The system shall allow users to log in."
    result = chunk_text(text)

    assert len(result) == 1
    assert result[0] == text


def test_returns_three_chunks_when_text_has_three_paragraphs():
    text = (
        "Section1 " * tenthOfMax
        + "\n\n"
        + "Section2 " * tenthOfMax
        + "\n\n"
        + "Section3 " * tenthOfMax
    )
    result = chunk_text(text)

    assert len(result) == 3


def test_splits_by_line_when_text_has_no_paragraphs_but_has_line_breaks():
    text = (
        "Section1 " * tenthOfMax
        + "\n"
        + "Section2 " * tenthOfMax
        + "\n"
        + "Section3 " * tenthOfMax
    )
    result = chunk_text(text)

    assert len(result) == 3


def test_splits_by_sentence_when_text_has_no_linebreaks():
    text = "a" * (MAX_CHARS - 2) + ". " + "b" * (MAX_CHARS - 2) + ". " + "c"
    result = chunk_text(text)

    assert len(result) == 3
    assert result[2] == "c"


def test_splits_by_max_chunk_size_when_sentence_is_too_long():
    text = "a" * MAX_CHARS + "b"
    result = chunk_text(text)

    assert len(result) == 2
    assert result[1] == "b"


def test_ignores_empty_paragraphs_when_chunking():
    text = (
        "Section1 " * tenthOfMax
        + "\n\n"
        + " " * tenthOfMax
        + "\n\n"
        + "Section2 " * tenthOfMax
    )
    result = chunk_text(text)

    assert len(result) == 2


def test_never_exceeds_max_chunk_size_when_building_chunks():
    text = (
        "a" * (MAX_CHARS - tenthOfMax - 4)
        + "\n\n"
        + "b" * tenthOfMax
        + "\n\n"
        + "c" * tenthOfMax
    )

    result = chunk_text(text)

    assert len(result) == 2
    assert result[1] == "c" * tenthOfMax
