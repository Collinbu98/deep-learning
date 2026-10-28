import pytest

from minigrad.chunking import (
    Chunk,
    Document,
    chunk_document,
    chunk_documents,
    chunk_text,
)


def test_windows_use_stride_chunk_size_minus_overlap():
    windows = chunk_text("abcdefghij", chunk_size=4, overlap=1)
    assert windows == [
        (0, 4, "abcd"),
        (3, 7, "defg"),
        (6, 10, "ghij"),
    ]


def test_zero_overlap_covers_every_character_exactly_once():
    text = "abcdefghij"
    windows = chunk_text(text, chunk_size=3, overlap=0)
    assert "".join(chunk for _, _, chunk in windows) == text
    assert [start for start, _, _ in windows] == [0, 3, 6, 9]


def test_overlap_windows_cover_every_character():
    text = "the quick brown fox jumps over the lazy dog"
    windows = chunk_text(text, chunk_size=10, overlap=4)
    covered = set()
    for start, end, _ in windows:
        covered.update(range(start, end))
    assert covered == set(range(len(text)))
    assert all(end - start <= 10 for start, end, _ in windows)


def test_short_text_is_one_chunk():
    assert chunk_text("hello", chunk_size=400) == [(0, 5, "hello")]
    assert chunk_text("hello", chunk_size=5, overlap=0) == [(0, 5, "hello")]


def test_empty_and_whitespace_text_produce_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t ") == []


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)],
)
def test_invalid_window_parameters(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=chunk_size, overlap=overlap)


def test_chunk_document_preserves_metadata_and_offsets():
    doc = Document(doc_id="mars", source="mars.txt", text="abcdefghij")
    chunks = chunk_document(doc, chunk_size=4, overlap=0)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.doc_id == "mars" for chunk in chunks)
    assert all(chunk.source == "mars.txt" for chunk in chunks)
    assert [(chunk.start, chunk.end, chunk.text) for chunk in chunks] == [
        (0, 4, "abcd"),
        (4, 8, "efgh"),
        (8, 10, "ij"),
    ]
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_chunk_documents_keeps_input_order_and_restarts_indices():
    docs = [
        Document(doc_id="a", source="a.txt", text="aaaa"),
        Document(doc_id="b", source="b.txt", text="bbbb"),
    ]
    chunks = chunk_documents(docs, chunk_size=2, overlap=0)
    assert [(c.doc_id, c.chunk_index, c.text) for c in chunks] == [
        ("a", 0, "aa"),
        ("a", 1, "aa"),
        ("b", 0, "bb"),
        ("b", 1, "bb"),
    ]
