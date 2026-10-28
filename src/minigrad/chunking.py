"""Deterministic character-level chunking for retrieval.

Retrieval systems split long documents into passages ("chunks") because a
whole document is usually too coarse a unit to score and too long to feed to
a model. This module provides the simplest useful scheme: fixed-size
character windows with an optional overlap.

Why characters, not tokens
--------------------------
Character windows need no tokenizer and behave identically for every corpus,
which keeps the retrieval experiment reproducible. The cost is that a window
can end in the middle of a word, and the same text can be counted twice when
windows overlap. Both are acceptable here and are documented rather than
hidden.

Window layout
-------------
``chunk_size`` is the maximum number of characters per chunk and ``overlap``
is how many characters neighbouring windows share, so the stride is
``chunk_size - overlap``. For text of length ``n``, windows start at
``0, stride, 2*stride, ...`` while ``start < n``. The final window is
truncated at ``n`` and may be shorter than ``chunk_size``. Every character of
the text therefore appears in at least one chunk, and with ``overlap == 0``
every character appears exactly once.

An empty (or whitespace-only) document produces no chunks. A document shorter
than ``chunk_size`` produces exactly one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Document",
    "Chunk",
    "chunk_text",
    "chunk_document",
    "chunk_documents",
]


@dataclass(frozen=True)
class Document:
    """A source document before chunking."""

    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A contiguous character window of a document."""

    doc_id: str
    source: str
    chunk_index: int
    text: str
    start: int
    end: int


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80):
    """Split ``text`` into ``(start, end, text)`` windows.

    Parameters
    ----------
    text:
        The text to split. Whitespace-only input yields an empty list.
    chunk_size:
        Maximum characters per window; must be at least 1.
    overlap:
        Characters shared by adjacent windows; must satisfy
        ``0 <= overlap < chunk_size``.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text or not text.strip():
        return []

    stride = chunk_size - overlap
    n = len(text)
    windows = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        windows.append((start, end, text[start:end]))
        if end == n:
            break
        start += stride
    return windows


def chunk_document(document: Document, chunk_size: int = 400, overlap: int = 80):
    """Chunk one document, preserving its id and source on every chunk."""
    return [
        Chunk(
            doc_id=document.doc_id,
            source=document.source,
            chunk_index=index,
            text=chunk,
            start=start,
            end=end,
        )
        for index, (start, end, chunk) in enumerate(
            chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        )
    ]


def chunk_documents(documents, chunk_size: int = 400, overlap: int = 80):
    """Chunk an iterable of documents in the order they are supplied."""
    chunks = []
    for document in documents:
        chunks.extend(
            chunk_document(document, chunk_size=chunk_size, overlap=overlap)
        )
    return chunks
