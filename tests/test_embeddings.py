import math

import pytest

from minigrad.chunking import Document, chunk_documents
from minigrad.embeddings import (
    EmbeddingIndex,
    HashingEmbeddingBackend,
    SentenceTransformerEmbeddings,
)


class FakeEmbeddingBackend:
    """Deterministic vector lookup that records calls for cache assertions."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []
        self.batch_sizes = []

    def encode(self, texts, batch_size=32):
        self.calls.append(list(texts))
        self.batch_sizes.append(batch_size)
        return [self.mapping[text] for text in texts]


def _chunks():
    documents = [
        Document(doc_id="a", source="a.txt", text="Alpha passage."),
        Document(doc_id="b", source="b.txt", text="Beta passage."),
    ]
    return chunk_documents(documents, chunk_size=400, overlap=0)


def test_index_embeds_documents_once_at_build_time():
    chunks = _chunks()
    backend = FakeEmbeddingBackend(
        {
            "Alpha passage.": [1.0, 0.0],
            "Beta passage.": [0.0, 1.0],
            "alpha query": [1.0, 0.0],
        }
    )
    index = EmbeddingIndex(chunks, backend, batch_size=7)
    assert backend.calls == [["Alpha passage.", "Beta passage."]]
    assert backend.batch_sizes == [7]
    assert index.dimension == 2

    assert index.search("alpha query", top_k=2)[0] == (0, pytest.approx(1.0))
    assert backend.calls == [["Alpha passage.", "Beta passage."], ["alpha query"]]


def test_query_embeddings_are_cached():
    chunks = _chunks()
    backend = FakeEmbeddingBackend(
        {
            "Alpha passage.": [1.0, 0.0],
            "Beta passage.": [0.0, 1.0],
            "alpha query": [1.0, 0.0],
        }
    )
    index = EmbeddingIndex(chunks, backend)
    first = index.search("alpha query")
    second = index.search("alpha query")
    assert first == second
    assert backend.calls.count(["alpha query"]) == 1


def test_index_search_uses_cosine_and_min_score():
    chunks = _chunks()
    backend = FakeEmbeddingBackend(
        {
            "Alpha passage.": [10.0, 0.0],
            "Beta passage.": [1.0, 1.0],
            "query": [1.0, 0.0],
        }
    )
    index = EmbeddingIndex(chunks, backend)
    assert index.search("query", top_k=2) == [
        (0, pytest.approx(1.0)),
        (1, pytest.approx(math.sqrt(0.5))),
    ]
    assert index.search("query", min_score=0.9) == [(0, pytest.approx(1.0))]


def test_empty_index_is_safe():
    backend = FakeEmbeddingBackend({})
    index = EmbeddingIndex([], backend)
    assert index.dimension == 0
    assert index.search("anything") == []
    assert backend.calls == []


def test_index_rejects_inconsistent_backend_output():
    chunks = _chunks()
    with pytest.raises(ValueError, match="dimension mismatch"):
        EmbeddingIndex(
            chunks,
            FakeEmbeddingBackend(
                {"Alpha passage.": [1.0, 0.0], "Beta passage.": [1.0]}
            ),
        )
    with pytest.raises(ValueError, match="non-finite"):
        EmbeddingIndex(
            chunks,
            FakeEmbeddingBackend(
                {
                    "Alpha passage.": [1.0, 0.0],
                    "Beta passage.": [float("nan"), 1.0],
                }
            ),
        )


def test_hashing_backend_is_deterministic_and_normalized():
    backend = HashingEmbeddingBackend(dimension=32)
    first = backend.encode(["Saturn has rings."])[0]
    second = backend.encode(["Saturn has rings."])[0]
    different = backend.encode(["Mars has moons."])[0]
    assert first == second
    assert first != different
    assert len(first) == 32
    assert sum(value * value for value in first) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        HashingEmbeddingBackend(dimension=0)


def test_sentence_transformer_backend_is_lazy():
    backend = SentenceTransformerEmbeddings(batch_size=4)
    assert backend._model is None
    assert backend.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert backend.normalize_embeddings is True
    with pytest.raises(ValueError):
        SentenceTransformerEmbeddings(batch_size=0)
