import pytest

from minigrad.chunking import Document, chunk_documents
from minigrad.embeddings import EmbeddingIndex, HashingEmbeddingBackend
from minigrad.rag import RAGPipeline
from minigrad.semantic_rag import (
    HybridRAGPipeline,
    SemanticRAGPipeline,
    calibrate_similarity_threshold,
)

class FakeEmbeddingBackend:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def encode(self, texts, batch_size=32):
        self.calls.append(list(texts))
        return [self.mapping[text] for text in texts]

DOCUMENTS = [
    Document(doc_id="alpha", source="alpha.txt", text="Alpha passage is first."),
    Document(doc_id="beta", source="beta.txt", text="Beta passage is second."),
]

MAPPING = {
    "Alpha passage is first.": [1.0, 0.0],
    "Beta passage is second.": [0.0, 1.0],
    "What is alpha?": [1.0, 0.0],
    "initial text paraphrase": [0.8, 0.2],
    "unknown topic": [0.1, 0.1],
    "calibration one": [1.0, 0.0],
    "calibration two": [0.6, 0.8],
}


def _semantic(**kwargs):
    backend = FakeEmbeddingBackend(MAPPING)
    defaults = {"min_similarity": 0.5}
    defaults.update(kwargs)
    return SemanticRAGPipeline.from_documents(
        DOCUMENTS, backend, chunk_size=400, overlap=0, **defaults
    )


def test_semantic_retrieval_uses_cosine_and_attribution():
    pipeline = _semantic()
    passages = pipeline.retrieve("What is alpha?", top_k=2)
    assert [passage.chunk.doc_id for passage in passages] == ["alpha", "beta"]
    assert [passage.rank for passage in passages] == [1, 2]
    assert passages[0].score == pytest.approx(1.0)
    assert passages[1].score == pytest.approx(0.0)


def test_semantic_abstention_uses_its_own_threshold():
    pipeline = _semantic(min_similarity=0.5)
    abstained, reason = pipeline.should_abstain("initial text paraphrase")
    assert not abstained
    assert reason == ""

    pipeline = _semantic(min_similarity=0.9)
    abstained, reason = pipeline.should_abstain("unknown topic")
    assert abstained
    assert "below min_similarity" in reason


def test_semantic_answer_reuses_extractive_answerer():
    pipeline = _semantic()
    answer = pipeline.answer("What is alpha?")
    assert not answer.abstained
    assert "Alpha passage" in answer.text
    assert answer.sentences[0].doc_id == "alpha"


def test_semantic_empty_question_abstains():
    pipeline = _semantic()
    abstained, reason = pipeline.should_abstain("   ")
    assert abstained
    assert "no tokens" in reason


def test_calibration_uses_out_of_domain_maxima():
    backend = FakeEmbeddingBackend(MAPPING)
    chunks = chunk_documents(DOCUMENTS, chunk_size=400, overlap=0)
    index = EmbeddingIndex(chunks, backend)
    threshold, scores = calibrate_similarity_threshold(
        index, ["calibration one", "calibration two"], percentile=100.0
    )
    assert scores == [pytest.approx(1.0), pytest.approx(0.8)]
    assert threshold == pytest.approx(1.0)

    threshold, _ = calibrate_similarity_threshold(
        index, ["calibration one", "calibration two"], percentile=50.0
    )
    assert threshold == pytest.approx(0.8)


def test_calibration_validation():
    backend = FakeEmbeddingBackend(MAPPING)
    index = EmbeddingIndex(
        chunk_documents(DOCUMENTS, chunk_size=400, overlap=0), backend
    )
    with pytest.raises(ValueError):
        calibrate_similarity_threshold(index, [])
    with pytest.raises(ValueError):
        calibrate_similarity_threshold(index, ["calibration one"], percentile=0)
    with pytest.raises(ValueError):
        calibrate_similarity_threshold(index, ["calibration one"], margin=-0.1)


def test_hybrid_fuses_rankings_and_requires_identical_chunks():
    bm25 = RAGPipeline.from_documents(DOCUMENTS, chunk_size=400, overlap=0)
    semantic = _semantic()
    hybrid = HybridRAGPipeline(bm25, semantic)
    passages = hybrid.retrieve("What is alpha?", top_k=2)
    assert passages[0].chunk.doc_id == "alpha"
    assert [passage.rank for passage in passages] == [1, 2]

    other = RAGPipeline.from_documents(
        [Document(doc_id="alpha", source="alpha.txt", text="Different text.")],
        chunk_size=400,
        overlap=0,
    )
    with pytest.raises(ValueError):
        HybridRAGPipeline(other, semantic)


def test_hybrid_lets_semantic_rescue_a_lexical_miss():
    bm25 = RAGPipeline.from_documents(DOCUMENTS, chunk_size=400, overlap=0)
    semantic = _semantic(min_similarity=0.5)
    hybrid = HybridRAGPipeline(bm25, semantic)

    assert bm25.retrieve("initial text paraphrase") == []
    abstained, reason = hybrid.should_abstain("initial text paraphrase")
    assert not abstained
    assert hybrid.retrieve("initial text paraphrase")[0].chunk.doc_id == "alpha"
    answer = hybrid.answer("initial text paraphrase")
    assert answer.abstained
    assert "no retrieved sentence" in answer.reason


def test_hybrid_abstains_only_when_both_retrievers_abstain():
    bm25 = RAGPipeline.from_documents(DOCUMENTS, chunk_size=400, overlap=0)
    semantic = _semantic(min_similarity=0.9)
    hybrid = HybridRAGPipeline(bm25, semantic)

    abstained, reason = hybrid.should_abstain("unknown topic")
    assert abstained
    assert "both retrievers abstained" in reason
    assert hybrid.answer("unknown topic").abstained


def test_semantic_pipeline_works_with_hashing_backend():
    pipeline = SemanticRAGPipeline.from_documents(
        DOCUMENTS,
        HashingEmbeddingBackend(dimension=64),
        chunk_size=400,
        overlap=0,
        min_similarity=-1.0,
    )
    assert pipeline.retrieve("Alpha passage", top_k=2)
    assert pipeline.index.dimension == 64
