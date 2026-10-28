"""Semantic retrieval over the same chunks and questions as Experiment 11.

The pipeline mirrors :class:`minigrad.rag.RAGPipeline`, but replaces BM25 with
cosine similarity over pretrained sentence embeddings. The evaluator, chunking
and extractive answerer are reused so that retrieval is the only changed
variable.

Abstention is different by construction. A BM25 score and a cosine similarity
live on different scales: a BM25 score is unbounded, while cosine similarity
is in ``[-1, 1]``. Reusing the BM25 threshold would be meaningless. Here
abstention uses an explicit cosine threshold ``min_similarity``, which should
be calibrated on questions that are disjoint from the evaluation set (see
:func:`calibrate_similarity_threshold`).

Cosine similarity is not a calibrated probability of answerability. A model
can give a moderate similarity to a passage that does not contain the answer.
The threshold is a heuristic for a particular corpus and embedding model.
"""

from __future__ import annotations

import math

from minigrad.bm25 import BM25Index, tokenize
from minigrad.chunking import chunk_documents
from minigrad.embeddings import EmbeddingIndex
from minigrad.rag import (
    DEFAULT_STOPWORDS,
    Answer,
    ExtractiveAnswerer,
    RetrievedPassage,
)

__all__ = [
    "SemanticRAGPipeline",
    "HybridRAGPipeline",
    "calibrate_similarity_threshold",
]


class SemanticRAGPipeline:
    """Chunk a corpus, embed the chunks once, and retrieve by cosine."""

    def __init__(
        self,
        chunks,
        embedding_index: EmbeddingIndex,
        top_k: int = 3,
        min_similarity: float = 0.30,
        stopwords=DEFAULT_STOPWORDS,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        self.chunks = list(chunks)
        self.index = embedding_index
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.stopwords = frozenset(stopwords)
        self.idf = (
            BM25Index([chunk.text for chunk in self.chunks]).idf
            if self.chunks
            else {}
        )

    @classmethod
    def from_documents(
        cls,
        documents,
        backend,
        chunk_size: int = 400,
        overlap: int = 80,
        batch_size: int = 32,
        **kwargs,
    ):
        documents = list(documents)
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        index = EmbeddingIndex(chunks, backend, batch_size=batch_size)
        pipeline = cls(chunks, index, **kwargs)
        pipeline.chunk_size = chunk_size
        pipeline.overlap = overlap
        pipeline.backend = backend
        pipeline.document_texts = {
            document.doc_id: document.text for document in documents
        }
        pipeline.chunks_by_doc = {}
        for chunk in chunks:
            pipeline.chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)
        return pipeline

    def retrieve(self, question: str, top_k=None):
        """Return cosine-ranked passages, best first."""
        if top_k is None:
            top_k = self.top_k
        return [
            RetrievedPassage(chunk=self.chunks[index], score=score, rank=rank)
            for rank, (index, score) in enumerate(
                self.index.search(question, top_k=top_k), start=1
            )
        ]

    def should_abstain(self, question: str, passages=None, top_k=None):
        """Return ``(abstain, reason)`` using an explicit cosine threshold."""
        if not tokenize(question):
            return True, "question has no tokens"
        if passages is None:
            passages = self.retrieve(question, top_k=top_k)
        if not passages:
            return True, "no passages"
        top = passages[0].score
        if top < self.min_similarity:
            return True, (
                f"top cosine similarity {top:.3f} below "
                f"min_similarity {self.min_similarity:.3f}"
            )
        return False, ""

    def answer(self, question: str, top_k=None, max_sentences: int = 2):
        """Retrieve, abstain if unsupported, then extract sentences."""
        if top_k is None:
            top_k = self.top_k
        passages = self.retrieve(question, top_k=top_k)
        abstained, reason = self.should_abstain(
            question, passages=passages, top_k=top_k
        )
        if abstained:
            return Answer(
                question=question,
                text="",
                passages=tuple(passages),
                abstained=True,
                reason=reason,
            )
        answerer = ExtractiveAnswerer(
            self.idf, self.stopwords, max_sentences=max_sentences
        )
        return answerer.answer(
            question,
            passages,
            document_texts=getattr(self, "document_texts", None),
            chunks_by_doc=getattr(self, "chunks_by_doc", None),
        )


class HybridRAGPipeline:
    """Combine BM25 and cosine rankings with reciprocal rank fusion (RRF).

    For a chunk at rank ``r`` (1-based) in a retriever's list, RRF contributes
    ``1 / (k + r)`` with ``k = 60`` by default. Contributions from both
    retrievers are summed. RRF uses only ranks, not raw scores, so an
    unbounded BM25 score and a cosine score never have to be placed on the
    same scale. ``k`` damps the influence of the very top ranks; a larger
    ``k`` makes the two lists' contributions more uniform.

    The fused score is a **ranking score, not a calibrated probability of
    relevance**. It is not bounded by 1 and must not be thresholded as if it
    were a similarity. Retrieval and abstention are deliberately separate:
    abstention asks whether either base retriever found acceptable evidence.

    Abstention requires both base retrievers to abstain. This lets the hybrid
    answer a lexical paraphrase that only the embedding model retrieves, while
    still declining when neither method finds support. The trade-off is that
    an over-confident base retriever can force an answer on an unsupported
    question; the experiment reports false acceptances separately so this
    failure mode stays visible.
    """

    def __init__(self, bm25_pipeline, semantic_pipeline, rrf_k: int = 60):
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        left = bm25_pipeline.chunks
        right = semantic_pipeline.chunks
        if len(left) != len(right) or any(
            (a.doc_id, a.chunk_index, a.text) != (b.doc_id, b.chunk_index, b.text)
            for a, b in zip(left, right)
        ):
            raise ValueError("hybrid retrievers must use identical chunks")
        self.bm25 = bm25_pipeline
        self.semantic = semantic_pipeline
        self.chunks = list(left)
        self._chunk_positions = {
            (chunk.doc_id, chunk.chunk_index): index
            for index, chunk in enumerate(self.chunks)
        }
        self.rrf_k = rrf_k
        self.top_k = bm25_pipeline.top_k
        self.idf = bm25_pipeline.retriever.idf

    def retrieve(self, question: str, top_k=None):
        if top_k is None:
            top_k = self.top_k
        if top_k <= 0 or not self.chunks:
            return []

        pool = len(self.chunks)
        lists = [
            self.bm25.retrieve(question, top_k=pool),
            self.semantic.retrieve(question, top_k=pool),
        ]
        scores = [0.0] * pool
        for passages in lists:
            for passage in passages:
                index = self._chunk_positions[
                    (passage.chunk.doc_id, passage.chunk.chunk_index)
                ]
                scores[index] += 1.0 / (self.rrf_k + passage.rank)

        ranked = sorted(
            (index for index in range(pool) if scores[index] > 0.0),
            key=lambda index: (-scores[index], index),
        )[:top_k]
        return [
            RetrievedPassage(chunk=self.chunks[index], score=scores[index], rank=rank)
            for rank, index in enumerate(ranked, start=1)
        ]

    def should_abstain(self, question: str, passages=None, top_k=None):
        bm25_abstain, bm25_reason = self.bm25.should_abstain(question)
        semantic_abstain, semantic_reason = self.semantic.should_abstain(question)
        if bm25_abstain and semantic_abstain:
            return True, (
                f"both retrievers abstained: BM25 ({bm25_reason}); "
                f"semantic ({semantic_reason})"
            )
        return False, ""

    def answer(self, question: str, top_k=None, max_sentences: int = 2):
        if top_k is None:
            top_k = self.top_k
        passages = self.retrieve(question, top_k=top_k)
        abstained, reason = self.should_abstain(
            question, passages=passages, top_k=top_k
        )
        if abstained:
            return Answer(
                question=question,
                text="",
                passages=tuple(passages),
                abstained=True,
                reason=reason,
            )
        answerer = ExtractiveAnswerer(
            self.idf, self.bm25.stopwords, max_sentences=max_sentences
        )
        return answerer.answer(
            question,
            passages,
            document_texts=getattr(self.bm25, "document_texts", None),
            chunks_by_doc=getattr(self.bm25, "chunks_by_doc", None),
        )


def calibrate_similarity_threshold(
    index: EmbeddingIndex,
    calibration_questions,
    percentile: float = 95.0,
    margin: float = 0.0,
):
    """Choose a cosine threshold from calibration questions.

    The questions must be disjoint from the evaluation set and should be
    in-domain but unanswerable, so that the threshold reflects the similarity
    that a plausible but unsupported question can reach. For each calibration
    question the highest cosine similarity to any chunk
    is computed. The threshold is the requested nearest-rank percentile of
    those maxima plus ``margin``, rounded upward to three decimal places.

    The calibration questions must be disjoint from the evaluation questions;
    otherwise the threshold is fitted to the test set. Even with a disjoint
    set this is a small, corpus-specific calibration, not a guarantee of
    calibrated probabilities.

    Returns ``(threshold, top_scores)``.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    questions = list(calibration_questions)
    if not questions:
        raise ValueError("at least one calibration question is required")

    top_scores = []
    for question in questions:
        results = index.search(question, top_k=1)
        if results:
            top_scores.append(results[0][1])
    if not top_scores:
        raise ValueError("calibration questions produced no scores")

    ordered = sorted(top_scores)
    rank = math.ceil(percentile / 100.0 * len(ordered))
    value = ordered[max(rank - 1, 0)] + margin
    threshold = min(1.0, max(-1.0, math.ceil(value * 1000.0) / 1000.0))
    return threshold, top_scores
