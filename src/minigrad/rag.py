"""A small retrieval-augmented generation (RAG) pipeline.

This module connects three independently testable pieces:

1. chunking (:mod:`minigrad.chunking`) splits documents into passages;
2. BM25 (:mod:`minigrad.bm25`) scores passages against a question;
3. an extractive answerer selects sentences from the retrieved passages.

There is no neural model and no generation. The "answer" is copied from the
corpus, which makes every claim traceable to a source passage and keeps the
experiment reproducible. The trade-off is that the system can only restate
what the corpus already says; it cannot summarise, paraphrase, or combine
facts into a new sentence.

Abstention
----------
A retrieval system that always answers will answer unanswerable questions
with irrelevant text. :class:`RAGPipeline` therefore abstains when:

* the question has no tokens, or no tokens outside the stopword list;
* retrieval returns no passage with a positive score;
* the best score is below ``min_score``;
* fewer than ``min_coverage`` of the question's content terms appear in the
  retrieved passages (default 0.6).

The coverage rule is a cheap, corpus-dependent heuristic. It does not detect
every unsupported question, and a threshold that works for this corpus is not
a general solution to hallucination. It is included to make the failure mode
visible and measurable.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from minigrad.bm25 import BM25Index, tokenize
from minigrad.chunking import Chunk, Document, chunk_documents

__all__ = [
    "Question",
    "RetrievedPassage",
    "ExtractedSentence",
    "Answer",
    "load_corpus",
    "load_questions",
    "RAGPipeline",
    "ExtractiveAnswerer",
    "evaluate_retrieval",
    "DEFAULT_STOPWORDS",
]

# Question words and function words are removed before measuring whether the
# retrieved evidence covers a question. "many" and "much" are included because
# they ask for a quantity ("how many moons") rather than naming a topic; left
# in, they match unrelated sentences such as "contains many small rocky
# bodies". The list is deliberately small and English-specific.
DEFAULT_STOPWORDS = frozenset(
    """
    a about all an and any are as at be been both but by can could did do does
    each for from had has have he her here his how i if in into is it its many
    me more most much my no not of on or other our she should than that the
    their them then there these they this those to too us was we were what when
    where which who whom why will with would yes you your
    """.split()
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    relevant_docs: tuple = ()
    answerable: bool = True
    notes: str = ""


@dataclass(frozen=True)
class RetrievedPassage:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class ExtractedSentence:
    text: str
    doc_id: str
    source: str
    chunk_index: int
    score: float


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sentences: tuple = ()
    passages: tuple = ()
    abstained: bool = False
    reason: str = ""


def load_corpus(directory):
    """Load ``*.txt`` files from ``directory`` as documents.

    Files are sorted by name so document ids and chunk order are
    deterministic. The file stem is the ``doc_id``; ``source`` is the path
    relative to ``directory``. Leading and trailing whitespace is stripped.
    """
    directory = Path(directory)
    documents = []
    for path in sorted(directory.glob("*.txt")):
        documents.append(
            Document(
                doc_id=path.stem,
                source=str(path.relative_to(directory)),
                text=path.read_text(encoding="utf-8").strip(),
            )
        )
    return documents


def load_questions(path):
    """Load question definitions from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = []
    for item in data["questions"]:
        questions.append(
            Question(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                relevant_docs=tuple(item.get("relevant_docs", ())),
                answerable=bool(item.get("answerable", True)),
                notes=item.get("notes", ""),
            )
        )
    return questions


class RAGPipeline:
    """Chunk a corpus, retrieve passages, and abstain when unsupported."""

    def __init__(
        self,
        chunks,
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 3,
        min_score: float = 0.0,
        min_coverage: float = 0.6,
        stopwords=DEFAULT_STOPWORDS,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be between 0 and 1")
        self.chunks = list(chunks)
        self.top_k = top_k
        self.min_score = min_score
        self.min_coverage = min_coverage
        self.stopwords = frozenset(stopwords)
        self.retriever = BM25Index([chunk.text for chunk in self.chunks], k1=k1, b=b)

    @classmethod
    def from_documents(
        cls, documents, chunk_size: int = 400, overlap: int = 80, **kwargs
    ):
        """Build a pipeline by chunking ``documents`` with the given sizes."""
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        pipeline = cls(chunks, **kwargs)
        pipeline.chunk_size = chunk_size
        pipeline.overlap = overlap
        pipeline.document_texts = {
            document.doc_id: document.text for document in documents
        }
        pipeline.chunks_by_doc = {}
        for chunk in chunks:
            pipeline.chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)
        return pipeline

    def retrieve(self, question: str, top_k=None):
        """Return scored passages, best first. Zero-score passages are dropped."""
        if top_k is None:
            top_k = self.top_k
        return [
            RetrievedPassage(chunk=self.chunks[index], score=score, rank=rank)
            for rank, (index, score) in enumerate(
                self.retriever.search(question, top_k=top_k), start=1
            )
        ]

    def content_terms(self, question: str):
        """Distinct query terms that are not stopwords, in first-seen order."""
        seen = set()
        terms = []
        for term in tokenize(question):
            if term in self.stopwords or term in seen:
                continue
            seen.add(term)
            terms.append(term)
        return terms

    def should_abstain(self, question: str, passages=None, top_k=None):
        """Return ``(abstain, reason)`` for a question."""
        if top_k is None:
            top_k = self.top_k
        terms = self.content_terms(question)
        if not terms:
            return True, "question has no content terms"
        if passages is None:
            passages = self.retrieve(question, top_k=top_k)
        if not passages:
            return True, "no matching passages"
        top = passages[0].score
        if top <= 0.0:
            return True, "no passage has a positive score"
        if top < self.min_score:
            return True, "top score below min_score"

        seen = set()
        for passage in passages:
            seen.update(tokenize(passage.chunk.text))
        covered = sum(1 for term in terms if term in seen)
        coverage = covered / len(terms)
        if coverage < self.min_coverage:
            return True, (
                f"content-term coverage {coverage:.2f} below "
                f"min_coverage {self.min_coverage:.2f}"
            )
        return False, ""

    def answer(self, question: str, top_k=None, max_sentences: int = 2):
        """Retrieve, abstain if unsupported, otherwise extract sentences."""
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
            self.retriever.idf, self.stopwords, max_sentences=max_sentences
        )
        return answerer.answer(
            question,
            passages,
            document_texts=getattr(self, "document_texts", None),
            chunks_by_doc=getattr(self, "chunks_by_doc", None),
        )


def _split_sentences(text):
    """Return ``(sentence, offset, index)`` triples.

    Splits after ``.``, ``!`` or ``?`` only when followed by whitespace, so
    decimals such as ``9.9`` stay inside one sentence. Abbreviations followed
    by a space (for example ``Dr.``) still split; that limitation is noted in
    the experiment report.
    """
    sentences = []
    position = 0
    for match in _SENTENCE_END.finditer(text):
        segment = text[position : match.start()]
        sentence = segment.strip()
        if sentence:
            offset = position + (len(segment) - len(segment.lstrip()))
            sentences.append((sentence, offset, len(sentences)))
        position = match.end()
    segment = text[position:]
    sentence = segment.strip()
    if sentence:
        offset = position + (len(segment) - len(segment.lstrip()))
        sentences.append((sentence, offset, len(sentences)))
    return sentences


class ExtractiveAnswerer:
    """Select the highest-scoring corpus sentences for a question.

    A sentence is ranked first by how many distinct question content terms it
    contains. Ties are broken by the sum of those terms' IDF weights, each
    down-weighted when the term also occurs in the other retrieved passages:
    a word shared by every passage does not help choose between their
    sentences. Sentences are de-duplicated, including near-duplicates caused
    by chunk boundaries. The selected sentences are returned verbatim, each
    with its source document and chunk index.

    Only sentences that match *as many distinct question terms as the best
    sentence* are quoted. Weaker matches are treated as insufficient evidence
    rather than padding: for "What is the atmosphere of Earth mostly made
    of?", a sentence about Jupiter that happens to contain "made" and
    "mostly" is not appended to the Earth sentence. The rule still quotes
    tied sentences from different documents, which is what allows a
    multi-document answer. It is a precision-oriented heuristic based on
    surface terms, not reasoning about what the answer means.
    """

    def __init__(
        self,
        idf,
        stopwords=DEFAULT_STOPWORDS,
        max_sentences: int = 2,
    ):
        if max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")
        self.idf = idf
        self.stopwords = frozenset(stopwords)
        self.max_sentences = max_sentences

    def _content_terms(self, question):
        return [
            term
            for term in dict.fromkeys(tokenize(question))
            if term not in self.stopwords
        ]

    def answer(self, question, passages, document_texts=None, chunks_by_doc=None):
        terms = self._content_terms(question)
        n_passages = len(passages)
        passage_terms = [set(tokenize(p.chunk.text)) for p in passages]
        passage_df = {}
        for terms_in_passage in passage_terms:
            for term in terms_in_passage:
                passage_df[term] = passage_df.get(term, 0) + 1

        def term_weight(term):
            idf = self.idf.get(term, 0.0)
            if idf <= 0.0 or n_passages == 0:
                return 0.0
            df = passage_df.get(term, 0)
            return idf * (1.0 - 0.9 * df / n_passages)

        if document_texts is None:
            sentence_sources = []
            for passage in passages:
                for sentence, _, sentence_index in _split_sentences(passage.chunk.text):
                    sentence_sources.append(
                        (sentence, sentence_index, passage, passage.chunk.chunk_index)
                    )
        else:
            sentence_sources = []
            seen_docs = set()
            for passage in passages:
                doc_id = passage.chunk.doc_id
                if doc_id in seen_docs or doc_id not in document_texts:
                    continue
                seen_docs.add(doc_id)
                for sentence, offset, sentence_index in _split_sentences(
                    document_texts[doc_id]
                ):
                    chunk_index = passage.chunk.chunk_index
                    if chunks_by_doc and doc_id in chunks_by_doc:
                        for chunk in chunks_by_doc[doc_id]:
                            if chunk.start <= offset < chunk.end:
                                chunk_index = chunk.chunk_index
                                break
                    sentence_sources.append(
                        (sentence, sentence_index, passage, chunk_index)
                    )

        candidates = []
        seen = []
        for sentence, sentence_index, passage, chunk_index in sentence_sources:
            text = " ".join(sentence.split())
            key = text.lower()
            if any(
                difflib.SequenceMatcher(None, key, other).ratio() >= 0.9
                for other in seen
            ):
                continue
            seen.append(key)
            sentence_terms = set(tokenize(text))
            matched = tuple(term for term in terms if term in sentence_terms)
            if not matched:
                continue
            candidates.append(
                {
                    "n_matched": len(matched),
                    "score": sum(term_weight(term) for term in matched),
                    "rank": passage.rank,
                    "sentence_index": sentence_index,
                    "text": text,
                    "passage": passage,
                    "chunk_index": chunk_index,
                    "matched": matched,
                }
            )

        if not candidates:
            return Answer(
                question=question,
                text="",
                passages=tuple(passages),
                abstained=True,
                reason="no retrieved sentence contains a question term",
            )

        candidates.sort(
            key=lambda item: (
                -item["n_matched"],
                -item["score"],
                item["rank"],
                item["sentence_index"],
            )
        )

        best_n_matched = candidates[0]["n_matched"]
        selected = [
            candidate
            for candidate in candidates[: self.max_sentences]
            if candidate["n_matched"] == best_n_matched
        ]

        extracted = tuple(
            ExtractedSentence(
                text=candidate["text"],
                doc_id=candidate["passage"].chunk.doc_id,
                source=candidate["passage"].chunk.source,
                chunk_index=candidate["chunk_index"],
                score=candidate["score"],
            )
            for candidate in selected
        )
        return Answer(
            question=question,
            text=" ".join(item.text for item in extracted),
            sentences=extracted,
            passages=tuple(passages),
        )


def evaluate_retrieval(pipeline: RAGPipeline, questions, k: int = 3):
    """Evaluate document-level retrieval and abstention.

    Relevance is document-level: a retrieved chunk is relevant when its
    ``doc_id`` is in the question's ``relevant_docs``. For answerable
    questions this reports:

    * ``recall`` - fraction of expected documents retrieved;
    * ``hit`` - 1 if every expected document was retrieved, else 0;
    * ``reciprocal_rank`` - 1 / rank of the first relevant chunk;
    * ``abstained`` - whether the pipeline declined to answer.

    Unanswerable questions (empty ``relevant_docs``) are excluded from the
    retrieval means and contribute to ``abstention_rate_unanswerable``.
    ``false_abstention_rate`` is the abstention rate over answerable
    questions. The pipeline's own ``top_k`` and thresholds are used for
    abstention, independently of the evaluation ``k``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    answerable_by_id = {question.id: question.answerable for question in questions}
    rows = []
    for question in questions:
        passages = pipeline.retrieve(question.question, top_k=k)
        doc_ids = [passage.chunk.doc_id for passage in passages]
        relevant = set(question.relevant_docs)
        abstained, reason = pipeline.should_abstain(question.question)

        if question.answerable and relevant:
            found = relevant.intersection(doc_ids)
            recall = len(found) / len(relevant)
            hit = 1.0 if len(found) == len(relevant) else 0.0
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(doc_ids, start=1):
                if doc_id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break
        else:
            recall = hit = reciprocal_rank = None

        rows.append(
            {
                "id": question.id,
                "category": question.category,
                "question": question.question,
                "retrieved_docs": doc_ids,
                "relevant_docs": sorted(relevant),
                "recall": recall,
                "hit": hit,
                "reciprocal_rank": reciprocal_rank,
                "abstained": abstained,
                "reason": reason,
            }
        )

    scored = [row for row in rows if row["recall"] is not None]
    unanswerable = [row for row in rows if not answerable_by_id[row["id"]]]
    answerable = [row for row in rows if answerable_by_id[row["id"]]]

    def mean(values):
        return sum(values) / len(values) if values else None

    return {
        "k": k,
        "rows": rows,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "recall_at_k": mean([row["recall"] for row in scored]),
        "hit_at_k": mean([row["hit"] for row in scored]),
        "mrr_at_k": mean([row["reciprocal_rank"] for row in scored]),
        "abstention_rate_unanswerable": mean(
            [row["abstained"] for row in unanswerable]
        ),
        "false_abstention_rate": mean([row["abstained"] for row in answerable]),
    }

