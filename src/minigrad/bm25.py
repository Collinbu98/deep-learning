"""Okapi BM25 lexical retrieval, implemented from scratch.

BM25 scores a document by the query terms it contains. A term contributes
more when it is rare in the collection (high inverse document frequency) and
when it occurs often in the document, with diminishing returns. A length
normalisation prevents long documents from winning simply because they
contain more words.

Scoring
-------
For a document ``d`` and query terms ``q`` (each distinct term counted once):

    score(d, q) = sum over t in q of
        idf(t) * tf(t, d) * (k1 + 1)
        / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

The inverse document frequency uses the always-positive form

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

so a term that appears in every document still has a small positive weight
instead of zero or negative weight. ``tf`` is the raw term frequency.
Defaults are the usual ``k1 = 1.5`` and ``b = 0.75``.

Limitations
-----------
BM25 matches surface tokens only. It has no notion of synonyms, morphology,
or word order, so a paraphrase that shares no tokens with the corpus receives
no score. That limitation is intentional in this experiment: it is what the
evaluation measures.
"""

from __future__ import annotations

import math
import re
from collections import Counter

__all__ = ["tokenize", "BM25Index"]

_POSSESSIVE = re.compile(r"['\u2019]s\b", re.IGNORECASE)
_WORD = re.compile(r"[^\W_]+")


def tokenize(text: str):
    """Lowercase word tokens, with English possessives removed.

    ``"Saturn's rings"`` becomes ``["saturn", "rings"]``. Apostrophes other
    than the possessive are treated as separators, as is every character that
    is not a Unicode letter, digit, or combining mark. Underscores are
    separators too.
    """
    text = _POSSESSIVE.sub("", text.lower())
    return _WORD.findall(text)


class BM25Index:
    """An in-memory inverted index over a collection of documents."""

    def __init__(self, documents, k1: float = 1.5, b: float = 0.75):
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self.documents = list(documents)
        self.k1 = k1
        self.b = b

        self.term_frequencies = [Counter(tokenize(doc)) for doc in self.documents]
        self.doc_lengths = [sum(tf.values()) for tf in self.term_frequencies]
        self.avgdl = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )

        self.postings = {}
        for doc_index, tf in enumerate(self.term_frequencies):
            for term, count in tf.items():
                self.postings.setdefault(term, []).append((doc_index, count))

        n_docs = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (n_docs - len(entries) + 0.5) / (len(entries) + 0.5))
            for term, entries in self.postings.items()
        }

    def score(self, query: str, doc_index: int):
        """BM25 score of one document for ``query``."""
        terms = set(tokenize(query))
        if not terms or not self.documents:
            return 0.0
        length = self.doc_lengths[doc_index]
        norm = 1.0 - self.b
        if self.avgdl > 0:
            norm += self.b * length / self.avgdl
        total = 0.0
        for term in terms:
            tf = self.term_frequencies[doc_index].get(term, 0)
            if tf:
                total += (
                    self.idf[term]
                    * tf
                    * (self.k1 + 1.0)
                    / (tf + self.k1 * norm)
                )
        return total

    def search(self, query: str, top_k: int = 5, min_score=None):
        """Return ``(doc_index, score)`` pairs, best first.

        Ties are broken by document index so results are deterministic.
        ``top_k <= 0``, an empty query, or an empty index returns ``[]``.
        Documents scoring zero are omitted. If ``min_score`` is given, scores
        at or below it are omitted as well.
        """
        if top_k <= 0 or not self.documents:
            return []
        terms = set(tokenize(query))
        if not terms:
            return []

        candidates = set()
        for term in terms:
            for doc_index, _ in self.postings.get(term, ()):
                candidates.add(doc_index)

        scored = []
        for doc_index in candidates:
            value = self.score(query, doc_index)
            if value <= 0.0:
                continue
            if min_score is not None and value <= min_score:
                continue
            scored.append((doc_index, value))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]
