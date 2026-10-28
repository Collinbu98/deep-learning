"""Small, dependency-free vector similarity primitives.

This module is deliberately a flat collection of functions over sequences of
numbers. It is not a vector database. The corpus in this project is tiny, so
an exhaustive scan of all vectors is both simpler and fast enough.

Definitions
-----------
For vectors ``a`` and ``b`` of the same dimension ``n``:

* dot product: ``a . b = sum(a[i] * b[i])``
* Euclidean norm: ``||a|| = sqrt(a . a)``
* cosine similarity::

      cos(a, b) = (a . b) / (||a|| * ||b||)

Cosine similarity is the cosine of the angle between the vectors. It ignores
their magnitudes and lies in ``[-1, 1]`` for real vectors. Text embeddings are
commonly compared this way because the direction of an embedding carries the
semantic signal, while its length can vary with the encoder or with text
length. If every vector is L2-normalised (``||a|| = ||b|| = 1``), then
``cos(a, b) = a . b`` exactly; normalising at encode time and taking dot
products is therefore a common optimisation.

Validation
----------
Every function rejects empty vectors, mismatched dimensions, and non-finite
values (``nan`` or ``inf``). Cosine similarity and normalisation additionally
reject zero vectors, whose direction is undefined. Errors are ``ValueError``
with a message naming the problem.
"""

from __future__ import annotations

import math

__all__ = [
    "dot",
    "norm",
    "normalize",
    "cosine_similarity",
    "rank_by_similarity",
]


def _vector(values, name="vector"):
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real numbers") from error
    if not vector:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _same_dimension(a, b):
    if len(a) != len(b):
        raise ValueError(
            f"dimension mismatch: {len(a)} != {len(b)}"
        )


def dot(a, b):
    """Return the dot product of two equal-length vectors."""
    left = _vector(a, "a")
    right = _vector(b, "b")
    _same_dimension(left, right)
    return sum(x * y for x, y in zip(left, right))


def norm(vector):
    """Return the Euclidean (L2) norm of ``vector``."""
    values = _vector(vector)
    return math.sqrt(sum(value * value for value in values))


def normalize(vector):
    """Return ``vector`` scaled to unit L2 norm.

    A zero vector has no direction and is rejected.
    """
    values = _vector(vector)
    length = math.sqrt(sum(value * value for value in values))
    if length == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return [value / length for value in values]


def cosine_similarity(a, b):
    """Return the cosine similarity between two non-zero vectors."""
    left = _vector(a, "a")
    right = _vector(b, "b")
    _same_dimension(left, right)
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot(left, right) / (left_norm * right_norm)


def rank_by_similarity(query, vectors, top_k=5, min_score=None):
    """Rank vectors by cosine similarity to ``query``.

    Returns a list of ``(index, score)`` pairs, best first. Equal scores are
    ordered by the vector's index so the result is deterministic.

    ``top_k <= 0`` returns ``[]``. If ``vectors`` is empty, ``[]`` is
    returned without validating the query. ``min_score``, when supplied,
    drops scores strictly below it.
    """
    if top_k <= 0:
        return []
    vector_list = list(vectors)
    if not vector_list:
        return []
    query_vector = _vector(query, "query")

    scored = []
    for index, vector in enumerate(vector_list):
        score = cosine_similarity(query_vector, _vector(vector, f"vectors[{index}]"))
        if min_score is not None and score < min_score:
            continue
        scored.append((index, score))

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:top_k]
