import math

import pytest

from minigrad.vector_search import (
    cosine_similarity,
    dot,
    norm,
    normalize,
    rank_by_similarity,
)


def test_dot_product_and_norm():
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert dot([1, -1], [-1, 1]) == -2
    assert norm([3, 4]) == 5
    assert norm([0, 0]) == 0


def test_cosine_similarity_known_values():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([1, 1], [2, 2]) == pytest.approx(1.0)


def test_cosine_is_dot_for_normalized_vectors():
    a = normalize([3.0, 4.0])
    b = normalize([-1.0, 2.0])
    assert norm(a) == pytest.approx(1.0)
    assert norm(b) == pytest.approx(1.0)
    assert cosine_similarity(a, b) == pytest.approx(dot(a, b))


def test_rank_by_similarity_orders_scores_and_breaks_ties_by_index():
    query = [1.0, 0.0]
    vectors = [[0.0, 1.0], [2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
    ranked = rank_by_similarity(query, vectors, top_k=4)
    assert [index for index, _ in ranked] == [1, 2, 0, 3]
    assert ranked[0][1] == pytest.approx(1.0)
    assert ranked[2][1] == pytest.approx(0.0)
    assert ranked[3][1] == pytest.approx(-1.0)


def test_rank_by_similarity_honours_top_k_and_min_score():
    query = [1.0, 0.0]
    vectors = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    assert len(rank_by_similarity(query, vectors, top_k=2)) == 2
    assert rank_by_similarity(query, vectors, top_k=0) == []
    assert rank_by_similarity(query, vectors, top_k=-1) == []
    assert rank_by_similarity(query, vectors, min_score=0.5) == [
        (0, pytest.approx(1.0)),
        (1, pytest.approx(math.sqrt(0.5))),
    ]


def test_rank_by_similarity_empty_collection():
    assert rank_by_similarity([1.0], []) == []


@pytest.mark.parametrize(
    "left,right",
    [
        ([], []),
        ([1.0, 2.0], [1.0]),
        ([], [1.0]),
        ([1.0, 2.0], []),
    ],
)
def test_dimension_and_empty_validation(left, right):
    with pytest.raises(ValueError):
        dot(left, right)
    with pytest.raises(ValueError):
        cosine_similarity(left, right)


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")], [1.0, -float("inf")]])
def test_norm_rejects_empty_and_non_finite(vector):
    with pytest.raises(ValueError):
        norm(vector)


def test_cosine_and_normalize_reject_zero_vectors():
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="zero vector"):
        normalize([0.0, 0.0])


def test_rank_validates_query_and_every_candidate():
    with pytest.raises(ValueError):
        rank_by_similarity([], [[1.0]])
    with pytest.raises(ValueError):
        rank_by_similarity([1.0], [[1.0, 2.0]])
    with pytest.raises(ValueError):
        rank_by_similarity([1.0], [[float("nan")]])
    with pytest.raises(ValueError):
        rank_by_similarity([0.0], [[0.0]])
