import math

import pytest

from minigrad.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("The quick, brown FOX!") == ["the", "quick", "brown", "fox"]


def test_tokenize_strips_ascii_and_unicode_possessives():
    assert tokenize("Saturn's rings and Mars' moons") == [
        "saturn",
        "rings",
        "and",
        "mars",
        "moons",
    ]
    assert tokenize("Venus\u2019s clouds") == ["venus", "clouds"]


def test_tokenize_keeps_letters_digits_and_marks_but_not_underscores():
    assert tokenize("x_1 caf\u00e9 42") == ["x", "1", "caf\u00e9", "42"]
    assert tokenize("") == []


def test_idf_is_positive_and_penalises_common_terms():
    index = BM25Index(["rare term", "common term", "common term"])
    assert index.idf["rare"] > index.idf["common"] > 0.0


def test_idf_matches_definition():
    index = BM25Index(["a", "a b", "b c"])
    n_docs = 3
    df_a = 2
    expected = math.log(1.0 + (n_docs - df_a + 0.5) / (df_a + 0.5))
    assert index.idf["a"] == pytest.approx(expected)


def test_rare_term_match_outranks_common_term_match():
    index = BM25Index(
        [
            "the common planet",
            "the common planet again",
            "the common planet once more",
            "the common planet yet again",
            "the unique caloris basin",
        ]
    )
    common = index.search("common planet", top_k=3)
    unique = index.search("caloris", top_k=3)
    assert common[0][0] in (0, 1, 2, 3)
    assert unique[0][0] == 4
    assert unique[0][1] > common[0][1]


def test_term_frequency_increases_score():
    index = BM25Index(["alpha", "alpha alpha alpha"])
    low = index.score("alpha", 0)
    high = index.score("alpha", 1)
    assert high > low


def test_length_normalisation_penalises_long_documents():
    index = BM25Index(["alpha", "alpha " + "filler " * 50])
    assert index.score("alpha", 0) > index.score("alpha", 1)


def test_b_zero_disables_length_normalisation():
    index = BM25Index(["alpha", "alpha " + "filler " * 50], b=0.0)
    assert index.score("alpha", 0) == pytest.approx(index.score("alpha", 1))


def test_search_orders_by_score_then_document_index():
    index = BM25Index(["beta", "beta beta", "beta beta beta"])
    results = index.search("beta", top_k=3)
    assert [doc for doc, _ in results] == [2, 1, 0]
    assert results[0][1] > results[1][1] > results[2][1]

    tied = BM25Index(["same", "same"])
    assert [doc for doc, _ in tied.search("same")] == [0, 1]


def test_search_honours_top_k_and_min_score():
    index = BM25Index(["alpha", "alpha beta", "gamma"])
    assert len(index.search("alpha", top_k=2)) == 2
    assert index.search("alpha", top_k=0) == []
    assert index.search("alpha", top_k=-1) == []

    results = index.search("alpha", top_k=5)
    threshold = results[0][1]
    assert index.search("alpha", top_k=5, min_score=threshold) == []


def test_search_ignores_unknown_terms_duplicates_and_empty_input():
    index = BM25Index(["alpha beta", "beta gamma"])
    assert index.search("") == []
    assert index.search("missing") == []
    assert index.search("beta beta") == index.search("beta")
    assert index.search("beta", top_k=2)[0][0] == 0


def test_document_matching_some_query_terms_scores_below_full_match():
    index = BM25Index(["alpha beta", "alpha gamma"])
    full = index.score("alpha beta", 0)
    partial = index.score("alpha beta", 1)
    assert full > partial > 0.0
    assert [doc_index for doc_index, _ in index.search("alpha beta")] == [0, 1]


def test_tokens_match_exactly_so_stems_and_synonyms_do_not_match():
    index = BM25Index(["The moon orbits the planet."])
    assert index.search("moon")
    assert index.search("moons") == []
    assert index.search("satellite") == []


def test_specific_term_outranks_passages_sharing_only_generic_terms():
    index = BM25Index(
        [
            "The planet has a cratered surface.",
            "The planet has a cratered surface.",
            "The planet has a large crater called Caloris Basin.",
        ]
    )
    results = index.search("planet Caloris Basin", top_k=3)
    assert results[0][0] == 2
    assert all(score > 0.0 for _, score in results)


def test_empty_index_returns_no_results_and_safe_average_length():
    index = BM25Index([])
    assert index.avgdl == 0.0
    assert index.search("anything") == []


@pytest.mark.parametrize("k1,b", [(-0.1, 0.75), (1.5, -0.1), (1.5, 1.1)])
def test_invalid_parameters(k1, b):
    with pytest.raises(ValueError):
        BM25Index(["text"], k1=k1, b=b)
