import pytest

from minigrad.bpe_tokenizer import BPETokenizer, UnknownCharacterError

UNK = "�"


def _repetitive_corpus():
    # Alphabet {a, b, c}. Pairs: (a,b)=10, (b,c)=10, (c,a)=9.
    # Ties are broken lexicographically, so merge order is (a,b) then (ab,c).
    return "abc" * 10


def _text_corpus():
    return "Hello world! Hello world again. Hello world!"


def test_fit_and_encode_small_fixed_corpus():
    # vocab_size=6 budgets 3 chars + 2 merges + 1 UNK.
    tok = BPETokenizer(vocab_size=6).fit(_repetitive_corpus())
    assert tok.alphabet == ["a", "b", "c"]
    a, b, c = 0, 1, 2
    # First merge: most frequent pair, lexicographic tie-break -> (a, b).
    # Second merge: (merged_ab, c).
    assert tok.merges == [(a, b), (3, c)]
    assert tok.vocab_size == 6  # 3 chars + 2 merges + UNK
    assert tok.unk_id == 5
    assert tok.token_char_length(tok.unk_id) == 1
    # "abc" collapses to the single token "abc".
    assert tok.encode("abc") == [4]
    assert tok.encode("abcabc") == [4, 4]
    # Text without the winning pair passes through as character ids.
    assert tok.encode("ca") == [c, a]
    assert tok.decode([4, 4]) == "abcabc"


def test_roundtrip_training_text():
    text = _text_corpus()
    tok = BPETokenizer(vocab_size=200).fit(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    # BPE must compress: fewer tokens than characters.
    assert len(ids) < len(text)
    # Every character of the training text is covered exactly once.
    assert sum(tok.token_char_length(i) for i in ids) == len(text)


def test_roundtrip_unseen_combinations_of_known_chars():
    training = "ab" * 10 + "cd" * 10
    tok = BPETokenizer(vocab_size=16).fit(training)
    # Every character was seen, but these sequences were not.
    for text in ["acbd", "dcab", "aaaa", "abcd", "c" * 30 + "a" * 30]:
        assert tok.decode(tok.encode(text)) == text


def test_empty_input():
    tok = BPETokenizer(vocab_size=16).fit(_repetitive_corpus())
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    with pytest.raises(ValueError):
        BPETokenizer(vocab_size=16).fit("")


def test_deterministic_vocabulary_and_merges():
    text = "the quick brown fox jumps over the lazy dog " * 5
    first = BPETokenizer(vocab_size=40).fit(text)
    second = BPETokenizer(vocab_size=40).fit(text)
    assert first.alphabet == second.alphabet
    assert first.merges == second.merges
    assert first.vocab_size == second.vocab_size
    sample = "jumps over the quick fox"
    assert first.encode(sample) == second.encode(sample)


def test_most_frequent_pair_merged_first():
    # "aab" * 20: (a,a) and (a,b) both occur 20 times -> tie -> the
    # lexicographically smaller (a, a) pair wins.
    tok = BPETokenizer(vocab_size=6).fit("aab" * 20)
    a_id = tok.alphabet.index("a")
    assert tok.merges[0] == (a_id, a_id)
    # "bbab" * 20: (b,b) occurs 40 times, strictly more than any other pair.
    tok2 = BPETokenizer(vocab_size=5).fit("bbab" * 20)
    b_id = tok2.alphabet.index("b")
    assert tok2.merges[0] == (b_id, b_id)
    assert tok2.encode("bb") == [tok2.alphabet_size]  # merged "bb" token id


def test_vocab_size_includes_unk_and_bounds():
    text = "abcabcabc abcabcabc"
    alphabet_size = len(set(text))
    # Target below alphabet+1: full alphabet is kept, no merges, UNK reserved.
    tok_small = BPETokenizer(vocab_size=2).fit(text)
    assert tok_small.vocab_size == alphabet_size + 1
    assert tok_small.merges == []
    assert tok_small.unk_id == tok_small.vocab_size - 1
    # Target exactly alphabet+1: still no merges.
    tok_exact = BPETokenizer(vocab_size=alphabet_size + 1).fit(text)
    assert tok_exact.merges == []
    assert tok_exact.vocab_size == alphabet_size + 1
    # Target larger than achievable: vocabulary stops at the saturated point.
    tok_max = BPETokenizer(vocab_size=10_000).fit("aaaa")
    assert tok_max.vocab_size < 10_000
    assert tok_max.vocab_size <= 3  # "aaaa" saturates after (a,a); + UNK
    assert tok_max.unk_id == tok_max.vocab_size - 1
    # In-between target is respected exactly when merges are available:
    # vocab = alphabet(3) + merges + UNK(1) == 5 -> 1 merge.
    tok_mid = BPETokenizer(vocab_size=5).fit(_repetitive_corpus())
    assert tok_mid.vocab_size == 5
    assert len(tok_mid.merges) == 5 - tok_mid.alphabet_size - 1
    # Invariant for any fitted tokenizer.
    for tok in (tok_small, tok_exact, tok_max, tok_mid):
        assert tok.vocab_size == tok.alphabet_size + tok.n_merges + 1
        assert tok.unk_id == tok.vocab_size - 1


def test_merge_reduces_sequence_length():
    tok = BPETokenizer(vocab_size=6).fit(_repetitive_corpus())
    ids = tok.encode(_repetitive_corpus())
    # 30 characters -> "abc" collapses to one token each.
    assert len(ids) == 10
    assert all(tok.token_char_length(i) == 3 for i in ids)


def test_unseen_characters_map_to_unk():
    tok = BPETokenizer(vocab_size=16).fit("abc" * 10)
    unk = tok.unk_id
    # A single unseen character encodes to exactly one UNK token.
    assert tok.encode("d") == [unk]
    assert tok.decode([unk]) == UNK
    # Consecutive unseen characters: one UNK each; UNK is never merged.
    assert tok.encode("dd") == [unk, unk]
    assert tok.encode("xyz") == [unk, unk, unk]
    # UNK tokens survive the merge passes untouched.
    assert tok.encode("adbd") == [tok.encode("a")[0], unk, tok.encode("b")[0], unk]


def test_mixed_known_and_unknown_input():
    tok = BPETokenizer(vocab_size=16).fit("abc" * 10)
    unk = tok.unk_id
    text = "ab!cé"  # '!' and 'é' are outside the training alphabet {a, b, c}
    ids = tok.encode(text)
    # Nothing is dropped: both unknown characters yield UNK tokens.
    assert unk in ids
    assert ids.count(unk) == 2
    # Character coverage is exact: token char lengths sum to the input length.
    assert sum(tok.token_char_length(i) for i in ids) == len(text)
    # Decoding is lossy exactly for the unknown characters -> U+FFFD.
    decoded = tok.decode(ids)
    assert decoded.count(UNK) == text.count("!") + text.count("é")
    assert decoded.replace(UNK, "") == "abc"


def test_encode_unknown_raise_option():
    tok = BPETokenizer(vocab_size=16).fit("abc" * 10)
    with pytest.raises(UnknownCharacterError):
        tok.encode("abcd", unknown="raise")
    with pytest.raises(UnknownCharacterError):
        tok.encode("🙂", unknown="raise")
    # Default is replace (UNK), never silent dropping.
    assert tok.encode("abcd") == tok.encode("abcd", unknown="replace")
    with pytest.raises(ValueError):
        tok.encode("abc", unknown="ignore")
    with pytest.raises(ValueError):
        tok.encode("abc", unknown="bogus")


def test_invalid_configuration():
    for bad in [0, 1, -3]:
        with pytest.raises(ValueError):
            BPETokenizer(vocab_size=bad)
    for bad in [2.5, "8", None, True]:
        with pytest.raises(TypeError):
            BPETokenizer(vocab_size=bad)
    tok = BPETokenizer(vocab_size=8).fit("abc" * 5)
    with pytest.raises(TypeError):
        tok.fit(b"not a string")
    with pytest.raises(RuntimeError):
        BPETokenizer(vocab_size=8).encode("a")
    with pytest.raises(ValueError):
        tok.token_char_length(tok.vocab_size)
    with pytest.raises(ValueError):
        tok.token_char_length(-1)


def test_encode_before_fit_raises():
    tok = BPETokenizer(vocab_size=8)
    for call in [
        lambda: tok.encode("a"),
        lambda: tok.decode([0]),
        lambda: tok.to_dict(),
        lambda: tok.unk_id,
        lambda: tok.token_char_length(0),
    ]:
        with pytest.raises(RuntimeError):
            call()


def test_decode_invalid_ids():
    tok = BPETokenizer(vocab_size=8).fit("abc" * 5)
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])
    with pytest.raises(ValueError):
        tok.decode([-1])
    with pytest.raises(ValueError):
        tok.decode([1.5])


def test_decode_accepts_tensor_like_input():
    class FakeTensor:
        def tolist(self):
            return [0, 1, 2]

    tok = BPETokenizer(vocab_size=8).fit("abc" * 5)
    assert tok.decode(FakeTensor()) == "abc"


def test_serialization_roundtrip():
    text = _text_corpus()
    tok = BPETokenizer(vocab_size=64).fit(text)
    data = tok.to_dict()
    clone = BPETokenizer.from_dict(data)
    assert clone.alphabet == tok.alphabet
    assert clone.merges == tok.merges
    assert clone.vocab_size == tok.vocab_size
    assert clone.unk_id == tok.unk_id
    for sample in [text, "Hello again", "", "zzz" * 3]:
        assert clone.encode(sample) == tok.encode(sample)
        assert clone.decode(clone.encode(sample)) == tok.decode(tok.encode(sample))
    # Serialization stays JSON-friendly.
    import json

    assert BPETokenizer.from_dict(json.loads(json.dumps(data))).encode(text) == tok.encode(text)
    with pytest.raises(ValueError):
        BPETokenizer.from_dict({"type": "gpt2", "alphabet": [], "merges": []})


def test_char_length_covers_arbitrary_text():
    # Known + unknown characters: coverage invariant holds for any text.
    tok = BPETokenizer(vocab_size=20).fit("abc" * 20)
    for text in ["abcabc", "a!b?", "🙂" * 5, "a b\tc\n", "zzz"]:
        ids = tok.encode(text)
        assert sum(tok.token_char_length(i) for i in ids) == len(text)
