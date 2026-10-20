"""Educational character-level Byte-Pair Encoding (BPE) tokenizer.

This is a small, deterministic BPE implementation intended for learning. It
follows the classic BPE algorithm described by Sennrich et al. (2016), with
one educational simplification: it starts from *characters*, not bytes.

Production BPE tokenizers (GPT-2, RoBERTa, ...) are *byte-level*: they start
from the 256 byte values and use a pre-tokenization step (e.g. splitting on
words) before learning merges. This implementation does neither. It operates
directly on the raw character sequence of the training corpus. The trade-off
is simplicity: the algorithm is easy to read and test, but it cannot encode
characters that were absent from the training text exactly (they become UNK
tokens), and it is O(n * num_merges) when encoding, which is fine for
small/medium corpora but not for production use.

Vocabulary layout
-----------------
* ids ``0 .. alphabet_size-1``: the characters of the training alphabet, in
  sorted order;
* ids ``alphabet_size .. alphabet_size + n_merges - 1``: learned merged
  tokens, in merge-rank order;
* the **last id** (``alphabet_size + n_merges``): the special UNK token
  (``unk_id``), reserved for characters outside the training alphabet. UNK
  is included in ``vocab_size``. ``vocab_size`` therefore always equals
  ``alphabet_size + n_merges + 1`` and is never smaller than
  ``alphabet_size + 1``.

Training procedure
------------------
1. The *starting representation* of the corpus is one token per character.
   The initial alphabet is ``sorted(set(training_text))``; character ids are
   assigned in that sorted order starting at 0. This is fully deterministic.
2. A *merge* combines the most frequent adjacent pair of tokens into a new
   token. Ties between equally frequent pairs are broken deterministically by
   choosing the lexicographically smallest ``(left_id, right_id)`` pair.
   All non-overlapping occurrences of the chosen pair are replaced in a
   single left-to-right pass (overlapping occurrences such as ``a a a`` with
   merge ``(a, a)`` resolve left-to-right, matching standard practice).
3. Each new merged token receives the next free id (``alphabet_size``,
   ``alphabet_size + 1``, ...), and the pair is appended to the merge list.
   The position of a merge in this list is its *rank*: rank 0 is the first
   learned merge.
4. Steps 2-3 repeat until the vocabulary (alphabet + merges + UNK) reaches
   the target size or no pair occurs more than once.

Encoding
--------
``encode`` first maps characters to their ids (unknown characters become
``unk_id`` unless ``unknown="raise"``), then applies the learned merges *in
rank order*, one left-to-right pass per merge. This is equivalent to the
canonical "repeatedly merge the lowest-ranked pair present in the sequence"
formulation: merging pair of rank r can only create new adjacent pairs that
involve a token created at step r, and such pairs necessarily have rank > r,
so no merge with rank <= r becomes newly applicable after its own pass.

Unknown characters
------------------
The alphabet is learned from the training text only. By default, encoding
text that contains a character outside that alphabet maps each such
character to the UNK token (``unknown="replace"``). Consecutive unknown
characters produce one UNK token each; UNK is never merged with anything, so
every input character is represented by at least one token and the encoded
sequence always covers the full input. ``decode`` renders UNK as the Unicode
replacement character ``"\\ufffd"`` (U+FFFD), so a round trip through
encode/decode is *lossy* exactly for the unknown characters: the original
bytes cannot be recovered. ``encode(..., unknown="raise")`` raises
``UnknownCharacterError`` instead, for callers that want strict behavior.

Empty input encodes to ``[]`` and decodes back to ``""``.
"""

from __future__ import annotations

from collections import Counter

__all__ = ["BPETokenizer", "UnknownCharacterError"]


class UnknownCharacterError(ValueError):
    """Raised when encoding text containing characters outside the alphabet."""


class BPETokenizer:
    """Character-level BPE tokenizer with a reserved UNK token.

    Args:
        vocab_size: Target vocabulary size, *including* the alphabet, the
            merged tokens, and UNK. The vocabulary always contains the full
            character alphabet plus UNK, so the *actual* vocabulary is
            ``max(vocab_size, alphabet_size + 1)``; it can be smaller than
            requested if the corpus runs out of repeated pairs. Must be an
            integer >= 2.

    Public special-token API (safe to use from experiments/tools):

    * ``unk_id``: id of the UNK token (always ``vocab_size - 1``).
    * ``unk_token``: the string ``"\\ufffd"`` used when decoding UNK.
    * ``token_char_length(token_id)``: number of raw characters a token
        represents (1 for character and UNK tokens, the concatenated length
        for merged tokens). Sums to the input length for any encoded text.

    Usage::

        tok = BPETokenizer(vocab_size=512).fit(training_text)
        ids = tok.encode("some text")          # unknown chars -> unk_id
        text = tok.decode(ids)
    """

    unk_token = "�"

    def __init__(self, vocab_size=512):
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError(f"vocab_size must be an int, got {type(vocab_size).__name__}")
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        self.vocab_size_target = vocab_size
        # Fitted state; populated by fit().
        self.alphabet = None  # sorted list of characters, index == char id
        self.merges = None  # list of (left_id, right_id) in rank order
        self._char_to_id = None
        self._itos = None  # id -> token string
        self._pair_to_rank = None
        self._vocab_size = None
        self.n_corpus_chars = None  # characters seen during fit

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self):
        return self.alphabet is not None

    @property
    def alphabet_size(self):
        self._check_fitted()
        return len(self.alphabet)

    @property
    def n_merges(self):
        self._check_fitted()
        return len(self.merges)

    @property
    def unk_id(self):
        """Id of the UNK token; always the last id in the vocabulary."""
        self._check_fitted()
        return self._vocab_size - 1

    @property
    def vocab_size(self):
        """Actual vocabulary size: alphabet plus learned merges plus UNK."""
        self._check_fitted()
        return self._vocab_size

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("BPETokenizer is not fitted; call fit(corpus) first")

    def token_char_length(self, token_id):
        """Number of raw characters represented by ``token_id``.

        Character tokens and the UNK token each represent exactly one input
        character; a merged token represents the sum of its parts. This is
        the weight used to normalize token-level losses to a per-character
        basis.
        """
        self._check_fitted()
        i = int(token_id)
        if i != token_id or i < 0 or i >= self._vocab_size:
            raise ValueError(
                f"invalid token id {token_id!r}; valid range is [0, {self._vocab_size})"
            )
        return len(self._itos[i])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, text):
        """Learn the alphabet and merge rules from ``text``.

        Deterministic: identical ``text`` and ``vocab_size`` always produce
        identical alphabets, merges, and vocabularies.

        Raises:
            TypeError: ``text`` is not a string.
            ValueError: ``text`` is empty.
        """
        if not isinstance(text, str):
            raise TypeError(f"training text must be a str, got {type(text).__name__}")
        if not text:
            raise ValueError("cannot fit BPE tokenizer on empty text")

        alphabet = sorted(set(text))
        char_to_id = {ch: i for i, ch in enumerate(alphabet)}
        ids = [char_to_id[ch] for ch in text]

        merges = []
        n_alphabet = len(alphabet)
        # Budget: alphabet + merges + the reserved UNK slot must fit the target.
        while n_alphabet + len(merges) + 1 < self.vocab_size_target:
            if len(ids) < 2:
                break
            counts = Counter(zip(ids, ids[1:]))
            # Highest count first; ties broken by smallest (left_id, right_id).
            (left, right), top_count = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if top_count < 2:
                break  # no repeated pair left to merge
            new_id = n_alphabet + len(merges)
            merged = []
            i = 0
            n = len(ids)
            while i < n:
                if i + 1 < n and ids[i] == left and ids[i + 1] == right:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
            merges.append((left, right))

        self.alphabet = alphabet
        self._char_to_id = char_to_id
        self.merges = merges
        self._pair_to_rank = {(a, b): r for r, (a, b) in enumerate(merges)}
        self._vocab_size = n_alphabet + len(merges) + 1  # +1 reserves UNK
        self._itos = {i: ch for i, ch in enumerate(alphabet)}
        for rank, (a, b) in enumerate(merges):
            self._itos[n_alphabet + rank] = self._itos[a] + self._itos[b]
        self._itos[self._vocab_size - 1] = self.unk_token
        self.n_corpus_chars = len(text)
        return self

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text, *, unknown="replace"):
        """Encode ``text`` to a list of token ids.

        Every input character is represented by at least one token; the
        encoded sequence never silently drops characters.

        Args:
            text: String to encode. Empty string encodes to ``[]``.
            unknown: ``"replace"`` (default) maps characters outside the
                learned alphabet to ``unk_id``; ``"raise"`` raises
                ``UnknownCharacterError``.
        """
        self._check_fitted()
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if unknown not in ("replace", "raise"):
            raise ValueError(f'unknown must be "replace" or "raise", got {unknown!r}')
        if not text:
            return []

        unk = self._vocab_size - 1
        ids = []
        bad = []
        lookup = self._char_to_id
        for ch in text:
            tid = lookup.get(ch)
            if tid is None:
                if unknown == "replace":
                    ids.append(unk)
                else:
                    bad.append(ch)
            else:
                ids.append(tid)
        if bad:
            uniq = sorted(set(bad))
            raise UnknownCharacterError(
                f"{len(bad)} character(s) not in the learned alphabet: {uniq!r}"
            )

        # Apply merges in rank order, one full pass per merge. UNK tokens are
        # never an argument of any merge (merges come from training text,
        # which only contains alphabet characters), so they pass through.
        for rank, (left, right) in enumerate(self.merges):
            new_id = len(self.alphabet) + rank
            merged = []
            i = 0
            n = len(ids)
            while i < n:
                if i + 1 < n and ids[i] == left and ids[i + 1] == right:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        return ids

    def decode(self, ids):
        """Decode a sequence of token ids back to a string.

        Accepts any iterable of ints (lists, tuples, torch tensors, ...).
        Empty input decodes to ``""``. UNK tokens decode to ``unk_token``
        (U+FFFD), so decoding is lossy for text that contained characters
        outside the training alphabet.
        """
        self._check_fitted()
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        parts = []
        for raw in ids:
            i = int(raw)
            if i != raw or i < 0 or i >= self._vocab_size:
                raise ValueError(
                    f"invalid token id {raw!r}; valid range is [0, {self._vocab_size})"
                )
            parts.append(self._itos[i])
        return "".join(parts)

    # ------------------------------------------------------------------
    # Serialization (for checkpointing alongside a model)
    # ------------------------------------------------------------------

    def to_dict(self):
        """Return a JSON-friendly description of the fitted tokenizer."""
        self._check_fitted()
        return {
            "type": "bpe_char",
            "version": 2,
            "unk_token": self.unk_token,
            "vocab_size_target": self.vocab_size_target,
            "alphabet": list(self.alphabet),
            "merges": [list(pair) for pair in self.merges],
            "n_corpus_chars": self.n_corpus_chars,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a fitted tokenizer from ``to_dict()`` output.

        The rebuilt tokenizer reserves UNK as the last id exactly as ``fit``
        does, so vocabularies round-trip through serialization unchanged.
        """
        if not isinstance(data, dict):
            raise TypeError(f"expected a dict, got {type(data).__name__}")
        if data.get("type") != "bpe_char":
            raise ValueError(f"not a bpe_char tokenizer dict: type={data.get('type')!r}")
        alphabet = data["alphabet"]
        merges = [tuple(pair) for pair in data["merges"]]
        unk_token = data.get("unk_token", cls.unk_token)
        if unk_token != cls.unk_token:
            raise ValueError(
                f"unsupported unk_token {unk_token!r}; expected {cls.unk_token!r}"
            )
        tok = cls(vocab_size=max(2, len(alphabet) + len(merges) + 1))
        tok.alphabet = list(alphabet)
        tok._char_to_id = {ch: i for i, ch in enumerate(tok.alphabet)}
        tok.merges = merges
        tok._pair_to_rank = {(a, b): r for r, (a, b) in enumerate(merges)}
        tok._vocab_size = len(tok.alphabet) + len(merges) + 1  # +1 reserves UNK
        tok._itos = {i: ch for i, ch in enumerate(tok.alphabet)}
        for rank, (a, b) in enumerate(merges):
            tok._itos[len(tok.alphabet) + rank] = tok._itos[a] + tok._itos[b]
        tok._itos[tok._vocab_size - 1] = tok.unk_token
        tok.n_corpus_chars = data.get("n_corpus_chars")
        return tok
