"""Deterministic evidence-window construction for claim verification.

The module keeps sentence splitting and window traceability independent of any
NLI model. A window always points back to the source passage, sentence range,
and character range used to build it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from minigrad.llm_answerer import EvidencePassage

__all__ = [
    "WHOLE_PASSAGE",
    "SENTENCE",
    "SENTENCE_ADJACENT",
    "WINDOW_STRATEGIES",
    "SentenceSpan",
    "EvidenceWindow",
    "split_sentences",
    "build_evidence_windows",
    "build_windows_for_passages",
]

WHOLE_PASSAGE = "whole_passage"
SENTENCE = "sentence"
SENTENCE_ADJACENT = "sentence_adjacent"
WINDOW_STRATEGIES = (WHOLE_PASSAGE, SENTENCE, SENTENCE_ADJACENT)

_ABBREVIATIONS = frozenset(
    {
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "jr",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "u.s",
        "vs",
    }
)
_CLOSING = set("\"'”’»)]}")
_TERMINATORS = ".!?"


@dataclass(frozen=True)
class SentenceSpan:
    """A non-empty sentence and its offsets in the original passage."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class EvidenceWindow:
    """A traceable slice of a cited passage."""

    window_id: str
    source_id: str
    doc_id: str
    chunk_index: int
    strategy: str
    start_sentence: int
    end_sentence: int
    char_start: int
    char_end: int
    text: str
    sentence_count: int


def _is_abbreviation(prefix: str) -> bool:
    match = re.search(r"([A-Za-z][A-Za-z.]*)$", prefix)
    if match is None:
        return False
    word = match.group(1).lower()
    if word.rstrip(".") in _ABBREVIATIONS:
        return True
    if re.fullmatch(r"(?:[a-z]\.){2,}", word):
        return True
    return bool(re.fullmatch(r"[a-z]\.", word))


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    if not text or not text.strip():
        return []

    spans: list[tuple[int, int]] = []
    length = len(text)
    start = 0
    index = 0
    while index < length:
        character = text[index]
        if character not in _TERMINATORS:
            index += 1
            continue

        end_punctuation = index
        while (
            end_punctuation + 1 < length
            and text[end_punctuation + 1] in _TERMINATORS
        ):
            end_punctuation += 1

        boundary = end_punctuation + 1
        while boundary < length and text[boundary] in _CLOSING:
            boundary += 1

        if character == "." and index > 0 and index + 1 < length:
            previous = text[index - 1]
            following = text[index + 1]
            if previous.isdigit() and following.isdigit():
                index = end_punctuation + 1
                continue
            if _is_abbreviation(text[start:boundary]):
                index = end_punctuation + 1
                continue

        if boundary == length:
            spans.append((start, length))
            return spans

        if not text[boundary].isspace():
            index = end_punctuation + 1
            continue

        next_start = boundary
        while next_start < length and text[next_start].isspace():
            next_start += 1
        if next_start == length:
            spans.append((start, length))
            return spans

        spans.append((start, boundary))
        start = next_start
        index = next_start

    if start < length:
        spans.append((start, length))
    return spans


def split_sentences(text: str) -> tuple[SentenceSpan, ...]:
    """Split text into sentence spans without third-party dependencies.

    Decimal points and common abbreviations are not treated as sentence ends.
    Punctuation without following whitespace is preserved. Empty or
    boundary-free input is handled without raising.
    """

    sentences = []
    for start, end in _sentence_spans(text):
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        sentences.append(
            SentenceSpan(
                text=stripped,
                start=start + leading,
                end=end - trailing,
            )
        )
    return tuple(sentences)


def _ranges_for_strategy(
    sentence_count: int, strategy: str, adjacent_context: int
) -> list[tuple[int, int]]:
    if strategy == WHOLE_PASSAGE:
        return [(0, sentence_count - 1)]
    if strategy == SENTENCE:
        return [(index, index) for index in range(sentence_count)]
    if strategy == SENTENCE_ADJACENT:
        ranges = []
        seen = set()
        for index in range(sentence_count):
            start = max(0, index - adjacent_context)
            end = min(sentence_count - 1, index + adjacent_context)
            if (start, end) not in seen:
                seen.add((start, end))
                ranges.append((start, end))
        return ranges
    raise ValueError(f"unsupported evidence-window strategy {strategy!r}")


def build_evidence_windows(
    passage: EvidencePassage,
    strategy: str = SENTENCE,
    *,
    adjacent_context: int = 1,
    source_id: str | None = None,
) -> tuple[EvidenceWindow, ...]:
    """Build deterministic windows for one passage."""

    if strategy not in WINDOW_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {WINDOW_STRATEGIES}; got {strategy!r}"
        )
    if adjacent_context < 0:
        raise ValueError("adjacent_context must be non-negative")

    sentences = split_sentences(passage.text)
    if not sentences:
        return ()

    traceable_id = passage.source_id if source_id is None else source_id
    windows = []
    for start_sentence, end_sentence in _ranges_for_strategy(
        len(sentences), strategy, adjacent_context
    ):
        char_start = sentences[start_sentence].start
        char_end = sentences[end_sentence].end
        text = passage.text[char_start:char_end].strip()
        if not text:
            continue
        if strategy == WHOLE_PASSAGE:
            suffix = "whole"
        elif strategy == SENTENCE:
            suffix = f"sentence:{start_sentence}"
        else:
            suffix = f"adjacent:{start_sentence}-{end_sentence}"
        windows.append(
            EvidenceWindow(
                window_id=f"{traceable_id}::{suffix}",
                source_id=traceable_id,
                doc_id=passage.doc_id,
                chunk_index=passage.chunk_index,
                strategy=strategy,
                start_sentence=start_sentence,
                end_sentence=end_sentence,
                char_start=char_start,
                char_end=char_end,
                text=text,
                sentence_count=end_sentence - start_sentence + 1,
            )
        )
    return tuple(windows)


def build_windows_for_passages(
    passages: Mapping[str, EvidencePassage],
    strategy: str = SENTENCE,
    *,
    adjacent_context: int = 1,
) -> tuple[EvidenceWindow, ...]:
    """Build windows for passages in mapping order, deduplicating adjacency."""

    windows = []
    seen_ids = set()
    for source_id, passage in passages.items():
        for window in build_evidence_windows(
            passage,
            strategy,
            adjacent_context=adjacent_context,
            source_id=source_id,
        ):
            if window.window_id in seen_ids:
                continue
            seen_ids.add(window.window_id)
            windows.append(window)
    return tuple(windows)
