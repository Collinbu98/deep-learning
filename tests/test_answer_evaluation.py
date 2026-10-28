import json
from pathlib import Path

import pytest

from minigrad.answer_evaluation import (
    AnswerKeyEntry,
    AnswerObservation,
    assess_answer,
    load_answer_key,
    observation_from_extractive,
    summarise_assessments,
)
from minigrad.chunking import Chunk
from minigrad.llm_answerer import EvidencePassage
from minigrad.rag import Answer, ExtractedSentence, RetrievedPassage


def evidence(source_id="jupiter#0", doc_id="jupiter", text="Jupiter has 95 known moons."):
    return EvidencePassage(
        source_id=source_id,
        doc_id=doc_id,
        chunk_index=0,
        source=f"{doc_id}.txt",
        text=text,
        score=1.0,
        rank=1,
    )


def observation(text, citations=(), passages=None, abstained=False):
    return AnswerObservation(
        question="How many moons does Jupiter have?",
        text=text,
        abstained=abstained,
        citations=tuple(citations),
        evidence=tuple(passages if passages is not None else (evidence(),)),
    )


ENTRY = AnswerKeyEntry(
    id="q1",
    required_fact_groups=(("95",),),
    required_doc_ids=("jupiter",),
)


def test_correct_answer_requires_fact_valid_citation_and_support():
    row = assess_answer(
        ENTRY,
        observation("Jupiter has 95 known moons [jupiter#0].", ("jupiter#0",)),
        answerable=True,
    )
    assert row["factual_correct"] is True
    assert row["citations_exist"] is True
    assert row["citations_support_claims"] is True
    assert row["required_docs_cited"] is True
    assert row["correct"] is True


def test_valid_citation_alone_is_not_enough_when_source_does_not_support_claim():
    passages = (evidence(text="Jupiter is a large planet."),)
    row = assess_answer(
        ENTRY,
        observation(
            "Jupiter has 95 known moons [jupiter#0].",
            ("jupiter#0",),
            passages,
        ),
        answerable=True,
    )
    assert row["factual_correct"] is True
    assert row["citations_exist"] is True
    assert row["citations_support_claims"] is False
    assert row["correct"] is False
    assert "does not support" in row["reason"]


def test_missing_and_invented_citations_fail():
    missing = assess_answer(
        ENTRY,
        observation("Jupiter has 95 known moons."),
        answerable=True,
    )
    assert missing["factual_correct"] is True
    assert missing["citation_present"] is False
    assert missing["correct"] is False
    assert "citation missing" in missing["reason"]

    invented = assess_answer(
        ENTRY,
        observation("Jupiter has 95 known moons [fake#99].", ("fake#99",)),
        answerable=True,
    )
    assert invented["citations_exist"] is False
    assert invented["correct"] is False


def test_missing_required_document_fails_multi_document_check():
    passages = (
        evidence("jupiter#0", "jupiter", "Jupiter has 95 moons."),
        evidence("saturn#0", "saturn", "Saturn has 146 moons."),
    )
    entry = AnswerKeyEntry(
        id="multi",
        required_fact_groups=(("95",), ("146",)),
        required_doc_ids=("jupiter", "saturn"),
    )
    partial = assess_answer(
        entry,
        observation(
            "Jupiter has 95 moons and Saturn has 146 moons [jupiter#0].",
            ("jupiter#0",),
            passages,
        ),
        answerable=True,
    )
    assert partial["required_docs_cited"] is False
    assert partial["correct"] is False

    complete = assess_answer(
        entry,
        observation(
            "Jupiter has 95 moons and Saturn has 146 moons "
            "[jupiter#0][saturn#0].",
            ("jupiter#0", "saturn#0"),
            passages,
        ),
        answerable=True,
    )
    assert complete["required_docs_cited"] is True
    assert complete["correct"] is True


def test_unanswerable_scores_abstention_only():
    entry = AnswerKeyEntry(id="q12")
    good = assess_answer(
        entry,
        observation("", abstained=True),
        answerable=False,
    )
    assert good["correct"] is True

    bad = assess_answer(
        entry,
        observation("Pluto's gravity is 0.62 [earth#0].", ("earth#0",)),
        answerable=False,
    )
    assert bad["correct"] is False
    assert "unanswerable" in bad["reason"]


def test_false_abstention_on_answerable_question_is_incorrect():
    row = assess_answer(
        ENTRY,
        observation("", abstained=True),
        answerable=True,
    )
    assert row["correct"] is False
    assert "answerable" in row["reason"]


def test_summary_excludes_manual_review_entries():
    automatic = assess_answer(
        ENTRY,
        observation("Jupiter has 95 known moons [jupiter#0].", ("jupiter#0",)),
        answerable=True,
    )
    manual = dict(automatic)
    manual["id"] = "manual"
    manual["manual_review"] = True
    manual["correct"] = False
    unanswerable = assess_answer(
        AnswerKeyEntry(id="q12"),
        observation("", abstained=True),
        answerable=False,
    )
    summary = summarise_assessments([automatic, manual, unanswerable])
    assert summary["n_total"] == 3
    assert summary["n_objective"] == 2
    assert summary["n_manual_review"] == 1
    assert summary["accuracy"] == pytest.approx(1.0)
    assert summary["abstention_rate_unanswerable"] == pytest.approx(1.0)


def test_load_answer_key_and_observation_from_extractive():
    key = load_answer_key(
        Path(__file__).resolve().parents[1] / "data" / "rag_corpus" / "answer_key.json"
    )
    assert key["q01"].required_doc_ids == ("jupiter",)
    assert key["q08"].manual_review is True
    assert "q12" in key

    chunk = Chunk(
        doc_id="jupiter",
        source="jupiter.txt",
        chunk_index=0,
        text="Jupiter has 95 moons.",
        start=0,
        end=21,
    )
    passage = RetrievedPassage(chunk=chunk, score=1.0, rank=1)
    answer = Answer(
        question="How many moons?",
        text="Jupiter has 95 moons.",
        sentences=(
            ExtractedSentence(
                text="Jupiter has 95 moons.",
                doc_id="jupiter",
                source="jupiter.txt",
                chunk_index=0,
                score=1.0,
            ),
        ),
        passages=(passage,),
    )
    converted = observation_from_extractive(answer)
    assert converted.citations == ("jupiter#0",)
    assert converted.evidence[0].source_id == "jupiter#0"


def test_load_answer_key_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "key.json"
    path.write_text(
        json.dumps(
            {
                "answers": [
                    {"id": "q1"},
                    {"id": "q1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_answer_key(path)
