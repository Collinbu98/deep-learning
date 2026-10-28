"""Answer-level evaluation for extractive and LLM-generated RAG answers.

Retrieval metrics answer a different question ("was a relevant document in
the top k?") from answer metrics ("is the generated claim correct and
grounded?"). This module keeps the second question separate and deliberately
small.

Each answer-key entry lists required fact groups and, optionally, required
source documents. A fact group contains alternative surface forms, so
``[["146", "one hundred forty six"]]`` is one fact with two acceptable
spellings. Matching is normalised substring matching with token boundaries;
it is a transparent heuristic, not natural-language inference.

An answer is counted as correct only when:

* it is not abstained;
* every required fact group appears in the answer;
* every citation exists in the supplied evidence;
* every required fact is also present in at least one *cited* passage;
* all required source documents are cited.

A valid citation alone is never enough. The source must support the claim
according to the same surface-form check. Entries marked ``manual_review``
are reported separately and excluded from automatic accuracy aggregates.
Manual review is required when surface matching cannot reliably distinguish
the intended relation, such as a comparison whose two entity names could be
swapped while preserving all keywords.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from minigrad.llm_answerer import make_evidence

__all__ = [
    "AnswerKeyEntry",
    "AnswerObservation",
    "load_answer_key",
    "observation_from_llm",
    "observation_from_extractive",
    "assess_answer",
    "summarise_assessments",
]


@dataclass(frozen=True)
class AnswerKeyEntry:
    id: str
    required_fact_groups: tuple = ()
    required_doc_ids: tuple = ()
    manual_review: bool = False
    notes: str = ""

    def __post_init__(self):
        if not self.id:
            raise ValueError("answer-key entry id must not be empty")
        for group in self.required_fact_groups:
            if not group or any(not alternative.strip() for alternative in group):
                raise ValueError(
                    f"answer-key entry {self.id!r} has an empty fact alternative"
                )


@dataclass(frozen=True)
class AnswerObservation:
    question: str
    text: str
    abstained: bool
    citations: tuple
    evidence: tuple


def load_answer_key(path):
    """Load and validate an answer key keyed by question id."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = {}
    for item in data["answers"]:
        entry = AnswerKeyEntry(
            id=item["id"],
            required_fact_groups=tuple(
                tuple(group) for group in item.get("required_fact_groups", ())
            ),
            required_doc_ids=tuple(item.get("required_doc_ids", ())),
            manual_review=bool(item.get("manual_review", False)),
            notes=item.get("notes", ""),
        )
        if entry.id in entries:
            raise ValueError(f"duplicate answer-key entry: {entry.id}")
        entries[entry.id] = entry
    return entries


def observation_from_llm(answer):
    return AnswerObservation(
        question=answer.question,
        text=answer.text,
        abstained=answer.abstained,
        citations=tuple(answer.citations),
        evidence=tuple(answer.evidence),
    )


def observation_from_extractive(answer):
    citations = []
    for sentence in answer.sentences:
        source_id = f"{sentence.doc_id}#{sentence.chunk_index}"
        if source_id not in citations:
            citations.append(source_id)
    return AnswerObservation(
        question=answer.question,
        text=answer.text,
        abstained=answer.abstained,
        citations=tuple(citations),
        evidence=make_evidence(answer.passages),
    )


def _normalise(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _has_alternative(normalised_text, group):
    for alternative in group:
        normalised = _normalise(alternative)
        if not normalised:
            continue
        pattern = (
            r"(?<![a-z0-9])" + re.escape(normalised) + r"(?![a-z0-9])"
        )
        if re.search(pattern, normalised_text):
            return True
    return False


def _citation_evidence_ids(observation, citations):
    supplied = {passage.source_id for passage in observation.evidence}
    return tuple(citation for citation in citations if citation in supplied)


def _cited_text(observation, citations):
    cited = set(_citation_evidence_ids(observation, citations))
    return " ".join(
        passage.text for passage in observation.evidence if passage.source_id in cited
    )


def assess_answer(entry, observation, answerable):
    """Return a transparent assessment for one answer.

    For unanswerable questions only abstention is scored. For answerable
    questions the assessment separates factual matching, citation existence,
    citation support, and required-document coverage.
    """
    evidence_ids = {passage.source_id for passage in observation.evidence}
    citations = tuple(observation.citations)
    citation_present = bool(citations)
    citations_exist = citation_present and all(
        citation in evidence_ids for citation in citations
    )

    if not answerable:
        correct = bool(observation.abstained)
        return {
            "id": entry.id,
            "answerable": False,
            "manual_review": entry.manual_review,
            "abstained": observation.abstained,
            "expected_abstain": True,
            "factual_correct": None,
            "fact_matches": (),
            "citation_present": citation_present,
            "citations_exist": citations_exist,
            "citations_support_claims": None,
            "required_docs_cited": None,
            "correct": correct,
            "text": observation.text,
            "citations": list(citations),
            "evidence_ids": sorted(evidence_ids),
            "reason": "abstained as expected" if correct else "answered an unanswerable question",
        }

    normalised_answer = _normalise(observation.text)
    fact_matches = tuple(
        _has_alternative(normalised_answer, group)
        for group in entry.required_fact_groups
    )
    factual_correct = bool(fact_matches) and all(fact_matches)
    if not entry.required_fact_groups:
        factual_correct = None

    support_text = _normalise(_cited_text(observation, citations))
    support_matches = tuple(
        _has_alternative(support_text, group)
        for group in entry.required_fact_groups
    )
    citations_support_claims = (
        bool(entry.required_fact_groups)
        and all(support_matches)
        and citations_exist
    )

    cited_docs = {
        passage.doc_id
        for passage in observation.evidence
        if passage.source_id in set(citations)
    }
    required_docs_cited = set(entry.required_doc_ids).issubset(cited_docs)

    correct = (
        not observation.abstained
        and bool(factual_correct)
        and citations_exist
        and citations_support_claims
        and required_docs_cited
    )
    return {
        "id": entry.id,
        "answerable": True,
        "manual_review": entry.manual_review,
        "abstained": observation.abstained,
        "expected_abstain": False,
        "factual_correct": factual_correct,
        "fact_matches": fact_matches,
        "citation_present": citation_present,
        "citations_exist": citations_exist,
        "citations_support_claims": citations_support_claims,
        "required_docs_cited": required_docs_cited,
        "correct": correct,
        "text": observation.text,
        "citations": list(citations),
        "evidence_ids": sorted(evidence_ids),
        "reason": "" if correct else _failure_reason(
            observation,
            factual_correct,
            citations_exist,
            citations_support_claims,
            required_docs_cited,
        ),
    }


def _failure_reason(
    observation,
    factual_correct,
    citations_exist,
    citations_support_claims,
    required_docs_cited,
):
    if observation.abstained:
        return "abstained on an answerable question"
    if factual_correct is False:
        return "required fact missing from answer"
    if not citations_exist:
        return "citation missing or not present in supplied evidence"
    if not citations_support_claims:
        return "cited evidence does not support a required fact"
    if not required_docs_cited:
        return "required source document was not cited"
    return ""


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def summarise_assessments(rows):
    """Aggregate automatic checks, excluding entries marked manual review."""
    objective = [row for row in rows if not row["manual_review"]]
    answerable = [row for row in objective if row["answerable"]]
    unanswerable = [row for row in objective if not row["answerable"]]

    def rate(rows_for_rate, key):
        return _mean(row[key] for row in rows_for_rate if row[key] is not None)

    return {
        "n_total": len(rows),
        "n_objective": len(objective),
        "n_manual_review": len(rows) - len(objective),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "accuracy": rate(answerable + unanswerable, "correct"),
        "factual_accuracy": rate(answerable, "factual_correct"),
        "citation_presence_rate": rate(answerable, "citation_present"),
        "citation_validity_rate": rate(answerable, "citations_exist"),
        "citation_support_rate": rate(answerable, "citations_support_claims"),
        "required_docs_cited_rate": rate(answerable, "required_docs_cited"),
        "abstention_rate_unanswerable": rate(unanswerable, "abstained"),
        "false_abstention_rate": rate(answerable, "abstained"),
    }
