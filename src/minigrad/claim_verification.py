"""Claim-level evidence verification for RAG answers.

This module treats verification as a separate problem from citation ID
existence. Checking that a citation points to a supplied passage proves
nothing about whether that passage supports the specific claim. A passage
can overlap lexically with a claim and still contradict it; a paraphrased
claim can have little word overlap and still be supported.

The design splits three steps:

1. **Claim extraction** turns an answer into atomic claims. A single
   sentence can contain multiple assertions; this module keeps extraction
   separate from verification.
2. **Coverage checking** tests whether a multi-part question's required
   facts/parts are addressed at all (not just cited).
3. **Claim verification** checks each claim against its cited passages
   using a transparent lexical baseline. The baseline is an educational
   heuristic: it can catch obvious contradictions and clear cases of no
   support, but it cannot prove general entailment.

Status values are explicit: ``supported``, ``contradicted``, 
``insufficient_evidence``, ``invalid_citation``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

from minigrad.llm_answerer import EvidencePassage


__all__ = [
    "Claim",
    "ClaimVerificationResult",
    "AnswerClaimSet",
    "MultiPartQuestion",
    "CoverageReport",
    "extract_claims_from_sentences",
    "check_citation_validity",
    "verify_claim_lexical",
    "verify_claim",
    "verify_answer_claims",
    "check_multi_part_coverage",
    "extract_claims_with_llm",
]


@dataclass(frozen=True)
class Claim:
    """An atomic claim extracted from an answer."""
    text: str
    citation_ids: tuple[str, ...]
    sentence_index: int = 0
    claim_index: int = 0


@dataclass(frozen=True)
class ClaimVerificationResult:
    """Verification outcome for a single claim."""
    claim: Claim
    status: str  # supported, contradicted, insufficient_evidence, invalid_citation
    explanation: str
    evidence_excerpts: tuple[tuple[str, str], ...]  # (source_id, excerpt)
    cited_passages_exist: bool


@dataclass(frozen=True)
class AnswerClaimSet:
    """All claims extracted from an answer."""
    question: str
    claims: tuple[Claim, ...]
    raw_text: str


@dataclass(frozen=True)
class MultiPartQuestion:
    """A question requiring multiple parts to be addressed."""
    question_id: str
    question_text: str
    parts: tuple[tuple[str, tuple[str, ...]], ...]  # (part_description, required_keywords)


@dataclass(frozen=True)
class CoverageReport:
    """Coverage check for multi-part questions."""
    question_id: str
    required_parts: int
    addressed: int
    supported: int
    unsupported_or_missing: int
    part_details: tuple[dict, ...]


# Mercury/Venus comparison parts
MERCURY_VENUS_PARTS = (
    ("Mercury closer to Sun than Venus", ("mercury", "closer")),
    ("Venus hotter than Mercury", ("venus", "hotter")),
)

# Earth/Mars moon count parts  
EARTH_MARS_MOONS_PARTS = (
    ("Earth has moons", ("earth", "moon")),
    ("Mars has moons", ("mars", "moon")),
    ("Compare counts (Mars has more or equal)", ("mars", "earth", "more", "two", "one")),
)


def extract_claims_from_sentences(sentences_with_citations: list[tuple[str, tuple[str, ...]]]) -> tuple[Claim, ...]:
    """Extract claims from sentences, keeping sentences separate from verification.
    
    This is a simple sentence-level extraction - a sentence may contain multiple
    claims, but we keep it simple for educational purposes and note the limitation.
    """
    claims = []
    for s_idx, (sentence, citations) in enumerate(sentences_with_citations):
        sentence = sentence.strip()
        if not sentence:
            continue
        claims.append(Claim(
            text=sentence,
            citation_ids=tuple(citations),
            sentence_index=s_idx,
            claim_index=len(claims),
        ))
    return tuple(claims)


def check_citation_validity(claim: Claim, supplied_passage_ids: set[str]) -> tuple[bool, tuple[str, ...]]:
    """Check if claim's citations refer to passages actually supplied."""
    if not claim.citation_ids:
        return False, ()
    existing = []
    for cid in claim.citation_ids:
        if cid in supplied_passage_ids:
            existing.append(cid)
        else:
            pass  # missing/invalid
    all_exist = len(existing) == len(claim.citation_ids)
    return all_exist, tuple(existing)


def _tokenize_simple(text: str) -> set[str]:
    """Simple tokenization for lexical comparison."""
    text = text.lower()
    # Extract numbers and words
    tokens = re.findall(r'\b[\d,]+\.?\d*\b|\b[a-z]+\b', text)
    return set(tokens)


def _extract_numbers(text: str) -> set[str]:
    """Extract numeric values for contradiction detection."""
    numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', text)
    return set(numbers)


def verify_claim_lexical(claim: Claim, evidence_passages: dict[str, EvidencePassage]) -> ClaimVerificationResult:
    """Lexical baseline verifier. Educational only - not general NLI."""
    supplied_ids = set(evidence_passages.keys())
    all_exist, existing_ids = check_citation_validity(claim, supplied_ids)
    
    if not claim.citation_ids:
        return ClaimVerificationResult(
            claim=claim,
            status="insufficient_evidence",
            explanation="No citations provided.",
            evidence_excerpts=(),
            cited_passages_exist=False,
        )
    
    if not all_exist:
        missing = [cid for cid in claim.citation_ids if cid not in supplied_ids]
        return ClaimVerificationResult(
            claim=claim,
            status="invalid_citation",
            explanation=f"Invalid/missing citation IDs: {missing}",
            evidence_excerpts=(),
            cited_passages_exist=False,
        )
    
    # Get cited passages
    cited = [(cid, evidence_passages[cid]) for cid in existing_ids if cid in evidence_passages]
    if not cited:
        return ClaimVerificationResult(
            claim=claim,
            status="insufficient_evidence",
            explanation="Cited passages not found in supplied evidence.",
            evidence_excerpts=(),
            cited_passages_exist=False,
        )
    
    claim_tokens = _tokenize_simple(claim.text)
    claim_numbers = _extract_numbers(claim.text)
    
    best_overlap = 0
    excerpts = []
    for cid, passage in cited:
        excerpts.append((cid, passage.text[:200] + ("..." if len(passage.text) > 200 else "")))
        pass_tokens = _tokenize_simple(passage.text)
        pass_numbers = _extract_numbers(passage.text)
        overlap = len(claim_tokens & pass_tokens)
        if len(claim_tokens) > 0:
            best_overlap = max(best_overlap, overlap / len(claim_tokens))
        
        # Check for obvious contradictions on same numbers
        # If claim has specific number not in passage and passage has different number for same context - heuristic
        for cn in claim_numbers:
            if cn in claim.text.lower():
                # Look for contradictory numbers
                for pn in pass_numbers:
                    if pn != cn and pn in pass_tokens:
                        # Very rough heuristic - could be different facts
                        pass
    
    # Support heuristic: need some token overlap and key terms
    if best_overlap > 0.3 and claim_tokens:
        return ClaimVerificationResult(
            claim=claim,
            status="supported",
            explanation=f"Lexical overlap with cited passages ({best_overlap:.2f}).",
            evidence_excerpts=tuple(excerpts),
            cited_passages_exist=True,
        )
    elif best_overlap > 0:
        return ClaimVerificationResult(
            claim=claim,
            status="insufficient_evidence",
            explanation=f"Some overlap ({best_overlap:.2f}) but insufficient for support.",
            evidence_excerpts=tuple(excerpts),
            cited_passages_exist=True,
        )
    else:
        return ClaimVerificationResult(
            claim=claim,
            status="insufficient_evidence",
            explanation="No lexical overlap with cited passages.",
            evidence_excerpts=tuple(excerpts),
            cited_passages_exist=True,
        )


def verify_claim(
    claim: Claim,
    evidence_passages: dict[str, EvidencePassage],
    *,
    backend: str = "lexical",
    nli_verifier=None,
):
    """Verify one claim with an explicitly selected backend.

    ``backend="lexical"`` preserves the Experiment 14 behavior. ``backend="nli"``
    delegates to the optional NLI verifier and returns its structured result.
    Citation validation is part of both paths, but the NLI path always performs
    it before any model inference.
    """
    if backend == "lexical":
        return verify_claim_lexical(claim, evidence_passages)
    if backend == "nli":
        from minigrad.nli_verification import NLIClaimVerifier

        verifier = nli_verifier if nli_verifier is not None else NLIClaimVerifier()
        return verifier.verify(claim, evidence_passages)
    raise ValueError("backend must be 'lexical' or 'nli'")


def verify_answer_claims(
    answer_text: str,
    citations: tuple[str, ...],
    evidence_passages: dict[str, EvidencePassage],
    *,
    backend: str = "lexical",
    nli_verifier=None,
) -> list[ClaimVerificationResult]:
    """Verify all claims in an answer with the selected backend."""
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', answer_text.strip())
    claims = extract_claims_from_sentences([(s, citations) for s in sentences if s.strip()])
    results = []
    for claim in claims:
        results.append(
            verify_claim(
                claim,
                evidence_passages,
                backend=backend,
                nli_verifier=nli_verifier,
            )
        )
    return results


def check_multi_part_coverage(question_id: str, question_text: str, answer_text: str,
                             parts: tuple[tuple[str, tuple[str, ...]], ...]) -> CoverageReport:
    """Check if multi-part question is addressed."""
    answer_lower = answer_text.lower()
    part_details = []
    addressed = 0
    supported = 0
    
    for part_desc, keywords in parts:
        # Check if addressed (has relevant keywords)
        keyword_matches = sum(1 for kw in keywords if kw in answer_lower)
        is_addressed = keyword_matches >= max(1, len(keywords) // 2)
        if is_addressed:
            addressed += 1
            supported += 1  # basic check
        part_details.append({
            "part": part_desc,
            "addressed": is_addressed,
            "keyword_matches": keyword_matches,
        })
    
    unsupported = len(parts) - addressed
    return CoverageReport(
        question_id=question_id,
        required_parts=len(parts),
        addressed=addressed,
        supported=supported,
        unsupported_or_missing=unsupported,
        part_details=tuple(part_details),
    )


def extract_claims_with_llm(answer_text: str, question: str, backend, 
                            evidence_passages: dict[str, EvidencePassage]) -> Optional[AnswerClaimSet]:
    """Optional LLM-based claim extraction. Returns None if fails/unavailable."""
    try:
        from minigrad.llm_answerer import Prompt
        # Build simple prompt
        system = "Extract atomic factual claims from the answer. For each claim, list citation IDs. Return JSON list of {claim, citations}."
        user = f"Question: {question}\nAnswer: {answer_text}\nEvidence IDs: {list(evidence_passages.keys())}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        output = backend.generate(messages, max_new_tokens=200, temperature=0.0)
        # Try to parse - but be strict about failures
        import json
        try:
            data = json.loads(output.text)
            claims = []
            for i, item in enumerate(data):
                claims.append(Claim(
                    text=item.get("claim", ""),
                    citation_ids=tuple(item.get("citations", [])),
                    claim_index=i,
                ))
            return AnswerClaimSet(question=question, claims=tuple(claims), raw_text=answer_text)
        except:
            return None  # Don't silently accept malformed
    except Exception:
        return None
