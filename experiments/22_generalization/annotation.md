# Experiment 22 - Generalization and Robustness Benchmark: Annotation Guidelines

This document describes how the independently authored generalization benchmark
in `benchmark.json` was annotated, which annotation ambiguities were flagged,
and how the annotation was reviewed before any evaluation numbers were computed.

## 1. Source of the benchmark

- The benchmark contains 98 manually authored cases under the `g*` identifier
  namespace (`experiments/22_generalization/benchmark.json`).
- Cases were written from general knowledge and the existing corpus vocabulary
  (`data/rag_corpus`) as topic anchors, not by re-running or re-labeling any
  prior experiment output.
- The original 33-case Experiment 15 benchmark (`experiments/15_verification_benchmark/benchmark.json`)
  is kept unchanged and is treated strictly as a previously used evaluation set
  for Experiment 22, not as a development or training set.

## 2. Label definitions

- `supported`: The cited evidence, taken as given, establishes the claim.
- `contradicted`: The cited evidence conflicts with the claim on the same
  factual attribute (negation, number, date, entity, qualifier, comparison
  direction).
- `insufficient_evidence`: The citation is valid and topically relevant, but
  the evidence does not establish the claim; the verifier is expected to abstain
  or remain conservative.
- `invalid_citation`: The citation ID is not present in the supplied evidence
  set, or no citation was supplied at all.

`insufficient_evidence` is the abstention-style label for cases where support is
unestablished. It is distinct from backend errors, which are counted separately.

## 3. Case-type taxonomy used in this benchmark

- `direct`: simple factual support.
- `paraphrase`: genuine restatements accepted as supported.
- `high_overlap_wrong`: lexical or topical overlap without entailment (expected
  `insufficient_evidence`).
- `negation`: negated claims about supported facts (expected `contradicted`).
- `contradiction`: conflicting numbers, dates, entities, qualifiers, or
  comparisons.
- `irrelevant_citation`: cited passage concerns a different subject.
- `missing_citation`: no citation supplied (expected `insufficient_evidence`).
- `invalid_citation`: citation ID not resolvable in the supplied evidence.
- `insufficient`: relevant evidence that does not establish the claim.
- `multi_component`: claims bundling two or more factual components.
- `conflicting_evidence`: multiple cited passages that conflict with each other.
- `long_passage`: support or conflict located inside long multi-sentence passages.

## 4. Annotation rules applied

1. A claim supported by a cited passage with no conflicting value is annotated
   `supported`.
2. A claim conflicting with a cited passage on the same attribute is annotated
   `contradicted`, including negations of supported facts.
3. A claim whose citation is missing or unresolvable is annotated
   `invalid_citation` (or `insufficient_evidence` when no citation was supplied).
4. A claim that is relevant but not established by the cited evidence is
   annotated `insufficient_evidence`.
5. Lexical or topical overlap alone never implies support or contradiction; such
   cases are annotated `insufficient_evidence` unless a real conflict exists.
6. Multi-component claims are resolved by conjunction: full support requires all
   components established; partial support or unsupported components yield
   `insufficient_evidence`.
7. Conflicting cited passages yield `insufficient_evidence` under the frozen
   conservative aggregation.

## 5. Flagged ambiguous cases

The following cases were considered ambiguous during annotation and were
resolved conservatively before labeling:

- `g45` (contradiction, qualifier "entirely covered by ice" vs. hot-desert
  classification): labeled `contradicted` because the qualifier directly opposes
  the classified temperature regime.
- `g51` (Amazon-superlative claim cited against a Nile passage): labeled
  `insufficient_evidence`; topical adjacency was accepted as relevant for the
  annotation but not as support.
- `g108` (river-length comparison across an ocean passage): labeled
  `insufficient_evidence`.
- `g112` (internally inconsistent claim): labeled `insufficient_evidence` because
  the cited passage could not resolve the internal conflict.

## 6. Second review pass

A second review pass focused on potential label ambiguity was performed over all
98 cases: claims were re-read against their cited passages, and expected labels
were re-checked for negation, numeric, date, entity, qualifier, and comparison
errors. No label was changed during evaluation. Cases where the annotation and
the eventual policy disagreement could not be resolved were left as authored;
they are reported as errors in the final results rather than silently relabeled.

## 7. Annotation guidelines for ambiguity

- Prefer `insufficient_evidence` when in doubt about support, because abstention
  is the safe failure mode for a verifier.
- Prefer `contradicted` only when the conflict is on the same attribute.
- Do not annotate `supported` when a component of a multi-component claim is
  unestablished.
