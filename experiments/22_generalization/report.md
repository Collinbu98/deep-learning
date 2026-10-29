# Experiment 22 - Generalization and Robustness Evaluation

Status: completed

## 1. Executive summary

Experiment 21 achieved 100% accuracy on the original 33-case verification
benchmark and reported a development-set accuracy of 0.75 for the final
paraphrase-aware policy. The central question of Experiment 22 was whether that
perfect score reflects a robust verification method or a policy that happens to
fit a small benchmark.

On the newly authored, independently labeled generalization benchmark of 98
cases, the frozen Experiment 21 paraphrase-aware policy (with the Experiment 20
relevance-aware base policy and the Experiment 19 confidence threshold of 0.70)
did **not** achieve 100% accuracy.

- New benchmark, Experiment 21 paraphrase-aware policy: accuracy **0.633**,
  macro-F1 **0.695**, false-supported rate **0.070** (4 false-supported cases),
  incorrect-contradiction rate among insufficient-evidence cases **0.463**
  (19 incorrect contradictions), abstention/coverage **0.673**.
- Original 33-case benchmark, Experiment 21 paraphrase-aware policy: accuracy
  **1.0**, false-supported rate **0.0**, incorrect-contradiction rate **0.0**,
  coverage **0.727**. This result is the previously used, small evaluation set
  and is reported here unchanged for comparison.

The generalization result is therefore disappointing in aggregate terms, but the
failure pattern is informative: the policy abstains on many supported cases
(incomplete recall) and incorrectly contradicts many insufficient-evidence or
high-overlap cases (over-claiming contradictions).

## 2. Benchmark design

- Path: `experiments/22_generalization/benchmark.json`
- Cases: 98, independently authored for Experiment 22.
- Type distribution: direct 18, paraphrase 15, high_overlap_wrong 11,
  negation 4, contradiction 10, irrelevant_citation 9, invalid_citation 2,
  missing_citation 1, insufficient 13, multi_component 5, conflicting_evidence 5,
  long_passage 5.

Coverage areas represented: direct support, genuine paraphrases, lexical and
topical similarity without entailment, contradictions (negation, numbers, dates,
entities, qualifiers), irrelevant citations, relevant evidence without establishing
support, multiple passages jointly supporting a claim, conflicting passages, invalid
and missing citations, multi-component claims, and long passages / window boundary
cases.

## 3. Annotation methodology

- Labels: `supported`, `contradicted`, `insufficient_evidence`, `invalid_citation`.
- Annotation rules: support requires establishing evidence; contradiction
  requires a conflict on the same factual attribute; overlap alone is insufficient;
  multi-component claims require conjunction; conflicting passages yield
  insufficient evidence under conservative aggregation.
- Ambiguous cases were flagged and resolved conservatively (documented in
  `experiments/22_generalization/annotation.md`).
- A second review pass focused on label ambiguity was performed; no label was
  changed during evaluation.

## 4. Overlap audit

- No case ID appeared in more than one set across the fixed benchmark and the
  development sets.
- Cross-set high token-overlap pairs between the new benchmark and prior sets:
  - (`experiments/15_verification_benchmark/benchmark.json` `p4` "The Solar
    System's largest planet is Jupiter." vs `experiments/22_generalization/benchmark.json`
    `g36` "The Moon is the largest planet in the Solar System.") - token overlap
    0.62.
  - Same prior `p4` case vs new `g44` "Saturn is the largest planet in the Solar
    System." - token overlap 0.62.
  - (`experiments/19_confidence_policy/dev_set.json` `dev_p1` vs new `g94`) -
    token overlap 0.64.

- No normalized claim text was shared across the new benchmark and the original
  33-case benchmark.

## 5. Frozen policy configurations

All policies were evaluated with the same frozen configuration:

- Strategy: `whole_passage`
- Adjacent context: 0
- Contradiction threshold: 0.70 (inherited from Experiment 19 development selection)
- Model: `cross-encoder/nli-MiniLM2-L6-H768` (cached snapshot, local_files_only)
- Device: CPU for this run; dtype float32.

Policies:

- `lexical_baseline`: Experiment 14 lexical verifier.
- `nli_conservative`: Experiment 18 windowed conservative NLI aggregation.
- `exp19_confidence`: Experiment 19 confidence-aware policy.
- `exp20_relevance_aware`: Experiment 20 relevance-aware policy (lexical topic
  anchor guard on top of the confidence policy).
- `exp21_paraphrase_aware`: Experiment 21 paraphrase-aware policy (paraphrase
  promotion on top of the relevance-aware policy).

## 6. Comparative metrics on the new benchmark (98 cases)

| Policy | Accuracy | Macro-F1 | False-supported rate | Incorrect-contradiction rate | Coverage |
| --- | --- | --- | --- | --- | --- |
| lexical_baseline | 0.500 | 0.470 | 0.860 | 0.000 | 0.939 |
| nli_conservative | 0.592 | 0.675 | 0.053 | 0.610 | 0.735 |
| exp19_confidence | 0.602 | 0.679 | 0.053 | 0.537 | 0.684 |
| exp20_relevance_aware | 0.622 | 0.691 | 0.053 | 0.463 | 0.643 |
| exp21_paraphrase_aware | 0.633 | 0.695 | 0.070 | 0.463 | 0.673 |

Experiment 21 vs. earlier policies on the new benchmark:

- vs. lexical baseline: accuracy delta +0.133, macro-F1 delta +0.225, false-
  supported rate delta -0.790, incorrect-contradiction delta +0.463, coverage delta
  -0.266.
- vs. conservative NLI: accuracy delta +0.041, macro-F1 delta +0.020, false-
  supported rate delta +0.017, incorrect-contradiction delta -0.147, coverage delta
  -0.062.
- vs. Experiment 19 confidence: accuracy delta +0.031, macro-F1 delta +0.016,
  false-supported rate delta +0.017, incorrect-contradiction delta -0.074,
  coverage delta -0.011.
- vs. Experiment 20 relevance: accuracy delta +0.011, macro-F1 delta +0.004,
  false-supported rate delta +0.017, incorrect-contradiction delta ~0.000,
  coverage delta +0.030.

The Experiment 21 paraphrase-aware policy did not clearly dominate the
Experiment 20 relevance-aware policy on the new benchmark: the differences are
small and the paraphrase policy introduced additional abstentions.

## 7. Robustness metrics for the Experiment 21 policy on the new benchmark

- Accuracy: 0.633 (62/98 correct).
- Macro-F1: 0.695.
- Confusion matrix (rows = expected, columns = predicted):
  - supported: 32 correct, 9 supported-cases abstained (predicted
    insufficient_evidence).
  - contradicted: 9 correct contradicted, 4 abstained, 1 false-supported.
  - insufficient_evidence: 19 predicted contradicted (incorrect contradictions
    among insufficient cases), 19 abstained correctly, 3 supported-cases predicted
    insufficient (not false-supported; these are recall misses).
  - invalid_citation: 2/2 correct.
- Per-class precision/recall/F1:
  - supported: P=0.889, R=0.780, F1=0.831, support=41.
  - contradicted: P=0.321, R=0.643, F1=0.429, support=14.
  - insufficient_evidence: P=0.594, R=0.463, F1=0.521, support=41.
  - invalid_citation: P=1.000, R=1.000, F1=1.000, support=2.
- False-supported: 4 cases out of 57 expected-not-supported cases (rate 0.070).
  Cases: `g30`, `g31`, `g103`, `g124`.
- Incorrect contradictions among insufficient-evidence cases: 19 out of 41
  (rate 0.463).

## 8. Representative errors

False-supported cases (expected insufficient / contradicted, predicted supported):

- `g30` (high_overlap_wrong): expected insufficient_evidence, predicted supported
  via `unanimous_entailment`; shared predicate and value but different substance.
- `g31` (high_overlap_wrong): expected insufficient_evidence, predicted supported;
  numeric overlap but different phenomenon.
- `g103` (long_passage): expected insufficient_evidence, predicted supported via
  `paraphrase_supported`; conflicting values inside one long passage.
- `g124` (contradiction): expected contradicted, predicted supported via
  `unanimous_entailment`; two conflicting numeric values in one claim.

Other errors (13): supported-cases predicted insufficient (e.g. `g9`, `g13`, `g14`,
`g21`, `g83`, `g93`, `g101`, `g110`, `g111`), contradictions abstained or demoted
(e.g. `g44`, `g45`, `g107`), and conflicting/insufficient cases predicted
contradicted.

## 9. Limitations

- The paraphrase check is deterministic and narrow, not a general semantic
  equivalence model.
- The new benchmark is manually authored and small.
- A single small cross-encoder NLI model was used.
- Abstention means low recall on supported cases can accompany zero
  false-supported cases.
- The original 33-case benchmark is small and was previously used for policy
  freezing decisions.

## 10. Exact reproduction commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python experiments/22_generalization.py --device cpu
.venv/bin/python experiments/22_generalization.py --device auto
```

## 11. Final test count

- Final full-suite result: **403 passed**.
