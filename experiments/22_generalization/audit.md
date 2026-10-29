# Experiment 22 - Evaluation Setup Audit (Experiments 15-21)

This audit documents how the evaluation setup looked at the close of Experiment
21, so that the Experiment 22 generalization run can be interpreted against the
full history of sets, labels, and policy decisions.

## 1. Sets used for development versus evaluation

| Set | Path | Role |
| --- | --- | --- |
| Fixed benchmark (evaluation) | `experiments/15_verification_benchmark/benchmark.json` | 33 manually labeled cases; used to produce the 100% Experiment 21 headline. |
| Experiment 19 development set | `experiments/19_confidence_policy/dev_set.json` | 26 cases; source of the threshold-selection candidates. |
| Experiment 20 relevance extension | `experiments/20_evidence_relevance/dev_relevance.json` | 7 cases written after Experiment 19, before the Experiment 20 policy; combined with the Experiment 19 set into one development pool. |
| Experiment 21 development set | `experiments/21_paraphrase_verification/dev_set.json` | 8 cases; source of the paraphrase-policy behavior checks. |

The Experiment 20 policy was evaluated on the combined 33-case development pool
(26 Experiment 19 cases + 7 relevance-extension cases). The Experiment 21 policy
was evaluated on its own 8-case development set. The fixed Experiment 15
benchmark was used as the final evaluation set for the frozen policies.

## 2. Which examples influenced policy design or threshold selection

- The Experiment 19 confidence threshold (0.70) was selected from development-set
  threshold candidates via `select_threshold` on the Experiment 19 development
  set; it was then inherited by the Experiment 20 relevance-aware policy and the
  Experiment 21 paraphrase-aware policy.
- The Experiment 20 relevance-aware policy was introduced after Experiment 19
  showed that the confidence threshold alone could not reject high-confidence
  contradictions against unrelated subjects.
- The Experiment 21 paraphrase-aware policy was introduced after Experiment 20
  failed to fix the two remaining paraphrase cases (`p1`, `p2`) on the fixed
  benchmark; the failure mechanism was that the NLI backend returned `neutral`,
  so no policy could promote the decision.
- The Experiment 21 development set (8 cases) was written before paraphrase-aware
  evaluation and drove the deterministic paraphrase-equivalence groups.

## 3. How the reported development accuracy of 0.75 is calculated

On the Experiment 21 development set (8 cases), the new paraphrase-aware policy
scored accuracy 0.75 (6/8 correct), macro-F1 0.4286, false-supported rate 0.0
(denominator: 4 expected-not-supported cases), incorrect-contradiction rate 0.0
(denominator: 3 expected-insufficient cases), and coverage 0.5 (4 definitive, 4
abstained). The 0.75 accuracy corresponds to the 6 supported/paraphrase cases that
were promoted plus the correctly handled contradiction and insufficient cases;
it is a small-sample development-set result, not a population estimate.

## 4. Label definitions and abstention semantics

- `supported`: cited evidence establishes the claim.
- `contradicted`: cited evidence conflicts with the claim on the same attribute.
- `insufficient_evidence`: support unestablished (abstention-style outcome).
- `invalid_citation`: citation missing or unresolvable.

Backend errors (`backend_unavailable`, `verification_error`) are counted separately
from abstentions.

## 5. Overlap audit findings

- No case ID appears in more than one set across the fixed benchmark and the
  development sets.
- Near-duplicate claim texts were found within the fixed benchmark and development
  sets (for example, `s1` and `i1` share the Jupiter moon claim; `s3` and `dev_w2`
  share the Venus longest-day claim). These are internal to prior sets and were
  not re-labeled here.
- Cross-set comparison between the Experiment 22 `g*` benchmark and the prior
  sets found only three high token-overlap pairs, all benign: Jupiter/Saturn
  largest-planet paraphrases (`p4`/`g36`, `p4`/`g44`) and a Sun-mass paraphrase
  (`dev_p1`/`g94`).
- Within the new Experiment 22 benchmark, a small number of repeated claim texts
  were used for citation-handling variants (for example, the Amazon-longest-river
  claim reused for invalid/missing citation cases). This is intentional within-set
  variation, not cross-set duplication.

## 6. Known prior-report discrepancies

- Experiment 17/18/19/20/21 stored results were recomputed from frozen confusion
  matrices; the historical note about a 0.636 vs. 0.545 lexical accuracy discrepancy
  was carried forward as a reporting-audit finding.

## 7. Conclusion of the audit

The prior evaluation record rests on a small fixed benchmark plus small
development pools, with a manually selected threshold. The Experiment 22
generalization run is intended to test whether the frozen policies generalize
beyond those sets.
