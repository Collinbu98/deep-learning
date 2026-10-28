"""Experiment 15: Benchmarking evidence verification.

Evaluates the lexical claim verifier from Experiment 14 against a manually
labeled benchmark of 33 cases spanning direct support, paraphrases,
contradictions, irrelevant citations, invalid citations, high-overlap
distractors, negation, numbers, and comparisons.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

from minigrad.claim_verification import verify_claim_lexical, Claim
from minigrad.llm_answerer import EvidencePassage


def load_benchmark(path=None):
    if path is None:
        path = Path(__file__).parent / "15_verification_benchmark" / "benchmark.json"
    return json.loads(Path(path).read_text())


def make_passage(source_id, text):
    doc_id = source_id.split('#')[0]
    chunk_idx = int(source_id.split('#')[1])
    return EvidencePassage(
        source_id=source_id,
        doc_id=doc_id,
        chunk_index=chunk_idx,
        source=f"{doc_id}.txt",
        text=text,
        score=1.0,
        rank=1,
    )


def evaluate_lexical(cases):
    results = []
    for case in cases:
        passages = {sid: make_passage(sid, text) for sid, text in case['passages'].items()}
        claim = Claim(text=case['claim'], citation_ids=tuple(case['citations']))
        result = verify_claim_lexical(claim, passages)
        results.append({
            'id': case['id'],
            'type': case['type'],
            'expected': case['expected'],
            'predicted': result.status,
            'correct': result.status == case['expected'],
            'explanation': result.explanation,
            'rationale': case['rationale'],
        })
    return results


def compute_metrics(results):
    labels = ['supported', 'contradicted', 'insufficient_evidence', 'invalid_citation']
    total = len(results)
    
    # Confusion matrix
    matrix = {t: Counter() for t in labels}
    for r in results:
        pred = r['predicted'] if r['predicted'] in labels else 'insufficient_evidence'
        exp = r['expected'] if r['expected'] in labels else 'insufficient_evidence'
        matrix[exp][pred] += 1
    
    # Per-class metrics
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[l][label] for l in labels if l != label)
        fn = sum(matrix[label][l] for l in labels if l != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': sum(matrix[label].values())}
    
    # Overall
    accuracy = sum(1 for r in results if r['correct']) / total if total > 0 else 0.0
    macro_f1 = sum(per_class[l]['f1'] for l in labels) / len(labels)
    
    # False supported rate: not supported but marked supported
    not_supported = [r for r in results if r['expected'] != 'supported']
    false_supported = sum(1 for r in not_supported if r['predicted'] == 'supported')
    false_supported_rate = false_supported / len(not_supported) if not_supported else 0.0
    
    return {
        'total': total,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'false_supported_rate': false_supported_rate,
        'confusion_matrix': {k: dict(v) for k, v in matrix.items()},
        'per_class': per_class,
    }


def main():
    cases = load_benchmark()
    results = evaluate_lexical(cases)
    metrics = compute_metrics(results)
    
    print("=" * 70)
    print("Experiment 15: Evidence Verification Benchmark")
    print("=" * 70)
    print(f"\nBenchmark: {metrics['total']} cases")
    
    # Label distribution
    dist = Counter(c['type'] for c in cases)
    print(f"\nLabel distribution by type:")
    for t, c in sorted(dist.items()):
        print(f"  {t}: {c}")
    
    print(f"\nOverall: accuracy={metrics['accuracy']:.3f}, macro-F1={metrics['macro_f1']:.3f}")
    print(f"False-supported rate: {metrics['false_supported_rate']:.3f}")
    
    print("\nPer-class:")
    for label, m in metrics['per_class'].items():
        print(f"  {label:20s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    
    print("\nConfusion matrix (rows=true, cols=pred):")
    labels = ['supported', 'contradicted', 'insufficient_evidence', 'invalid_citation']
    print(f"{'':20s} " + " ".join(f"{l[:12]:>12s}" for l in labels))
    for t in labels:
        row = matrix_str(metrics['confusion_matrix'][t], labels)
        print(f"{t:20s} {row}")
    
    print("\nBy failure type:")
    by_type = defaultdict(list)
    for r in results:
        by_type[r['type']].append(r)
    for t in sorted(by_type.keys()):
        acc = sum(1 for r in by_type[t] if r['correct']) / len(by_type[t])
        print(f"  {t}: {acc:.3f} ({len(by_type[t])} cases)")
    
    print("\nCorrect examples:")
    for r in results[:5]:
        if r['correct']:
            print(f"  ✓ {r['id']}: {r['expected']} - {r['rationale']}")
    
    print("\nFailures:")
    for r in results:
        if not r['correct']:
            print(f"  ✗ {r['id']} [{r['type']}]: expected={r['expected']}, pred={r['predicted']}")
            print(f"     rationale: {r['rationale']}")
            print(f"     exp: {r['explanation']}")


def matrix_str(row_dict, labels):
    parts = []
    for l in labels:
        parts.append(f"{row_dict.get(l,0):>12d}")
    return " ".join(parts)


if __name__ == '__main__':
    main()
