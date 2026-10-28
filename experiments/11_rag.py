"""Experiment 11: retrieval-augmented generation with BM25 and extraction.

This experiment builds a RAG system from scratch on a small, checked-in
Solar System corpus. There is no neural network and no text generation:

    documents -> character chunks -> BM25 index -> retrieved passages
              -> sentence extraction -> answer with source attribution

The point is to measure retrieval honestly. The script reports document-level
recall, hit rate, MRR, and abstention on a small set of questions, for two
chunk configurations chosen in advance rather than tuned on the test set.

Usage
-----
    python experiments/11_rag.py
    python experiments/11_rag.py --chunk-size 400 --overlap 80
    python experiments/11_rag.py --ask "How many moons does Mars have?"
    python experiments/11_rag.py --interactive

The corpus lives in ``data/rag_corpus``. See its README for the relevance
definition and the known limitations of the fixtures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from minigrad.rag import (
    RAGPipeline,
    evaluate_retrieval,
    load_corpus,
    load_questions,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = ROOT / "data" / "rag_corpus"
CONFIGS = ((400, 80), (900, 0))
EVAL_KS = (1, 3, 5)

EXAMPLE_QUESTIONS = [
    "What is the diameter of Jupiter?",
    "How long is a day on Venus?",
    "Which is hotter, Mercury or Venus, and which of them is closer to the Sun?",
    "Which planet has the shortest day?",
    "What is the surface gravity of Pluto?",
    "Which world has the most natural satellites?",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="questions JSON (default: <corpus-dir>/questions.json)",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--max-sentences", type=int, default=2)
    parser.add_argument(
        "--ask",
        action="append",
        default=[],
        help="ask one question after the evaluation (repeatable)",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="do not print the retrieved passages before each answer",
    )
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args(argv)


def build_pipeline(documents, chunk_size, overlap, args):
    return RAGPipeline.from_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
        k1=args.k1,
        b=args.b,
        top_k=args.top_k,
        min_score=args.min_score,
        min_coverage=args.min_coverage,
    )


def print_corpus_stats(documents, questions):
    print("Corpus")
    print("------")
    print(f"  documents : {len(documents)}")
    print(f"  characters: {sum(len(doc.text) for doc in documents)}")
    print(f"  questions : {len(questions)}")
    print(f"  answerable: {sum(q.answerable for q in questions)}")
    unanswerable = sum(not q.answerable for q in questions)
    print(f"  unanswerable: {unanswerable}")
    print()


def print_evaluation(pipeline, questions, label):
    print(f"Chunk configuration: {label}")
    print(f"  chunks: {len(pipeline.chunks)}")
    lengths = [len(chunk.text) for chunk in pipeline.chunks]
    print(
        f"  chunk characters: min {min(lengths)}, max {max(lengths)}, "
        f"mean {sum(lengths) / len(lengths):.1f}"
    )
    print(
        f"  BM25: k1={pipeline.retriever.k1}, b={pipeline.retriever.b}, "
        f"top_k={pipeline.top_k}, min_coverage={pipeline.min_coverage}"
    )
    n_answerable = sum(1 for question in questions if question.answerable)
    print(
        f"  questions: {len(questions)} "
        f"({n_answerable} answerable, {len(questions) - n_answerable} unanswerable)"
    )
    for k in EVAL_KS:
        metrics = evaluate_retrieval(pipeline, questions, k=k)
        print(
            f"  k={k}: recall {metrics['recall_at_k']:.3f}  "
            f"hit {metrics['hit_at_k']:.3f}  "
            f"MRR {metrics['mrr_at_k']:.3f}  "
            f"abstain(unanswerable) {metrics['abstention_rate_unanswerable']:.3f}  "
            f"false abstain {metrics['false_abstention_rate']:.3f}"
        )

    metrics = evaluate_retrieval(pipeline, questions, k=3)
    print("  per-question @ k=3:")
    for row in metrics["rows"]:
        status = "abstain" if row["abstained"] else "answer "
        rr = "-" if row["reciprocal_rank"] is None else f"{row['reciprocal_rank']:.2f}"
        retrieved = ",".join(row["retrieved_docs"]) or "-"
        expected = ",".join(row["relevant_docs"]) or "-"
        print(
            f"    {row['id']}  {status}  rr={rr}  "
            f"retrieved=[{retrieved}] expected=[{expected}]"
        )
    print()


def print_evidence(passages, preview_chars=180):
    """Show the BM25 passages on which the extractive answer is based."""
    print("  retrieved evidence (BM25 lexical matches, not semantic):")
    if not passages:
        print("    (no passage shared a term with the question)")
        return
    for passage in passages:
        text = " ".join(passage.chunk.text.split())
        if len(text) > preview_chars:
            text = text[: preview_chars - 3].rstrip() + "..."
        print(
            f"    {passage.rank}. {passage.chunk.source} "
            f"chunk {passage.chunk.chunk_index} score {passage.score:.3f}"
        )
        print(f"       {text}")


def print_answer(pipeline, question, max_sentences, show_evidence=True):
    answer = pipeline.answer(question, max_sentences=max_sentences)
    print(f"Q: {question}")
    if show_evidence:
        print_evidence(answer.passages)
    if answer.abstained:
        print(f"A: [abstained] {answer.reason}")
        print()
        return
    print(f"A (extractive, copied from the sources below): {answer.text}")
    for sentence in answer.sentences:
        print(
            f"   source: {sentence.source} chunk {sentence.chunk_index} "
            f"(score {sentence.score:.3f})"
        )
    print()


def main(argv=None):
    args = parse_args(argv)
    documents = load_corpus(args.corpus_dir)
    if not documents:
        raise SystemExit(f"no .txt files found in {args.corpus_dir}")

    questions_path = args.questions or args.corpus_dir / "questions.json"
    questions = load_questions(questions_path) if questions_path.exists() else []
    print_corpus_stats(documents, questions)

    if args.chunk_size is not None:
        overlap = args.overlap if args.overlap is not None else 0
        if overlap < 0:
            raise SystemExit("--overlap must be non-negative")
        configs = ((args.chunk_size, overlap),)
    else:
        configs = CONFIGS

    pipelines = []
    for chunk_size, overlap in configs:
        pipeline = build_pipeline(documents, chunk_size, overlap, args)
        pipelines.append(pipeline)
        if questions:
            label = f"chunk_size={chunk_size}, overlap={overlap}"
            print_evaluation(pipeline, questions, label)

    pipeline = pipelines[0]
    show_evidence = not args.no_evidence
    print("Example answers")
    print("---------------")
    print(
        f"(using chunk_size={pipeline.chunk_size}, overlap={pipeline.overlap}; "
        f"top_k={pipeline.top_k})"
    )
    for question in EXAMPLE_QUESTIONS + list(args.ask):
        print_answer(pipeline, question, args.max_sentences, show_evidence)

    if args.interactive:
        print("Interactive mode. Enter an empty line, 'exit', or 'quit' to stop.")
        while True:
            try:
                question = input("Question: ").strip()
            except EOFError:
                print()
                break
            if not question or question.lower() in {"exit", "quit"}:
                break
            print_answer(pipeline, question, args.max_sentences, show_evidence)


if __name__ == "__main__":
    main()
