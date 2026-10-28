"""Experiment 12: BM25 vs. semantic and hybrid retrieval.

The benchmark is exactly Experiment 11's corpus, questions, relevance labels,
chunking configurations, and document-level metrics. Each run compares:

* BM25, the Experiment 11 lexical baseline;
* cosine similarity over an embedding backend;
* reciprocal rank fusion (RRF) of BM25 and the embedding retriever.

The embedding backend is either ``sentence-transformers`` (compact pretrained
``all-MiniLM-L6-v2``, 384 dimensions) or the deterministic ``hash`` fallback.
The hash backend is a signed bag-of-words baseline, **not** a semantic model:
it is useful as an offline plumbing check and as evidence that vector search
alone does not fix lexical mismatch. Run the benchmark twice to compare the
two backends, and report them separately.

This experiment measures retrieval and abstention, not answer generation.
Answers shown in ``--ask``/``--interactive`` mode are still produced by the
Experiment 11 extractive answerer, so a retrieval success can still produce a
weak answer.

Abstention for the semantic method uses a cosine-similarity threshold. The
primary threshold is calibrated before evaluation from ten in-domain
*unanswerable* probe questions that are not in ``questions.json``: the
threshold is the maximum top-1 similarity those probes achieve. This is a
small, corpus-specific calibration, not a calibrated probability. A
sensitivity table is printed for transparency; it is diagnostic only and is
not used to choose the reported threshold. False abstentions (answerable
questions declined) and false acceptances (unanswerable questions answered)
are reported separately.

Usage
-----
    # Pretrained comparison (uses a local model/cache; downloads if missing)
    python experiments/12_semantic_retrieval.py --device cpu

    # Offline, non-semantic hash baseline
    python experiments/12_semantic_retrieval.py --backend hash

    # Never download; fail if the model is not already cached
    python experiments/12_semantic_retrieval.py --local-files-only --device cpu

    # One chunk configuration and one question
    python experiments/12_semantic_retrieval.py --chunk-size 400 --overlap 80 \
        --ask "Which world has the most natural satellites?"

    # Interactive retrieval comparison
    python experiments/12_semantic_retrieval.py --interactive
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from minigrad.embeddings import (
    DEFAULT_MODEL_NAME,
    EmbeddingModelUnavailable,
    HashingEmbeddingBackend,
    SentenceTransformerEmbeddings,
)
from minigrad.rag import (
    RAGPipeline,
    evaluate_retrieval,
    load_corpus,
    load_questions,
)
from minigrad.semantic_rag import (
    HybridRAGPipeline,
    SemanticRAGPipeline,
    calibrate_similarity_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = ROOT / "data" / "rag_corpus"
CONFIGS = ((400, 80), (900, 0))
EVAL_KS = (1, 3, 5)

# Ten in-domain questions whose answers are not in the corpus. They are
# disjoint from questions.json and are used only to calibrate the semantic
# abstention threshold. They mention other Solar System bodies rather than
# out-of-domain topics, because out-of-domain probes score much lower and
# would produce a threshold that is far too permissive for this corpus.
CALIBRATION_QUESTIONS = [
    "What is the surface gravity of Neptune?",
    "How many moons does Uranus have?",
    "What is the mass of the asteroid Ceres?",
    "How deep is the ocean on Europa?",
    "What is the orbital period of Halley's comet?",
    "Who discovered Titania?",
    "How large is the Kuiper Belt?",
    "What is the temperature of Eris?",
    "How many rings does Uranus have?",
    "What is the diameter of the dwarf planet Makemake?",
]

FAILURE_QUESTIONS = ("q03", "q04", "q08", "q09", "q14", "q12", "q13")
SENSITIVITY_THRESHOLDS = (0.30, 0.40, 0.50, 0.60)


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
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--backend",
        choices=("sentence-transformers", "hash"),
        default="sentence-transformers",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or a torch device such as cuda:0",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--hash-dimension",
        type=int,
        default=256,
        help="dimension of the offline hashing backend",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="for the pretrained backend, never download; use only a local path/cache",
    )
    parser.add_argument("--calibration-percentile", type=float, default=100.0)
    parser.add_argument("--calibration-margin", type=float, default=0.0)
    parser.add_argument("--max-sentences", type=int, default=2)
    parser.add_argument(
        "--ask",
        action="append",
        default=[],
        help="ask one question after the benchmark (repeatable)",
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    return parser.parse_args(argv)


def load_backend(args):
    """Return ``(backend, device_label)``, verifying inference on the device."""
    if args.backend == "hash":
        backend = HashingEmbeddingBackend(dimension=args.hash_dimension)
        return backend, "cpu (hash fallback, not semantic)"

    if args.device == "auto":
        candidates = ["cpu"]
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    candidates.insert(0, "cuda:0")
            except Exception:
                pass
    else:
        candidates = [args.device]

    errors = []
    for device in candidates:
        print(f"Loading embedding model {args.model!r} on {device} ...")
        backend = SentenceTransformerEmbeddings(
            model_name=args.model,
            device=device,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
        try:
            probe = backend.encode(["embedding probe"])
            if len(probe) != 1 or not probe[0]:
                raise EmbeddingModelUnavailable("model returned an invalid probe vector")
            return backend, device
        except EmbeddingModelUnavailable as error:
            errors.append(f"{device}: {error}")
            print(f"  failed: {error}")
    raise SystemExit(
        "Could not load the pretrained embedding model.\n  "
        + "\n  ".join(errors)
        + "\nInstall sentence-transformers and ensure the model is available, "
        "or run with `--backend hash` for the offline non-semantic fallback."
    )


def print_embedding_info(backend, device, args):
    print("Embedding backend")
    print("-----------------")
    print(f"  model          : {backend.model_name}")
    print(f"  resolved name  : {getattr(backend, 'resolved_model_name') or backend.model_name}")
    print(f"  requested device: {device}")
    print(f"  normalization  : {getattr(backend, 'normalize_embeddings', True)}")
    print(f"  batch size     : {args.batch_size}")
    print(f"  dependency     : sentence-transformers (optional)")
    print(f"  downloaded     : {getattr(backend, 'was_downloaded')}")
    if backend.model_name.startswith("hashing-"):
        print(
            "  WARNING: the hash backend is a lexical/signature fallback, not a "
            "semantic model."
        )
        print(
            "           Its rankings and calibrated threshold are only a "
            "plumbing check."
        )
    print()


def print_corpus_stats(documents, questions):
    print("Corpus")
    print("------")
    print(f"  documents : {len(documents)}")
    print(f"  characters: {sum(len(doc.text) for doc in documents)}")
    print(f"  questions : {len(questions)}")
    print(f"  answerable: {sum(q.answerable for q in questions)}")
    print(f"  unanswerable: {sum(not q.answerable for q in questions)}")
    print()


def print_metrics(method, pipeline, questions):
    print(f"  {method}:")
    for k in EVAL_KS:
        metrics = evaluate_retrieval(pipeline, questions, k=k)
        print(
            f"    k={k}: recall {metrics['recall_at_k']:.3f}  "
            f"hit {metrics['hit_at_k']:.3f}  "
            f"MRR {metrics['mrr_at_k']:.3f}  "
            f"abstain(unanswerable) {metrics['abstention_rate_unanswerable']:.3f}  "
            f"false abstain {metrics['false_abstention_rate']:.3f}"
        )


def print_per_question(methods, questions, k=3):
    print(f"  per-question @ k={k}:")
    rows = {
        name: {row["id"]: row for row in evaluate_retrieval(pipeline, questions, k=k)["rows"]}
        for name, pipeline in methods.items()
    }
    body = []
    for question in questions:
        cells = []
        for name in methods:
            row = rows[name][question.id]
            status = "abs" if row["abstained"] else "ans"
            retrieved = ",".join(row["retrieved_docs"]) or "-"
            cells.append(f"{status} [{retrieved}]")
        body.append((question.id, cells))
    width = max(28, max(len(cell) + 2 for _, cells in body for cell in cells))
    print("    id   " + "".join(f"{name:<{width}}" for name in methods))
    for question_id, cells in body:
        print(f"    {question_id}  " + "".join(f"{cell:<{width}}" for cell in cells))
    print()


def print_failure_analysis(methods, questions, top_k=3):
    by_id = {question.id: question for question in questions}
    print("Failure-case inspection (top 3 passages)")
    print("----------------------------------------")
    for question_id in FAILURE_QUESTIONS:
        question = by_id[question_id]
        relevant = ",".join(question.relevant_docs) or "-"
        print(f"{question_id} [{question.category}] {question.question}")
        print(f"  labelled relevant documents: {relevant}")
        for name, pipeline in methods.items():
            passages = pipeline.retrieve(question.question, top_k=top_k)
            rendered = " | ".join(
                f"{passage.rank}. {passage.chunk.doc_id}#{passage.chunk.chunk_index} "
                f"({passage.score:.3f})"
                for passage in passages
            )
            print(f"  {name:<7} {rendered or '-'}")
            if passages:
                snippet = " ".join(passages[0].chunk.text.split())[:100]
                print(f"          top text: {snippet!r}")
        print()


def print_abstention_sensitivity(pipeline, questions, calibrated, label):
    print(f"  Semantic abstention sensitivity ({label})")
    print("    threshold  unanswerable-abstain  false-abstain")
    thresholds = sorted(set(SENSITIVITY_THRESHOLDS) | {round(calibrated, 3)})
    original = pipeline.min_similarity
    try:
        for threshold in thresholds:
            pipeline.min_similarity = threshold
            metrics = evaluate_retrieval(pipeline, questions, k=3)
            marker = " <- calibrated" if threshold == round(calibrated, 3) else ""
            print(
                f"    {threshold:9.3f}  {metrics['abstention_rate_unanswerable']:19.3f}"
                f"  {metrics['false_abstention_rate']:12.3f}{marker}"
            )
    finally:
        pipeline.min_similarity = original


def main(argv=None):
    args = parse_args(argv)
    documents = load_corpus(args.corpus_dir)
    if not documents:
        raise SystemExit(f"no .txt files found in {args.corpus_dir}")

    questions_path = args.questions or args.corpus_dir / "questions.json"
    if not questions_path.exists():
        raise SystemExit(f"questions file not found: {questions_path}")
    questions = load_questions(questions_path)
    if not questions:
        raise SystemExit("question set is empty")

    print_corpus_stats(documents, questions)
    backend, device = load_backend(args)
    print_embedding_info(backend, device, args)

    if args.chunk_size is not None:
        overlap = args.overlap if args.overlap is not None else 0
        if overlap < 0:
            raise SystemExit("--overlap must be non-negative")
        configs = ((args.chunk_size, overlap),)
    else:
        configs = CONFIGS

    for chunk_size, overlap in configs:
        print("=" * 78)
        print(f"Chunk configuration: chunk_size={chunk_size}, overlap={overlap}")
        print("=" * 78)

        bm25 = RAGPipeline.from_documents(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=args.top_k,
        )
        semantic = SemanticRAGPipeline.from_documents(
            documents,
            backend,
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )
        if [chunk.text for chunk in bm25.chunks] != [
            chunk.text for chunk in semantic.chunks
        ]:
            raise SystemExit("internal error: methods received different chunks")

        threshold, calibration_scores = calibrate_similarity_threshold(
            semantic.index,
            CALIBRATION_QUESTIONS,
            percentile=args.calibration_percentile,
            margin=args.calibration_margin,
        )
        semantic.min_similarity = threshold

        print(f"  chunks: {len(semantic.chunks)}")
        print(f"  embedding dimension: {semantic.index.dimension}")
        print(f"  chunks embedded once at index build: {len(semantic.index.texts)}")
        print(f"  cosine abstention threshold: {threshold:.3f}")
        print(
            "  calibration top-1 similarities: "
            + ", ".join(f"{score:.3f}" for score in calibration_scores)
        )
        print()

        methods = {"BM25": bm25, "semantic": semantic}
        if not args.no_hybrid:
            methods["hybrid"] = HybridRAGPipeline(bm25, semantic, rrf_k=args.rrf_k)

        print("Retrieval metrics")
        print("-----------------")
        for name, pipeline in methods.items():
            print_metrics(name, pipeline, questions)
        print()
        print_per_question(methods, questions, k=3)
        print_abstention_sensitivity(semantic, questions, threshold, "diagnostic only")
        print()
        print_failure_analysis(methods, questions, top_k=3)

    print("Answer generation is unchanged from Experiment 11; these are")
    print("retrieval and abstention results, not answer-quality results.")
    print()

    show_evidence = True
    pipeline_for_answers = methods.get("hybrid") or methods["semantic"]
    questions_to_answer = list(args.ask)
    if questions_to_answer or args.interactive:
        print("Answers (extractive, via the selected method's passages)")
        print("------------------------------------------------------")
        print(f"  showing answers from: { 'hybrid' if not args.no_hybrid else 'semantic' }")
        for question in questions_to_answer:
            answer = pipeline_for_answers.answer(question, max_sentences=args.max_sentences)
            print(f"Q: {question}")
            if show_evidence:
                for passage in answer.passages:
                    text = " ".join(passage.chunk.text.split())
                    if len(text) > 140:
                        text = text[:137].rstrip() + "..."
                    print(
                        f"  {passage.rank}. {passage.chunk.source} "
                        f"chunk {passage.chunk.chunk_index} ({passage.score:.3f})  {text}"
                    )
            if answer.abstained:
                print(f"A: [abstained] {answer.reason}")
            else:
                print(f"A (extractive): {answer.text}")
                for sentence in answer.sentences:
                    print(
                        f"   source: {sentence.source} chunk {sentence.chunk_index} "
                        f"(score {sentence.score:.3f})"
                    )
            print()
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
                answer = pipeline_for_answers.answer(question, max_sentences=args.max_sentences)
                print(f"Q: {question}")
                for passage in answer.passages:
                    text = " ".join(passage.chunk.text.split())
                    if len(text) > 140:
                        text = text[:137].rstrip() + "..."
                    print(
                        f"  {passage.rank}. {passage.chunk.source} "
                        f"chunk {passage.chunk.chunk_index} ({passage.score:.3f})  {text}"
                    )
                if answer.abstained:
                    print(f"A: [abstained] {answer.reason}")
                else:
                    print(f"A (extractive): {answer.text}")
                print()


if __name__ == "__main__":
    main()
