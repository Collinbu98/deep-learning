"""Experiment 13: local LLM answer generation over the RAG corpus.

Retrieval is reused from Experiments 11 and 12. This experiment adds one new
stage: a small locally running instruction model receives the retrieved
passages and produces a grounded answer with citations. The extractive
answerer from Experiment 11 is still available on the same retrieved passages,
so retrieval, extraction, and generation are easy to compare.

The default model is Qwen2.5-3B-Instruct (Apache-2.0), loaded lazily with
Hugging Face Transformers. It uses about 6.3 GB of VRAM in float16.
The model is not loaded for retrieval-only work or unit tests.

Usage
-----
    # Full labelled evaluation with hybrid retrieval and the extractive baseline
    .venv/bin/python experiments/13_local_rag.py

    # Ask a single question, showing evidence and both answerers
    .venv/bin/python experiments/13_local_rag.py --ask "How many moons does Mars have?"

    # Interactive mode
    .venv/bin/python experiments/13_local_rag.py --interactive

    # BM25 only, CPU inference
    .venv/bin/python experiments/13_local_rag.py --retrieval bm25 --llm-device cpu

    # Prompt-injection / distractor safety suite
    .venv/bin/python experiments/13_local_rag.py \
        --corpus-dir data/rag_corpus_safety --output /tmp/exp13_safety.json

    # Save every evaluated answer, evidence list, citation check, and metric
    .venv/bin/python experiments/13_local_rag.py --output /tmp/exp13_eval.json

Model downloads are never performed by the test suite. This script loads the
model only when generation is requested.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from minigrad.answer_evaluation import (
    assess_answer,
    load_answer_key,
    observation_from_extractive,
    observation_from_llm,
    summarise_assessments,
)
from minigrad.embeddings import (
    DEFAULT_MODEL_NAME,
    EmbeddingModelUnavailable,
    HashingEmbeddingBackend,
    SentenceTransformerEmbeddings,
)
from minigrad.llm_answerer import (
    DEFAULT_LLM_MODEL,
    EvidencePolicy,
    LLMAnswerer,
    LLMBackendUnavailable,
    TransformersLLMBackend,
    make_evidence,
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
EVAL_KS = (1, 3, 5)

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="questions JSON (default: <corpus-dir>/questions.json)",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=None,
        help="answer key JSON (default: <corpus-dir>/answer_key.json)",
    )
    parser.add_argument(
        "--retrieval",
        choices=("bm25", "semantic", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--embedding-backend",
        choices=("sentence-transformers", "hash"),
        default="sentence-transformers",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="override the calibrated semantic abstention threshold",
    )
    parser.add_argument("--calibration-percentile", type=float, default=100.0)
    parser.add_argument("--calibration-margin", type=float, default=0.0)
    parser.add_argument(
        "--no-retrieval-fallback",
        action="store_true",
        help="fail instead of falling back to BM25 if embeddings are unavailable",
    )

    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-device", default="auto")
    parser.add_argument(
        "--llm-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-passages", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--min-evidence-passages", type=int, default=1)
    parser.add_argument(
        "--ignore-retrieval-abstention",
        action="store_true",
        help="let the LLM answer even when the retriever abstains",
    )
    parser.add_argument("--ask", action="append", default=[])
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def print_corpus_stats(documents, questions):
    print("Corpus")
    print("------")
    print(f"  documents : {len(documents)}")
    print(f"  characters: {sum(len(document.text) for document in documents)}")
    print(f"  questions : {len(questions)}")
    print(f"  answerable: {sum(question.answerable for question in questions)}")
    print(f"  unanswerable: {sum(not question.answerable for question in questions)}")
    print()


def _embedding_candidates(requested):
    if requested != "auto":
        return [requested]
    candidates = ["cpu"]
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                candidates.insert(0, "cuda:0")
        except Exception:
            pass
    return candidates


def load_embedding_backend(args):
    if args.embedding_backend == "hash":
        return HashingEmbeddingBackend(dimension=256), "cpu (hash fallback, not semantic)"

    errors = []
    for device in _embedding_candidates(args.embedding_device):
        print(f"Loading embedding model {args.embedding_model!r} on {device} ...")
        backend = SentenceTransformerEmbeddings(
            model_name=args.embedding_model,
            device=device,
            batch_size=args.embedding_batch_size,
        )
        try:
            probe = backend.encode(["embedding probe"])
            if len(probe) != 1 or not probe[0]:
                raise EmbeddingModelUnavailable("model returned an invalid probe vector")
            return backend, device
        except EmbeddingModelUnavailable as error:
            errors.append(f"{device}: {error}")
            print(f"  failed: {error}")
    raise EmbeddingModelUnavailable("; ".join(errors))


def print_embedding_info(backend, device):
    print("Embedding backend")
    print("-----------------")
    print(f"  model          : {backend.model_name}")
    print(f"  device         : {device}")
    print(f"  normalization  : {getattr(backend, 'normalize_embeddings', True)}")
    if backend.model_name.startswith("hashing-"):
        print("  WARNING: hashing is lexical, not a semantic model; only a plumbing check.")
    print()


def build_retriever(args, documents):
    bm25 = RAGPipeline.from_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
    )
    if args.retrieval == "bm25":
        return bm25, "BM25"

    backend, device = load_embedding_backend(args)
    print_embedding_info(backend, device)
    semantic = SemanticRAGPipeline.from_documents(
        documents,
        backend,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.embedding_batch_size,
        top_k=args.top_k,
        min_similarity=0.0,
    )
    if args.min_similarity is None:
        threshold, scores = calibrate_similarity_threshold(
            semantic.index,
            CALIBRATION_QUESTIONS,
            percentile=args.calibration_percentile,
            margin=args.calibration_margin,
        )
        semantic.min_similarity = threshold
        print(f"  cosine threshold: {threshold:.3f} (calibrated)")
        print(
            "  calibration top-1 similarities: "
            + ", ".join(f"{score:.3f}" for score in scores)
        )
        if backend.model_name.startswith("hashing-"):
            print("  WARNING: threshold calibration is not meaningful for hashing.")
        print()
    else:
        semantic.min_similarity = args.min_similarity
        print(f"  cosine threshold: {args.min_similarity:.3f} (manual)\n")

    if args.retrieval == "semantic":
        return semantic, "semantic"
    hybrid = HybridRAGPipeline(bm25, semantic, rrf_k=args.rrf_k)
    return hybrid, "hybrid (BM25 + semantic RRF)"


def print_retrieval_metrics(pipeline, questions, top_k):
    if not questions:
        return
    print("Retrieval metrics")
    print("-----------------")
    for k in EVAL_KS:
        metrics = evaluate_retrieval(pipeline, questions, k=k)
        print(
            f"  k={k}: recall {metrics['recall_at_k']:.3f}  "
            f"hit {metrics['hit_at_k']:.3f}  MRR {metrics['mrr_at_k']:.3f}  "
            f"abstain(unanswerable) {metrics['abstention_rate_unanswerable']:.3f}  "
            f"false abstain {metrics['false_abstention_rate']:.3f}"
        )
    print(f"  generation retrieval top-k: {top_k}")
    print()


def print_passages(passages, max_text=160):
    if not passages:
        print("  evidence: none")
        return
    print("  evidence:")
    for passage in passages:
        text = " ".join(passage.chunk.text.split())
        if len(text) > max_text:
            text = text[:max_text] + "..."
        print(
            f"    {passage.rank}. [{passage.chunk.doc_id}#{passage.chunk.chunk_index}] "
            f"score={passage.score:.4f} source={passage.chunk.source}"
        )
        print(f"       {text}")


def print_llm_info(backend, answerer, args):
    print("Language model")
    print("--------------")
    print(f"  model          : {backend.model_name}")
    print(f"  device         : {backend.device}")
    print(f"  dtype          : {backend.dtype}")
    print(f"  quantization   : none (native {backend.dtype})")
    print(f"  parameters     : {backend.parameter_count:,}")
    print(f"  load time      : {backend.load_seconds:.3f} s")
    if backend.vram_allocated_bytes is not None:
        print(f"  VRAM allocated : {backend.vram_allocated_bytes / 1e9:.2f} GB")
    print(f"  decoding       : {'greedy' if args.temperature == 0 else 'sampled'}")
    print(f"  max new tokens : {args.max_new_tokens}")
    print(f"  temperature    : {args.temperature}")
    print(f"  top_p          : {args.top_p}")
    print(f"  max passages   : {args.max_passages}")
    print(f"  context budget : {args.max_context_chars} serialised characters")
    print(
        "  evidence policy: honor retrieval abstention="
        f"{not args.ignore_retrieval_abstention}; "
        f"min passages={args.min_evidence_passages}"
    )
    print()


def _baseline_record(answer):
    citations = [
        f"{sentence.doc_id}#{sentence.chunk_index}"
        for sentence in answer.sentences
    ]
    return {
        "text": answer.text,
        "abstained": answer.abstained,
        "reason": answer.reason,
        "citations": citations,
        "sentences": [
            {
                "text": sentence.text,
                "source_id": f"{sentence.doc_id}#{sentence.chunk_index}",
                "doc_id": sentence.doc_id,
                "chunk_index": sentence.chunk_index,
                "score": sentence.score,
            }
            for sentence in answer.sentences
        ],
        "evidence": [passage.to_dict() for passage in make_evidence(answer.passages)],
    }


def _answer_record(answer):
    return {
        "text": answer.text,
        "abstained": answer.abstained,
        "reason": answer.reason,
        "citations": list(answer.citations),
        "citation_check": {
            "present": answer.citation_check.present,
            "all_exist": answer.citation_check.all_exist,
            "existing": list(answer.citation_check.existing),
            "missing": list(answer.citation_check.missing),
        },
        "evidence": [passage.to_dict() for passage in answer.evidence],
        "generation": (
            {
                **{
                    key: value
                    for key, value in answer.generation.__dict__.items()
                    if key != "vram_peak_bytes"
                },
                "vram_peak_bytes": answer.generation.vram_peak_bytes,
            }
            if answer.generation
            else None
        ),
    }


def ask_one(question, pipeline, answerer, args, retrieval_label, show=True):
    passages = pipeline.retrieve(question, top_k=args.top_k)
    retrieval_abstained, retrieval_reason = pipeline.should_abstain(
        question, passages=passages, top_k=args.top_k
    )
    baseline = None
    if not args.no_baseline:
        baseline = pipeline.answer(question, top_k=args.top_k)
    llm = answerer.answer(
        question,
        passages,
        retrieval_abstained=retrieval_abstained,
        retrieval_reason=retrieval_reason,
        retriever=retrieval_label,
    )
    if not show:
        return {
            "passages": passages,
            "retrieval_abstained": retrieval_abstained,
            "retrieval_reason": retrieval_reason,
            "baseline": baseline,
            "llm": llm,
        }

    print("=" * 78)
    print(f"Question: {question}")
    print(f"Retrieval method: {retrieval_label}")
    print_passages(passages)
    if retrieval_abstained:
        print(f"  retrieval abstention: {retrieval_reason}")
    if baseline is not None:
        print("Extractive baseline (Experiment 11)")
        if baseline.abstained:
            print(f"  [abstained] {baseline.reason}")
        else:
            print(f"  answer: {baseline.text}")
            print(
                "  sources: "
                + ", ".join(
                    f"{sentence.doc_id}#{sentence.chunk_index}"
                    for sentence in baseline.sentences
                )
            )
    print("LLM answer")
    if llm.abstained:
        print(f"  [abstained] {llm.reason}")
    else:
        print(f"  answer: {llm.text}")
        print(f"  citations: {', '.join(llm.citations) or '(none)'}")
        if llm.citation_check.missing:
            print(
                "  WARNING invalid citations: "
                + ", ".join(llm.citation_check.missing)
            )
        generation = llm.generation
        print(
            f"  generated with {generation.model_name} on {generation.device} "
            f"({generation.dtype}), {generation.latency_seconds:.3f} s, "
            f"{generation.completion_tokens} new tokens"
        )
    print()
    if args.show_prompt:
        print("Prompt: system")
        print(llm.prompt.system)
        print("Prompt: user")
        print(llm.prompt.user)
        print()
    return {
        "passages": passages,
        "retrieval_abstained": retrieval_abstained,
        "retrieval_reason": retrieval_reason,
        "baseline": baseline,
        "llm": llm,
    }


def _print_assessment_summary(label, summary):
    def fmt(value):
        return "-" if value is None else f"{value:.3f}"

    print(f"  {label}")
    print(
        f"    objective answers: {summary['n_objective']} "
        f"(manual review: {summary['n_manual_review']})"
    )
    print(
        f"    accuracy: {fmt(summary['accuracy'])}  "
        f"factual accuracy: {fmt(summary['factual_accuracy'])}"
    )
    print(
        f"    citation present: {fmt(summary['citation_presence_rate'])}  "
        f"valid ID: {fmt(summary['citation_validity_rate'])}  "
        f"supported: {fmt(summary['citation_support_rate'])}"
    )
    print(
        f"    required docs cited: {fmt(summary['required_docs_cited_rate'])}  "
        f"unanswerable abstain: "
        f"{fmt(summary['abstention_rate_unanswerable'])}  "
        f"false abstain: {fmt(summary['false_abstention_rate'])}"
    )


def run_evaluation(pipeline, answerer, questions, key, args, retrieval_label):
    if not answerer:
        return None
    print("=" * 78)
    print("Labelled answer evaluation")
    print("=" * 78)
    print(
        "Automatic correctness is a small, transparent answer-key check. "
        "Manual-review entries are excluded from accuracy aggregates."
    )
    print()

    rows = []
    llm_assessments = []
    baseline_assessments = []
    for question in questions:
        entry = key.get(question.id)
        if entry is None:
            raise SystemExit(f"answer key has no entry for {question.id}")
        result = ask_one(
            question.question,
            pipeline,
            answerer,
            args,
            retrieval_label,
            show=False,
        )
        llm_obs = observation_from_llm(result["llm"])
        llm_assessment = assess_answer(entry, llm_obs, question.answerable)
        llm_assessments.append(llm_assessment)

        baseline_record = None
        baseline_assessment = None
        if result["baseline"] is not None:
            baseline_obs = observation_from_extractive(result["baseline"])
            baseline_assessment = assess_answer(
                entry, baseline_obs, question.answerable
            )
            baseline_assessments.append(baseline_assessment)
            baseline_record = {
                "answer": _baseline_record(result["baseline"]),
                "assessment": baseline_assessment,
            }

        evidence = [
            {
                "source_id": passage.source_id,
                "doc_id": passage.doc_id,
                "chunk_index": passage.chunk_index,
                "source": passage.source,
                "score": passage.score,
                "rank": passage.rank,
                "text": passage.text,
            }
            for passage in result["llm"].evidence
        ]
        rows.append(
            {
                "id": question.id,
                "category": question.category,
                "question": question.question,
                "answerable": question.answerable,
                "retrieval_abstained": result["retrieval_abstained"],
                "retrieval_reason": result["retrieval_reason"],
                "evidence": evidence,
                "llm": _answer_record(result["llm"]),
                "llm_assessment": llm_assessment,
                "baseline": baseline_record,
            }
        )

    print("Answer-quality metrics")
    print("----------------------")
    llm_summary = summarise_assessments(llm_assessments)
    _print_assessment_summary("LLM", llm_summary)
    if baseline_assessments:
        _print_assessment_summary(
            "extractive baseline", summarise_assessments(baseline_assessments)
        )
    print()

    print("Per-question results")
    print("--------------------")
    print(
        f"  {'id':<4} {'category':<22} {'manual':<7} {'LLM':<28} "
        f"{'baseline':<10} reason"
    )
    for row in rows:
        llm_row = row["llm_assessment"]
        baseline_row = (
            row["baseline"]["assessment"] if row["baseline"] is not None else None
        )
        status = "abs" if row["llm"]["abstained"] else "ans"
        llm_text = (
            f"{status} correct={llm_row['correct']} "
            f"cites={len(row['llm']['citations'])}"
        )
        baseline_text = "-"
        if baseline_row is not None:
            baseline_text = (
                f"{'abs' if row['baseline']['answer']['abstained'] else 'ans'} "
                f"{baseline_row['correct']}"
            )
        print(
            f"  {row['id']:<4} {row['category']:<22} "
            f"{str(row['llm_assessment']['manual_review']):<7} "
            f"{llm_text:<28} {baseline_text:<10} {llm_row['reason']}"
        )
    print()

    manual_rows = [
        row for row in rows if row["llm_assessment"]["manual_review"]
    ]
    if manual_rows:
        print("Manual review entries")
        print("---------------------")
        for row in manual_rows:
            print(f"  {row['id']}: {row['question']}")
            print(f"    LLM: {row['llm']['text'] or '[abstained]'}")
            print(f"    citations: {', '.join(row['llm']['citations']) or '(none)'}")
        print()

    if args.output:
        output = {
            "experiment": 13,
            "corpus_dir": str(args.corpus_dir),
            "retrieval": retrieval_label,
            "retrieval_top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "model": {
                "name": answerer.backend.model_name,
                "device": answerer.backend.device,
                "dtype": answerer.backend.dtype,
                "load_seconds": answerer.backend.load_seconds,
                "parameter_count": answerer.backend.parameter_count,
                "vram_allocated_bytes": answerer.backend.vram_allocated_bytes,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_passages": args.max_passages,
                "max_context_chars": args.max_context_chars,
            },
            "retrieval_metrics": {
                str(k): evaluate_retrieval(pipeline, questions, k=k)
                for k in EVAL_KS
            },
            "llm_summary": llm_summary,
            "baseline_summary": (
                summarise_assessments(baseline_assessments)
                if baseline_assessments
                else None
            ),
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote {args.output}")
        print()
    return {"summary": llm_summary, "rows": rows}


def main(argv=None):
    args = parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if args.overlap < 0 or args.overlap >= args.chunk_size:
        raise SystemExit("--overlap must satisfy 0 <= overlap < chunk-size")

    documents = load_corpus(args.corpus_dir)
    if not documents:
        raise SystemExit(f"no .txt files found in {args.corpus_dir}")
    questions_path = args.questions or args.corpus_dir / "questions.json"
    questions = load_questions(questions_path) if questions_path.exists() else []
    key_path = args.answer_key or args.corpus_dir / "answer_key.json"
    key = load_answer_key(key_path) if key_path.exists() else {}
    if questions and not key:
        raise SystemExit(f"answer key not found: {key_path}")
    missing = [question.id for question in questions if question.id not in key]
    if missing:
        raise SystemExit(f"answer key is missing: {', '.join(missing)}")
    extra = sorted(set(key) - {question.id for question in questions})
    if extra:
        raise SystemExit(f"answer key has unknown ids: {', '.join(extra)}")

    print_corpus_stats(documents, questions)

    retrieval_label = args.retrieval
    try:
        pipeline, retrieval_label = build_retriever(args, documents)
    except EmbeddingModelUnavailable as error:
        if args.no_retrieval_fallback:
            raise SystemExit(f"embedding backend unavailable: {error}")
        print(f"Embedding backend unavailable: {error}")
        print("Falling back to BM25-only retrieval.\n")
        args.retrieval = "bm25"
        pipeline = RAGPipeline.from_documents(
            documents,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            top_k=args.top_k,
        )
        retrieval_label = "BM25 (fallback: embeddings unavailable)"
    print(f"Retrieval: {retrieval_label}")
    print(f"  chunks: {len(pipeline.chunks)}")
    print()
    print_retrieval_metrics(pipeline, questions, args.top_k)

    run_eval = args.evaluate or (not args.ask and not args.interactive)
    if run_eval and not questions:
        raise SystemExit("cannot evaluate: question set is empty")
    if not args.ask and not args.interactive and not run_eval:
        return 0

    if args.retrieval_only:
        for question in args.ask:
            passages = pipeline.retrieve(question, top_k=args.top_k)
            retrieval_abstained, retrieval_reason = pipeline.should_abstain(
                question, passages=passages, top_k=args.top_k
            )
            print("=" * 78)
            print(f"Question: {question}")
            print_passages(passages)
            if retrieval_abstained:
                print(f"  retrieval abstention: {retrieval_reason}")
            baseline = pipeline.answer(question, top_k=args.top_k)
            print("Extractive baseline (Experiment 11)")
            if baseline.abstained:
                print(f"  [abstained] {baseline.reason}")
            else:
                print(f"  answer: {baseline.text}")
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
                passages = pipeline.retrieve(question, top_k=args.top_k)
                retrieval_abstained, retrieval_reason = pipeline.should_abstain(
                    question, passages=passages, top_k=args.top_k
                )
                print_passages(passages)
                if retrieval_abstained:
                    print(f"  retrieval abstention: {retrieval_reason}")
                baseline = pipeline.answer(question, top_k=args.top_k)
                if baseline.abstained:
                    print(f"  [abstained] {baseline.reason}")
                else:
                    print(f"  answer: {baseline.text}")
                print()
        return 0

    backend = TransformersLLMBackend(
        model_name=args.llm_model,
        device=args.llm_device,
        dtype=args.llm_dtype,
    )
    llm_available = True
    try:
        backend.load()
    except LLMBackendUnavailable as error:
        print("Language-model backend unavailable")
        print("-----------------------------------")
        print(f"  {error}")
        print("Retrieval and the extractive baseline remain usable.")
        print()
        llm_available = False

    answerer = None
    if llm_available:
        policy = EvidencePolicy(
            honor_retrieval_abstention=not args.ignore_retrieval_abstention,
            min_passages=args.min_evidence_passages,
        )
        answerer = LLMAnswerer(
            backend,
            policy=policy,
            max_passages=args.max_passages,
            max_context_chars=args.max_context_chars,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print_llm_info(backend, answerer, args)

    for question in args.ask:
        if answerer is not None:
            ask_one(question, pipeline, answerer, args, retrieval_label)
        else:
            passages = pipeline.retrieve(question, top_k=args.top_k)
            print("=" * 78)
            print(f"Question: {question}")
            print_passages(passages)
            baseline = pipeline.answer(question, top_k=args.top_k)
            print("Extractive baseline (Experiment 11)")
            if baseline.abstained:
                print(f"  [abstained] {baseline.reason}")
            else:
                print(f"  answer: {baseline.text}")
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
            if answerer is not None:
                ask_one(question, pipeline, answerer, args, retrieval_label)
            else:
                baseline = pipeline.answer(question, top_k=args.top_k)
                if baseline.abstained:
                    print(f"  [abstained] {baseline.reason}")
                else:
                    print(f"  answer: {baseline.text}")

    if run_eval:
        if answerer is None:
            print("Skipping LLM evaluation because the model is unavailable.")
            print("Running extraction-only evaluation.")
            extractive_rows = []
            for question in questions:
                entry = key[question.id]
                answer = pipeline.answer(question.question, top_k=args.top_k)
                observation = observation_from_extractive(answer)
                extractive_rows.append(
                    assess_answer(entry, observation, question.answerable)
                )
            summary = summarise_assessments(extractive_rows)
            print("Extractive-only summary")
            _print_assessment_summary("extractive", summary)
            return 2
        run_evaluation(pipeline, answerer, questions, key, args, retrieval_label)

    if answerer is not None and answerer.backend.vram_allocated_bytes is not None:
        print(
            f"Model held {answerer.backend.vram_allocated_bytes / 1e9:.2f} GB "
            f"of VRAM after loading."
        )
    return 0 if llm_available else 2


if __name__ == "__main__":
    raise SystemExit(main())
