import json
from pathlib import Path

import pytest

from minigrad.chunking import Document
from minigrad.rag import (
    Answer,
    ExtractiveAnswerer,
    RAGPipeline,
    evaluate_retrieval,
    load_corpus,
    load_questions,
)

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "rag_corpus"


@pytest.fixture(scope="module")
def documents():
    return load_corpus(CORPUS_DIR)


@pytest.fixture(scope="module")
def questions():
    return load_questions(CORPUS_DIR / "questions.json")


@pytest.fixture(scope="module")
def pipeline(documents):
    return RAGPipeline.from_documents(documents, chunk_size=400, overlap=80)


def test_load_corpus_sorts_files_and_strips_text(tmp_path):
    (tmp_path / "b.txt").write_text("  second  \n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("no", encoding="utf-8")

    loaded = load_corpus(tmp_path)
    assert [doc.doc_id for doc in loaded] == ["a", "b"]
    assert [doc.source for doc in loaded] == ["a.txt", "b.txt"]
    assert [doc.text for doc in loaded] == ["first", "second"]


def test_load_questions_reads_labels(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "category": "single",
                        "question": "What?",
                        "relevant_docs": ["a"],
                        "answerable": True,
                        "notes": "n",
                    },
                    {
                        "id": "q2",
                        "category": "none",
                        "question": "Pluto?",
                        "relevant_docs": [],
                        "answerable": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_questions(path)
    assert loaded[0].relevant_docs == ("a",)
    assert loaded[0].answerable is True
    assert loaded[0].notes == "n"
    assert loaded[1].relevant_docs == ()
    assert loaded[1].answerable is False


def test_pipeline_from_documents_records_chunking_configuration(documents):
    pipeline = RAGPipeline.from_documents(documents, chunk_size=400, overlap=80)
    assert pipeline.chunk_size == 400
    assert pipeline.overlap == 80
    assert len(pipeline.chunks) == len(pipeline.retriever.documents)
    assert all(chunk.doc_id in {doc.doc_id for doc in documents} for chunk in pipeline.chunks)


def test_retrieve_returns_ranked_passages_with_attribution(pipeline):
    passages = pipeline.retrieve("What is the diameter of Jupiter?", top_k=3)
    assert passages
    assert [passage.rank for passage in passages] == [1, 2, 3]
    assert passages[0].chunk.doc_id == "jupiter"
    assert passages[0].score >= passages[1].score >= passages[2].score
    assert all(passage.chunk.text for passage in passages)


def test_answers_single_fact_question_with_source(pipeline):
    answer = pipeline.answer("What is the diameter of Jupiter?")
    assert not answer.abstained
    assert answer.text
    assert answer.sentences
    assert answer.sentences[0].doc_id == "jupiter"
    assert "diameter" in answer.text.lower()


def test_answerer_does_not_quote_the_opposite_planet(pipeline):
    answer = pipeline.answer("Which planet has the shortest day?")
    assert not answer.abstained
    assert "shortest day" in answer.text.lower()
    assert "longest day" not in answer.text.lower()


def test_question_quantifiers_are_stopwords(pipeline):
    assert pipeline.content_terms("How many moons does Mars have?") == ["moons", "mars"]
    assert pipeline.content_terms("How much does Jupiter weigh?") == ["jupiter", "weigh"]


def test_answerer_drops_sentences_with_weaker_term_coverage():
    document = Document(
        doc_id="a",
        source="a.txt",
        text=(
            "The planet Jupiter has the shortest day, only 9.9 hours. "
            "Venus has a longer day than Jupiter."
        ),
    )
    pipeline = RAGPipeline.from_documents([document], chunk_size=400, overlap=0)
    answer = pipeline.answer("Which planet has the shortest day?")
    assert "shortest day" in answer.text.lower()
    assert "venus" not in answer.text.lower()
    assert len(answer.sentences) == 1


def test_answerer_quotes_equally_strong_sentences_from_two_documents():
    documents = [
        Document(doc_id="a", source="a.txt", text="Alpha has two moons."),
        Document(doc_id="b", source="b.txt", text="Beta has five moons."),
    ]
    pipeline = RAGPipeline.from_documents(documents, chunk_size=400, overlap=0)
    answer = pipeline.answer("How many moons are there?")
    assert {sentence.doc_id for sentence in answer.sentences} == {"a", "b"}
    assert {sentence.source for sentence in answer.sentences} == {"a.txt", "b.txt"}


def test_abstains_on_empty_question(pipeline):
    abstained, reason = pipeline.should_abstain("   ")
    assert abstained
    assert "content terms" in reason

    answer = pipeline.answer("")
    assert answer.abstained


def test_abstains_when_no_passage_matches(pipeline):
    abstained, reason = pipeline.should_abstain("quasar luminosity function")
    assert abstained
    assert "no matching passages" in reason


def test_abstains_on_low_content_term_coverage(pipeline):
    abstained, reason = pipeline.should_abstain(
        "What is the atmosphere like on Pluto?"
    )
    assert abstained
    assert "coverage" in reason


def test_abstains_when_top_score_below_min_score(documents):
    pipeline = RAGPipeline.from_documents(
        documents, chunk_size=400, overlap=80, min_score=1000.0
    )
    abstained, reason = pipeline.should_abstain("What is the diameter of Jupiter?")
    assert abstained
    assert reason == "top score below min_score"


def test_answerer_limits_sentences_and_deduplicates():
    chunks = [
        Document(doc_id="a", source="a.txt", text="Alpha is red. Beta is blue."),
    ]
    pipeline = RAGPipeline.from_documents(chunks, chunk_size=400, overlap=0)
    answer = pipeline.answer("What colour is alpha beta?", max_sentences=1)
    assert not answer.abstained
    assert len(answer.sentences) == 1

    answerer = ExtractiveAnswerer(pipeline.retriever.idf, max_sentences=5)
    from minigrad.chunking import chunk_documents

    passages = pipeline.retrieve("alpha beta")
    all_sentences = answerer.answer("alpha beta", passages)
    texts = [sentence.text for sentence in all_sentences.sentences]
    assert len(texts) == len(set(texts))


def test_answerer_abstains_when_no_sentence_matches():
    pipeline = RAGPipeline.from_documents(
        [Document(doc_id="a", source="a.txt", text="Alpha is red.")],
        chunk_size=400,
        overlap=0,
    )
    passages = pipeline.retrieve("red")
    answerer = ExtractiveAnswerer(pipeline.retriever.idf)
    answer = answerer.answer("unrelated query", passages)
    assert answer.abstained
    assert "no retrieved sentence" in answer.reason


def test_answer_is_immutable_dataclass(pipeline):
    answer = pipeline.answer("What is the diameter of Jupiter?")
    assert isinstance(answer, Answer)
    with pytest.raises(Exception):
        answer.text = "changed"


def test_evaluate_retrieval_reports_document_level_metrics(tmp_path):
    (tmp_path / "alpha.txt").write_text("Alpha is red. Alpha is round.", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("Beta is blue.", encoding="utf-8")
    (tmp_path / "gamma.txt").write_text("Gamma is green.", encoding="utf-8")
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "category": "single",
                        "question": "What is Alpha?",
                        "relevant_docs": ["alpha"],
                        "answerable": True,
                    },
                    {
                        "id": "q2",
                        "category": "multi",
                        "question": "Alpha and beta",
                        "relevant_docs": ["alpha", "beta"],
                        "answerable": True,
                    },
                    {
                        "id": "q3",
                        "category": "unanswerable",
                        "question": "What is delta?",
                        "relevant_docs": [],
                        "answerable": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    pipeline = RAGPipeline.from_documents(
        load_corpus(tmp_path), chunk_size=400, overlap=0
    )
    questions = load_questions(questions_path)

    at_three = evaluate_retrieval(pipeline, questions, k=3)
    assert at_three["n_answerable"] == 2
    assert at_three["n_unanswerable"] == 1
    assert at_three["recall_at_k"] == pytest.approx(1.0)
    assert at_three["hit_at_k"] == pytest.approx(1.0)
    assert at_three["mrr_at_k"] == pytest.approx(1.0)
    assert at_three["abstention_rate_unanswerable"] == pytest.approx(1.0)
    assert at_three["false_abstention_rate"] == pytest.approx(0.0)

    at_one = evaluate_retrieval(pipeline, questions, k=1)
    assert at_one["recall_at_k"] == pytest.approx(0.75)
    assert at_one["hit_at_k"] == pytest.approx(0.5)
    assert at_one["rows"][1]["retrieved_docs"] == ["alpha"]

    with pytest.raises(ValueError):
        evaluate_retrieval(pipeline, questions, k=0)


def test_real_corpus_evaluation_is_stable(pipeline, questions):
    metrics = evaluate_retrieval(pipeline, questions, k=3)
    assert metrics["n_answerable"] == 12
    assert metrics["n_unanswerable"] == 2
    assert metrics["recall_at_k"] == pytest.approx(0.875)
    assert metrics["hit_at_k"] == pytest.approx(0.8333333333333334)
    assert metrics["abstention_rate_unanswerable"] == pytest.approx(1.0)
    assert metrics["false_abstention_rate"] == pytest.approx(0.08333333333333333)
    by_id = {row["id"]: row for row in metrics["rows"]}
    assert by_id["q14"]["abstained"] is True
    assert by_id["q01"]["retrieved_docs"][0] == "jupiter"


@pytest.mark.parametrize("top_k,min_coverage", [(0, 0.5), (3, -0.1), (3, 1.1)])
def test_invalid_pipeline_parameters(documents, top_k, min_coverage):
    with pytest.raises(ValueError):
        RAGPipeline.from_documents(
            documents, top_k=top_k, min_coverage=min_coverage
        )
