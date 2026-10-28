# Solar System RAG Corpus

A small, hand-written corpus for Experiment 11 (retrieval-augmented generation).
Each `.txt` file is one document. The file stem is the document id
(`doc_id`), so `jupiter.txt` has `doc_id = "jupiter"`.

The documents are plain ASCII with numbers written as digits. They are
deliberately short and self-contained so that retrieval behaviour is easy to
inspect.

## Files

| File | Contents |
| --- | --- |
| `00_overview.txt` | Formation of the Solar System, planet order, planet classes, asteroid belt, the Sun |
| `mercury.txt` | Mercury: size, orbit, day/year, temperature, moons, Caloris Basin |
| `venus.txt` | Venus: size, orbit, retrograde rotation, atmosphere, temperature, moons |
| `earth.txt` | Earth: size, orbit, day/year, moon, oceans, atmosphere, temperature |
| `mars.txt` | Mars: size, orbit, day/year, moons, atmosphere, Olympus Mons, Valles Marineris |
| `jupiter.txt` | Jupiter: size, orbit, mass, rotation, composition, Great Red Spot, moons |
| `saturn.txt` | Saturn: size, orbit, rotation, density, rings, moons, Titan |
| `questions.json` | Evaluation questions with document-level relevance labels |

## Relevance definition

Relevance is defined at the **document** level. A retrieved chunk is relevant
when its `doc_id` appears in the question's `relevant_docs` list. This avoids
labelling a chunk as irrelevant merely because an answer sentence straddles a
chunk boundary.

Unanswerable questions have an empty `relevant_docs` list. They are excluded
from recall and MRR and are reported through the abstention rate instead.

## Corpus limitations

- Two documents concern one planet each, but no document is a complete
  reference; facts were chosen to avoid duplicating the facts that the
  questions ask about.
- Some questions need facts from two documents (multi-document questions) or
  numerical comparison. A lexical retriever can find the passages without
  understanding the comparison, and the extractive answerer may return only
  one side of it.
- `q14` is a deliberate paraphrase case. The words in the question do not
  match the wording of the corpus, so a BM25 retriever is expected to fail.
- `q12` and `q13` abstain, but the abstention heuristic is not a general
  detector of unsupported questions. A near-miss question such as "How many
  moons does Uranus have?" shares both `moons` and `uranus` (the latter only
  in the planet list in `00_overview.txt`) with retrieved text, so the
  content-term coverage threshold is satisfied and the system answers with a
  generic sentence instead of abstaining. This is a deliberate demonstration
  that lexical scores measure word overlap, not whether the requested fact is
  present.
- Chunks are fixed character windows, so they can start or end mid-word.
  Retrieval and answer extraction both tolerate this, but evidence previews
  can show partial words at chunk boundaries.
