"""Dense retrieval and one generation call on the existing Chroma index.

Retrieval is cosine nearest neighbors in ``data/chroma``. Generation is one
call through ``src.generate``. Scoring is the deterministic checks in
``eval_generation.py`` in this folder. Answers and evals are written here.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eval_generation import Generation, parse_generation, score_answers
from src import config
from src.embed import embed_texts
from src.generate import generate
from src.vectordb import SearchHit, connect, search

ANSWERS_PATH = HERE / "answers.json"
EVALS_PATH = HERE / "evals.json"
GOLD_PATH = ROOT / "data" / "gold" / "policy_rag_golden.json"
K = 5
ATTEMPTS = 3


def retrieve(client, query: str, k: int = K) -> list[SearchHit]:
    """Return the ``k`` nearest chunks for ``query`` from the open Chroma client."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not query.strip():
        return []
    vector = embed_texts([query])[0]
    return search(client, vector, k=k)


def generation_prompt(question: str, hits: list[SearchHit]) -> str:
    """Ask for an answer, the sections used, and a confidence."""
    blocks: list[str] = []
    for hit in hits:
        blocks.append(
            "\n".join(
                (
                    f"section_id: {hit.parent_id}",
                    f"section_name: {hit.section_name}",
                    "body:",
                    hit.body,
                )
            )
        )
    chunks = "\n\n".join(blocks) if blocks else "(no chunks retrieved)"
    return (
        "Answer the question using only the chunks. "
        "If they do not contain the answer, say the documents do not say.\n"
        "Reply with one JSON object and no other text:\n"
        '{"answer": "...", "sections": [{"section_id": "...", "section_name": "..."}], '
        '"confidence": 0.0}\n'
        "confidence is a number from 0 to 1.\n\n"
        f"Question: {question}\n\n"
        f"Chunks:\n{chunks}\n"
    )


def answer(question: str, hits: list[SearchHit]) -> str:
    """Generate one reply for ``question`` from ``hits``."""
    return generate(generation_prompt(question, hits))


def run() -> None:
    """Write answers.json, score it with src.eval_generation, and write evals.json."""
    dataset = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    items: list[dict[str, object]] = []
    client = connect()
    try:
        for item in dataset["items"]:
            question = str(item["question"])
            hits = retrieve(client, question)
            text, latency, error = _generate_once(question, hits)
            parsed = parse_generation(text) if text is not None else Generation("", (), None)
            items.append(
                {
                    "id": item["id"],
                    "question": question,
                    "raw": text,
                    "answer": parsed.answer,
                    "sections": [
                        {"section_id": section.section_id, "section_name": section.section_name}
                        for section in parsed.sections
                    ],
                    "confidence": parsed.confidence,
                    "latency_seconds": latency,
                    "retrieved": [_stored_hit(hit) for hit in hits],
                    "error": error,
                }
            )
            _write_json(
                ANSWERS_PATH,
                {
                    "model": config.MODEL,
                    "model_id": config.MODELS[config.MODEL],
                    "retrieval": "chroma",
                    "k": K,
                    "items": items,
                },
            )
            if error is None:
                print(f"{item['id']} answer latency_seconds={latency:.4f}", flush=True)
            else:
                print(f"{item['id']} answer failed: {error}", flush=True)
    finally:
        client.close()

    print(f"answers: {ANSWERS_PATH}", flush=True)
    result = score_answers(ANSWERS_PATH, EVALS_PATH)
    print(f"evals: {EVALS_PATH}", flush=True)
    if result is None:
        print("no scored questions", flush=True)
        return
    print(f"questions: {result.questions}", flush=True)
    print(f"required_phrases: {result.required_phrases:.4f}", flush=True)
    print(f"cited_section: {result.cited_section:.4f}", flush=True)
    print(f"hard_checks_passed: {result.hard_checks_passed:.4f}", flush=True)


def _call_with_retry(fn):
    """Call ``fn`` up to ``ATTEMPTS`` times. The last exception is raised."""
    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if attempt == ATTEMPTS:
                raise
    raise last


def _generate_once(question: str, hits: list[SearchHit]) -> tuple[str | None, float, str | None]:
    """Generate one answer, retrying the model call. Returns text, latency, error."""
    started = time.perf_counter()
    try:
        text = _call_with_retry(lambda: answer(question, hits))
    except Exception as exc:
        return None, time.perf_counter() - started, str(exc)
    return text, time.perf_counter() - started, None


def _stored_hit(hit: SearchHit) -> dict[str, object]:
    return {
        "chunk_id": hit.chunk_id,
        "section_id": hit.parent_id,
        "section_name": hit.section_name,
        "filename": hit.filename,
        "body": hit.body,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
