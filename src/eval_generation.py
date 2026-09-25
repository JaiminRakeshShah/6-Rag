"""Generation evals for the golden question set.

Answers are generated first and written to ``data/generation/answers.json``.
Jev Choice then scores faithfulness, groundedness, and correctness into
``data/generation/evals_jev.json``. Recorded gpt-oss judge scores stay in
``data/generation/evals_gptoss.json`` and ``data/generation/evals.json``.
This module does not call gpt-oss.

The prompt asks the generator to name the sections it used, with the same
``section_id`` and ``section_name`` fields the gold file stores, and to give
a confidence from 0 to 1. Generation latency is the wait for that call only.

Deterministic checks remain: phrase coverage, citation recall, distractor
citation, token F1 against ``expected_answer``, and abstain contamination.
The two pass/fail checks are that every ``answer_must_include`` phrase is
present, and that a cited section matches a gold supporting section.

The generator call and the Jev call each retry up to ``ATTEMPTS`` times.
A question that still fails is stored with ``error`` set, and the run
continues.
"""

import json
import os
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src import config
from src.generate import generate, generation_prompt
from src.retrieve import INITIAL_K, TYPESAFE_URL, HybridHit, hybrid_search
from src.vectordb import connect

K = INITIAL_K
ATTEMPTS = 3
_DECLINE_TOKENS = frozenset(
    {
        "a",
        "an",
        "the",
        "documents",
        "document",
        "do",
        "does",
        "did",
        "not",
        "no",
        "say",
        "says",
        "contain",
        "contains",
        "answer",
        "they",
        "it",
        "this",
    }
)
ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "gold" / "policy_rag_golden.json"
ANSWERS_PATH = ROOT / "data" / "generation" / "answers.json"
JEV_EVALS_PATH = ROOT / "data" / "generation" / "evals_jev.json"
_JUDGE_NAMES = ("faithfulness", "groundedness", "correctness")
_JEV_MODEL = "jev-latest"
_JEV_PASS = "meets"
_JEV_CRITERIA: dict[str, tuple[str, dict[str, str]]] = {
    "faithfulness": (
        "Do the factual claims in the answer agree with the retrieval context?",
        {
            "meets": "Every factual claim in the answer agrees with the retrieval context",
            "misses": "A factual claim in the answer contradicts the retrieval context",
        },
    ),
    "groundedness": (
        "Is every fact in the answer stated in the retrieval context?",
        {
            "meets": "Every fact in the answer is stated in the retrieval context",
            "misses": "The answer states a fact the retrieval context does not contain",
        },
    ),
    "correctness": (
        "Does the answer include the facts required by the expected output?",
        {
            "meets": "The answer includes every fact required by the expected output",
            "misses": "A fact required by the expected output is missing, or the answer states a figure or a name when the expected output says the documents do not contain the answer",
        },
    ),
}


@dataclass(frozen=True)
class CitedSection:
    """One section the generator says it used."""

    section_id: str
    section_name: str


@dataclass(frozen=True)
class Generation:
    """Parsed generator output."""

    answer: str
    sections: tuple[CitedSection, ...]
    confidence: float | None


@dataclass(frozen=True)
class QuestionScore:
    """Judge scores, the model's confidence, latency, and the deterministic scores.

    ``token_f1``, ``phrase_coverage``, and ``citation_recall`` are ``None``
    when that score does not apply. ``abstain_contamination`` is ``None`` on
    a scored question and a bool on an abstain question.
    """

    item_id: str
    faithfulness: float
    groundedness: float
    correctness: float
    confidence: float | None
    latency_seconds: float
    required_phrases: bool
    cited_section: bool
    token_f1: float | None = None
    phrase_coverage: float | None = None
    citation_recall: float | None = None
    distractor_citation: float = 0.0
    abstain_contamination: bool | None = None


@dataclass(frozen=True)
class GenerationEval:
    """Means over the golden questions. Rates are the share that passed.

    A deterministic mean uses only the questions where that score applies.
    ``confidence_when_passed`` and ``confidence_when_failed`` split the
    reported confidence by ``hard_checks_passed``.
    """

    faithfulness: float
    groundedness: float
    correctness: float
    confidence: float
    latency_seconds: float
    required_phrases: float
    cited_section: float
    questions: int
    confidence_questions: int
    rows: tuple[QuestionScore, ...]
    token_f1: float = 0.0
    phrase_coverage: float = 0.0
    citation_recall: float = 0.0
    distractor_citation: float = 0.0
    abstain_contamination: float = 0.0
    confidence_when_passed: float = 0.0
    confidence_when_failed: float = 0.0
    token_f1_questions: int = 0
    phrase_coverage_questions: int = 0
    citation_recall_questions: int = 0
    abstain_questions: int = 0
    confidence_passed_questions: int = 0
    confidence_failed_questions: int = 0


def parse_generation(text: str) -> Generation:
    """Read the JSON object. A non-JSON reply is the answer, with no citation."""
    payload = _json_object(text)
    if payload is None:
        return Generation(answer=text.strip(), sections=(), confidence=None)
    answer = payload.get("answer")
    if not isinstance(answer, str):
        answer = text.strip()
    sections: list[CitedSection] = []
    raw_sections = payload.get("sections")
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, Mapping):
                continue
            section_id = item.get("section_id")
            if not isinstance(section_id, str):
                chunk_id = item.get("chunk_id")
                section_id = chunk_id if isinstance(chunk_id, str) else ""
            section_name = item.get("section_name")
            if not isinstance(section_name, str):
                section_name = ""
            if section_id.strip() or section_name.strip():
                sections.append(CitedSection(section_id.strip(), section_name.strip()))
    return Generation(answer=answer, sections=tuple(sections), confidence=_confidence(payload.get("confidence")))


def required_phrases(answer: str, item: Mapping[str, object]) -> bool:
    """Gold phrase check. Comparison is case-insensitive.

    A scored question must contain every ``answer_must_include`` phrase.
    An abstain question must not repeat a distractor evidence quote.
    """
    text = answer.casefold()
    if item["abstain"] is True:
        return all(quote.casefold() not in text for quote in _evidence_quotes(item.get("distractor_sections", [])))
    phrases = item["answer_must_include"]
    if not isinstance(phrases, list):
        raise ValueError("answer_must_include must be a list")
    return all(isinstance(phrase, str) and phrase.casefold() in text for phrase in phrases)


def cited_section(parsed: Generation, item: Mapping[str, object]) -> bool:
    """True when a cited section matches a gold supporting section.

    ``section_id`` matches the gold id, and a ``chunk_id`` matches the
    section it belongs to. A section name matches only when no distractor
    uses that same name. An abstain question matches when nothing is cited.
    """
    if item["abstain"] is True:
        return not any(section.section_id or section.section_name for section in parsed.sections)
    supporting = item["supporting_sections"]
    if not isinstance(supporting, list):
        raise ValueError("supporting_sections must be a list")
    distractor_names = {
        section["section_name"].casefold()
        for section in _sections(item.get("distractor_sections", []))
        if isinstance(section.get("section_name"), str)
    }
    for section in _sections(supporting):
        section_id = section["section_id"]
        if isinstance(section_id, str) and _mentions_id(parsed, section_id):
            return True
        name = section.get("section_name")
        if not isinstance(name, str) or name.casefold() in distractor_names:
            continue
        if name.casefold() in parsed.answer.casefold():
            return True
        if any(cited.section_name.casefold() == name.casefold() for cited in parsed.sections):
            return True
    return False


def token_f1(answer: str, expected: str) -> float:
    """Word overlap of ``answer`` and ``expected``, from 0 to 1.

    Words are case-insensitive runs of letters and digits, so ``1.0`` and
    ``100%`` become ``1``, ``0`` and ``100``. The score is the harmonic mean
    of precision and recall over those words. Two empty strings score 1.
    """
    got = Counter(_tokens(answer))
    gold = Counter(_tokens(expected))
    if not got and not gold:
        return 1.0
    if not got or not gold:
        return 0.0
    overlap = sum((got & gold).values())
    precision = overlap / sum(got.values())
    recall = overlap / sum(gold.values())
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def phrase_coverage(answer: str, item: Mapping[str, object]) -> float | None:
    """Share of ``answer_must_include`` phrases present in ``answer``.

    Comparison is a case-insensitive substring, the same rule as
    ``required_phrases``. An abstain question has no gold phrases, so it
    returns ``None`` and stays out of the mean.
    """
    if item["abstain"] is True:
        return None
    phrases = item["answer_must_include"]
    if not isinstance(phrases, list):
        raise ValueError("answer_must_include must be a list")
    if not phrases:
        return 1.0
    text = answer.casefold()
    found = sum(1 for phrase in phrases if isinstance(phrase, str) and phrase.casefold() in text)
    return found / len(phrases)


def citation_recall(parsed: Generation, item: Mapping[str, object]) -> float | None:
    """Share of gold supporting section ids named by the answer.

    A ``chunk_id`` counts for the section it belongs to. An abstain question
    has no supporting section, so it returns ``None``.
    """
    if item["abstain"] is True:
        return None
    supporting = item["supporting_sections"]
    if not isinstance(supporting, list):
        raise ValueError("supporting_sections must be a list")
    sections = _sections(supporting)
    if not sections:
        return None
    found = 0
    for section in sections:
        section_id = section["section_id"]
        if isinstance(section_id, str) and _mentions_id(parsed, section_id):
            found += 1
    return found / len(sections)


def distractor_citation(parsed: Generation, item: Mapping[str, object]) -> float:
    """Share of cited sections whose id matches a gold distractor.

    An answer that cites nothing scores 0. A name with no ``section_id``
    does not count, because a distractor can share that name with the gold
    section.
    """
    cited = [section for section in parsed.sections if section.section_id]
    if not cited:
        return 0.0
    distractor_ids = [
        section["section_id"]
        for section in _sections(item.get("distractor_sections", []))
        if isinstance(section.get("section_id"), str)
    ]
    if not distractor_ids:
        return 0.0
    hits = sum(
        1
        for section in cited
        if any(_id_in(section.section_id, distractor_id) for distractor_id in distractor_ids)
    )
    return hits / len(cited)


def abstain_contamination(answer: str, item: Mapping[str, object]) -> bool | None:
    """True when an abstain answer adds a number or a name.

    A number is a word containing a digit. A name is a capitalized word.
    Words that already appear in the question, and the small decline
    vocabulary (``the``, ``documents``, ``not``, ``say``), do not count.
    A scored question returns ``None``.
    """
    if item["abstain"] is not True:
        return None
    question_tokens = set(_tokens(str(item.get("question", ""))))
    for token in _tokens(answer):
        if any(character.isdigit() for character in token) and token not in question_tokens:
            return True
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", answer):
        folded = _tokens(raw)
        if not folded:
            continue
        token = folded[0]
        if token in question_tokens or token in _DECLINE_TOKENS:
            continue
        if raw[0].isupper():
            return True
    return False


def hard_checks_passed(row: QuestionScore) -> bool:
    """True when every deterministic pass/fail check on ``row`` passed.

    Token F1 is a similarity, not a gate. Phrase coverage and citation
    recall fail when they apply and are below 1. A cited distractor fails.
    An abstain answer fails when it adds a figure or a name.
    """
    if not row.required_phrases or not row.cited_section:
        return False
    if row.phrase_coverage is not None and row.phrase_coverage < 1.0:
        return False
    if row.citation_recall is not None and row.citation_recall < 1.0:
        return False
    if row.distractor_citation > 0.0:
        return False
    return row.abstain_contamination is not True


def expected_output(item: Mapping[str, object]) -> str:
    """Reference text the correctness judge compares against."""
    if item["abstain"] is True:
        return (
            "The documents do not contain the answer. "
            "A correct reply declines and does not invent a figure or a name."
        )
    phrases = item["answer_must_include"]
    if not isinstance(phrases, list):
        raise ValueError("answer_must_include must be a list")
    included = ", ".join(str(phrase) for phrase in phrases)
    return f"{item['expected_answer']}. The answer must include: {included}."


def actual_output(parsed: Generation) -> str:
    """Answer plus the sections the model named, for the judge."""
    if not parsed.sections:
        return parsed.answer
    lines = [parsed.answer, "", "Sections used:"]
    for section in parsed.sections:
        lines.append(f"- {section.section_id}: {section.section_name}")
    return "\n".join(lines)


def context_block(hit: HybridHit) -> str:
    """One retrieved chunk, labeled the way the gold sections are labeled."""
    return (
        f"chunk_id: {hit.chunk_id}\n"
        f"section_id: {hit.parent_id}\n"
        f"section_name: {hit.section_name}\n"
        f"{hit.body}"
    )


class JevScoreJudge(BaseMetric):
    """Three Choice questions in one Jev request.

    Each question has two options, ``meets`` and ``misses``. ``passed`` is true
    when Jev chooses ``meets``.
    """

    _required_params = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]
    async_mode = False
    evaluation_model = _JEV_MODEL

    def __init__(self) -> None:
        self.threshold = 0.5
        self.stored_result: dict[str, object] = {}

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        del args, kwargs
        result = _jev_score_once(test_case)
        self.stored_result = result
        self.input_tokens = _optional_int(result.get("prompt_tokens"))
        self.output_tokens = _optional_int(result.get("generation_tokens"))
        answers = result["answers"]
        if not isinstance(answers, Mapping):
            raise ValueError("Jev choice result has no answers")
        passes = [1.0 if answers[name]["passed"] is True else 0.0 for name in _JUDGE_NAMES]
        self.score = sum(passes) / len(passes)
        self.reason = None
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        return self.measure(test_case, *args, **kwargs)


def jev_score_payload(test_case: LLMTestCase) -> dict[str, object]:
    """One System One request. The three Choice questions share the test case."""
    questions: dict[str, object] = {}
    for name, (instructions, criteria) in _JEV_CRITERIA.items():
        questions[name] = {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
    return {
        "model": _JEV_MODEL,
        "state": {
            "question": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "retrieval_context": test_case.retrieval_context,
        },
        "questions": questions,
    }


def measure_jev(test_case: LLMTestCase) -> dict[str, object]:
    """Score one test case with Jev. Retries the request, then raises."""
    metric = JevScoreJudge()

    def once() -> dict[str, object]:
        metric.measure(test_case)
        return metric.stored_result

    return call_with_retry(once)


def _jev_score_once(test_case: LLMTestCase) -> dict[str, object]:
    """Post the Choice request and read which option Jev picked."""
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY is required to score with Jev")
    started = time.perf_counter()
    response = httpx.post(
        TYPESAFE_URL,
        json=jev_score_payload(test_case),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    )
    latency_seconds = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage") or {}
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise ValueError("Jev choice response has no answers")
    answers: dict[str, object] = {}
    for name in _JEV_CRITERIA:
        answer = raw_answers.get(name)
        if not isinstance(answer, Mapping) or answer.get("type") != "choice":
            raise ValueError(f"Jev choice response has no {name} choice")
        choice = answer.get("choice")
        if not isinstance(choice, str):
            raise ValueError(f"Jev {name} choice is missing")
        probabilities = answer.get("probabilities") or {}
        if not isinstance(probabilities, Mapping):
            raise ValueError(f"Jev {name} choice has no probabilities")
        confidence = answer.get("confidence")
        answers[name] = {
            "choice": choice,
            "passed": choice == _JEV_PASS,
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
            "probabilities": {str(key): float(value) for key, value in probabilities.items()},
        }
    return {
        "model": body.get("model") or _JEV_MODEL,
        "latency_seconds": latency_seconds,
        "prompt_tokens": usage.get("input_tokens"),
        "generation_tokens": usage.get("output_tokens"),
        "answers": answers,
    }


def call_with_retry(fn):
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


def generate_answers(path: Path = ANSWERS_PATH, k: int = K) -> Path:
    """Retrieve and generate every golden answer, then write ``path``."""
    dataset = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    items: list[dict[str, object]] = []
    client = connect()
    try:
        for item in dataset["items"]:
            hits = hybrid_search(client, str(item["question"]), k=k)
            text, latency, error = _generate_once(item["question"], hits)
            parsed = parse_generation(text) if text is not None else Generation("", (), None)
            items.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "raw": text,
                    "answer": parsed.answer,
                    "sections": [
                        {"section_id": section.section_id, "section_name": section.section_name}
                        for section in parsed.sections
                    ],
                    "confidence": parsed.confidence,
                    "latency_seconds": latency,
                    "retrieval_jev": _retrieval_jev(hybrid_search),
                    "retrieved": [_stored_hit(hit) for hit in hits],
                    "error": error,
                }
            )
            _write_json(
                path,
                {
                    "model": config.MODEL,
                    "model_id": config.MODELS[config.MODEL],
                    "items": items,
                },
            )
            if error is None:
                print(f"{item['id']} answer latency_seconds={latency:.4f}")
            else:
                print(f"{item['id']} answer failed: {error}")
    finally:
        client.close()
    return path


def score_answers(
    answers_path: Path = ANSWERS_PATH,
    jev_path: Path = JEV_EVALS_PATH,
) -> Path:
    """Judge the stored answers with Jev Choice. gpt-oss is not called."""
    stored = json.loads(answers_path.read_text(encoding="utf-8"))
    dataset = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    gold = {item["id"]: item for item in dataset["items"]}
    jev_items: list[dict[str, object]] = []
    for record in stored["items"]:
        item = gold[record["id"]]
        if record.get("error"):
            result: dict[str, object] = {"error": record["error"]}
        else:
            hits = [_hit_from_stored(hit) for hit in record["retrieved"]]
            parsed = parse_generation(str(record["raw"]))
            test_case = LLMTestCase(
                input=str(item["question"]),
                actual_output=actual_output(parsed),
                expected_output=expected_output(item),
                retrieval_context=[context_block(hit) for hit in hits],
            )
            try:
                result = measure_jev(test_case)
            except Exception as exc:
                result = {"error": str(exc)}
        row = _jev_score_row({"id": record["id"], "jev": result})
        jev_items.append(row)
        _write_json(jev_path, {"judge": _JEV_MODEL, "items": jev_items})
        _print_jev({"id": record["id"], "jev": result})
    return jev_path


def evaluate(rows: Sequence[QuestionScore]) -> GenerationEval:
    """Average the per-question scores. Confidence uses the rows that reported one."""
    if not rows:
        raise ValueError("no scored questions")
    count = len(rows)
    confidences = [row.confidence for row in rows if row.confidence is not None]
    token_f1_scores = [row.token_f1 for row in rows if row.token_f1 is not None]
    phrase_scores = [row.phrase_coverage for row in rows if row.phrase_coverage is not None]
    citation_scores = [row.citation_recall for row in rows if row.citation_recall is not None]
    abstain_flags = [row.abstain_contamination for row in rows if row.abstain_contamination is not None]
    passed_confidence = [
        row.confidence for row in rows if row.confidence is not None and hard_checks_passed(row)
    ]
    failed_confidence = [
        row.confidence for row in rows if row.confidence is not None and not hard_checks_passed(row)
    ]
    token_mean, token_count = _average(token_f1_scores)
    phrase_mean, phrase_count = _average(phrase_scores)
    citation_mean, citation_count = _average(citation_scores)
    abstain_mean, abstain_count = _average([float(flag) for flag in abstain_flags])
    passed_mean, passed_count = _average(passed_confidence)
    failed_mean, failed_count = _average(failed_confidence)
    return GenerationEval(
        faithfulness=sum(row.faithfulness for row in rows) / count,
        groundedness=sum(row.groundedness for row in rows) / count,
        correctness=sum(row.correctness for row in rows) / count,
        confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        latency_seconds=sum(row.latency_seconds for row in rows) / count,
        required_phrases=sum(row.required_phrases for row in rows) / count,
        cited_section=sum(row.cited_section for row in rows) / count,
        questions=count,
        confidence_questions=len(confidences),
        rows=tuple(rows),
        token_f1=token_mean,
        phrase_coverage=phrase_mean,
        citation_recall=citation_mean,
        distractor_citation=sum(row.distractor_citation for row in rows) / count,
        abstain_contamination=abstain_mean,
        confidence_when_passed=passed_mean,
        confidence_when_failed=failed_mean,
        token_f1_questions=token_count,
        phrase_coverage_questions=phrase_count,
        citation_recall_questions=citation_count,
        abstain_questions=abstain_count,
        confidence_passed_questions=passed_count,
        confidence_failed_questions=failed_count,
    )


def run_golden(k: int = K) -> Path:
    """Write every answer, then score that file with Jev."""
    generate_answers(k=k)
    return score_answers()


def _generate_once(
    question: object, hits: Sequence[HybridHit]
) -> tuple[str | None, float, str | None]:
    """Generate one answer, retrying the model call. Returns text, latency, error."""
    prompt = generation_prompt(str(question), hits)
    started = time.perf_counter()
    try:
        text = call_with_retry(lambda: generate(prompt))
    except Exception as exc:
        return None, time.perf_counter() - started, str(exc)
    return text, time.perf_counter() - started, None


def _print_jev(record: Mapping[str, object]) -> None:
    """Print the Jev Score choices for one question."""
    jev = record.get("jev")
    item_id = record["id"]
    if not isinstance(jev, Mapping):
        print(f"{item_id} jev skipped")
        return
    if jev.get("error"):
        print(f"{item_id} jev failed: {jev['error']}")
        return
    answers = jev.get("answers")
    if not isinstance(answers, Mapping):
        print(f"{item_id} jev failed: no answers")
        return
    parts = [
        f"{name}={answers[name].get('choice')}"
        for name in _JUDGE_NAMES
        if isinstance(answers.get(name), Mapping)
    ]
    latency = jev.get("latency_seconds")
    latency_text = f"{float(latency):.4f}" if isinstance(latency, (int, float)) else "none"
    print(
        f"{item_id} jev {' '.join(parts)} "
        f"latency_seconds={latency_text} "
        f"prompt_tokens={jev.get('prompt_tokens')} "
        f"generation_tokens={jev.get('generation_tokens')}"
    )


def _retrieval_jev(search: object) -> dict[str, object]:
    """Latency and generation tokens from the Jev call after the cross-encoder."""
    result = getattr(search, "last_jev", None)
    if not isinstance(result, Mapping):
        return {"latency_seconds": None, "generation_tokens": None}
    latency_ms = result.get("latency_ms")
    tokens = result.get("generation_tokens")
    latency = float(latency_ms) / 1000 if isinstance(latency_ms, (int, float)) else None
    generation = tokens if isinstance(tokens, int) else None
    return {"latency_seconds": latency, "generation_tokens": generation}


def _jev_score_row(record: Mapping[str, object]) -> dict[str, object]:
    """One Jev Choice row: latency, generation tokens, and the three answers."""
    row: dict[str, object] = {"id": record["id"]}
    jev = record.get("jev")
    if not isinstance(jev, Mapping):
        row["latency_seconds"] = None
        row["generation_tokens"] = None
        row["error"] = None
        for name in _JUDGE_NAMES:
            row[name] = None
        return row
    row["latency_seconds"] = jev.get("latency_seconds")
    row["generation_tokens"] = jev.get("generation_tokens")
    row["error"] = jev.get("error")
    answers = jev.get("answers")
    answers = answers if isinstance(answers, Mapping) else {}
    for name in _JUDGE_NAMES:
        row[name] = answers.get(name)
    return row


def _stored_hit(hit: HybridHit) -> dict[str, object]:
    return {
        "chunk_id": hit.chunk_id,
        "section_id": hit.parent_id,
        "section_name": hit.section_name,
        "filename": hit.filename,
        "body": hit.body,
    }


def _hit_from_stored(record: Mapping[str, object]) -> HybridHit:
    return HybridHit(
        chunk_id=str(record["chunk_id"]),
        filename=str(record["filename"]),
        title="",
        page_no=None,
        section_name=str(record["section_name"]),
        parent_id=str(record["section_id"]),
        start_span=None,
        end_span=None,
        document_date=None,
        date_source=None,
        version=None,
        superseded=False,
        body=str(record["body"]),
        score=0.0,
        rrf_score=0.0,
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _json_object(text: str) -> dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _tokens(text: str) -> list[str]:
    """Case-insensitive letter and digit words in ``text``."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def _optional_int(value: object) -> int | None:
    """Return ``value`` as an int when the usage field is a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _average(values: Sequence[float]) -> tuple[float, int]:
    """Mean of ``values``. An empty sequence averages to 0 over 0 questions."""
    if not values:
        return 0.0, 0
    return sum(values) / len(values), len(values)


def _confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if 0 <= number <= 1:
        return number
    if 1 < number <= 100:
        return number / 100
    return None


def _mentions_id(parsed: Generation, section_id: str) -> bool:
    texts = [parsed.answer, *(section.section_id for section in parsed.sections)]
    return any(_id_in(text, section_id) for text in texts)


def _id_in(text: str, section_id: str) -> bool:
    """True when ``section_id`` occurs and is not a prefix of a longer index."""
    start = 0
    while True:
        index = text.find(section_id, start)
        if index < 0:
            return False
        end = index + len(section_id)
        if end == len(text) or not text[end].isdigit():
            return True
        start = index + 1


def _sections(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [section for section in value if isinstance(section, Mapping)]


def _evidence_quotes(value: object) -> list[str]:
    quotes: list[str] = []
    for section in _sections(value):
        evidence = section.get("evidence", [])
        if not isinstance(evidence, list):
            continue
        for quote in evidence:
            if isinstance(quote, Mapping) and isinstance(quote.get("text"), str):
                quotes.append(quote["text"])
    return quotes


def main() -> None:
    """Write answers.json, then the Jev judge file."""
    generate_answers()
    print(f"answers: {ANSWERS_PATH}")
    score_answers()
    print(f"jev evals: {JEV_EVALS_PATH}")
    print(f"generator: {config.MODEL} ({config.MODELS[config.MODEL]})")
    print(f"jev_judge: {_JEV_MODEL}")


if __name__ == "__main__":
    main()
