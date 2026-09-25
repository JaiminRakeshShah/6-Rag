"""Generation evals for the golden question set.

Answers are generated first and written to ``data/generation/answers.json``.
Evals then read that file and write one row per question to
``data/generation/evals.json``. The prompt asks the generator to name the
sections it used, with the same ``section_id`` and ``section_name`` fields
the gold file stores, and to give a confidence from 0 to 1. Generation
latency is the wait for that call only.

DeepEval GEval, judged by ``gptoss`` (``gpt-oss:20b``), scores
faithfulness, groundedness, and correctness. Temperature is 0. ``top_k``
and ``top_p`` are left unset. Deterministic scores sit beside the judge:
phrase coverage, citation recall, distractor citation, token F1 against
``expected_answer``, and abstain contamination. The two pass/fail checks
remain: every ``answer_must_include`` phrase is present, and a cited
section matches a gold supporting section.

The generator call and the three judge calls each retry up to ``ATTEMPTS``
times. A question that still fails is stored with ``error`` set, and the
run continues.
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
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src import config
from src.generate import generate
from src.retrieve import INITIAL_K, HybridHit, hybrid_search
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
EVALS_PATH = ROOT / "data" / "generation" / "evals.json"
_DEFAULT_BASE_URL = "http://host.docker.internal:11434"
_JUDGE_NAMES = ("faithfulness", "groundedness", "correctness")


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


def generation_prompt(question: str, hits: Sequence[HybridHit]) -> str:
    """Ask for an answer, the sections used, and a confidence."""
    blocks: list[str] = []
    for hit in hits:
        blocks.append(
            "\n".join(
                (
                    f"chunk_id: {hit.chunk_id}",
                    f"section_id: {hit.parent_id}",
                    f"section_name: {hit.section_name}",
                    f"filename: {hit.filename}",
                    "body:",
                    hit.body,
                )
            )
        )
    chunks = "\n\n".join(blocks) if blocks else "(no chunks retrieved)"
    return (
        "Answer the question using only the retrieved chunks. "
        "If they do not contain the answer, say the documents do not say, "
        "and do not invent a figure or a name.\n"
        "Name every section you used. Copy section_id and section_name "
        "exactly as they are written on the chunk. Those are the same "
        "fields a gold supporting section records.\n"
        "Reply with one JSON object and no other text:\n"
        '{"answer": "...", "sections": [{"section_id": "...", "section_name": "..."}], '
        '"confidence": 0.0}\n'
        "confidence is a number from 0 to 1.\n\n"
        f"Question: {question}\n\n"
        f"Chunks:\n{chunks}\n"
    )


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


class OllamaJudge(DeepEvalBaseLLM):
    """GEval judge that posts to the host Ollama server over HTTP.

    Same call as the generator: ``POST {OLLAMA_BASE_URL}/api/generate``,
    stream off, temperature 0. ``top_k`` and ``top_p`` are not sent.
    ``think`` is off so ``gpt-oss`` writes the score into ``response``.
    A JSON schema in ``format`` makes that field come back empty, so the
    schema DeepEval passes is not forwarded.
    """

    def __init__(self, model_id: str) -> None:
        self.temperature = 0
        super().__init__(model_id)

    def load_model(self) -> str:
        return self.name

    def generate(self, prompt: str, schema: type | None = None) -> str:
        del schema
        base_url = os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        response = httpx.post(
            f"{base_url}/api/generate",
            json={
                "model": self.name,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": self.temperature},
            },
            timeout=180.0,
        )
        response.raise_for_status()
        return _judge_text(response.json())

    async def a_generate(self, prompt: str, schema: type | None = None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return self.name


def judge_model() -> OllamaJudge:
    """GEval judge. The model is ``gptoss``, temperature 0."""
    return OllamaJudge(config.MODELS["gptoss"])


def geval_metrics(model: OllamaJudge | None = None) -> dict[str, GEval]:
    """Faithfulness, groundedness, and correctness, each a GEval metric."""
    judge = judge_model() if model is None else model
    return {
        "faithfulness": GEval(
            name="Faithfulness",
            evaluation_steps=[
                "Extract the factual claims in the actual output.",
                "Check each claim against the retrieval context.",
                "Score lower when a claim contradicts the retrieval context.",
            ],
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            model=judge,
            async_mode=False,
        ),
        "groundedness": GEval(
            name="Groundedness",
            evaluation_steps=[
                "Extract the factual claims in the actual output.",
                "Mark a claim as grounded only when the retrieval context states it.",
                "Score lower when a claim is not stated in the retrieval context, "
                "even if it does not contradict that context.",
            ],
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            model=judge,
            async_mode=False,
        ),
        "correctness": GEval(
            name="Correctness",
            evaluation_steps=[
                "Compare the facts in the actual output with the expected output.",
                "Score lower when a fact required by the expected output is missing.",
                "When the expected output says the documents do not contain the answer, score lower if the actual output states a figure or a name as fact.",
            ],
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=judge,
            async_mode=False,
        ),
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


def score_item(
    item: Mapping[str, object],
    hits: Sequence[HybridHit],
    text: str,
    latency_seconds: float,
    metrics: Mapping[str, GEval],
) -> QuestionScore:
    """Score one generated reply with the judge and the two checks."""
    record, row = eval_record(item, hits, text, latency_seconds, metrics)
    if row is None:
        raise ValueError(record["error"])
    return row


def eval_record(
    item: Mapping[str, object],
    hits: Sequence[HybridHit],
    text: str,
    latency_seconds: float,
    metrics: Mapping[str, GEval],
) -> tuple[dict[str, object], QuestionScore | None]:
    """Judge one stored answer. A failed metric is retried, then left empty."""
    parsed = parse_generation(text)
    test_case = LLMTestCase(
        input=str(item["question"]),
        actual_output=actual_output(parsed),
        expected_output=expected_output(item),
        retrieval_context=[context_block(hit) for hit in hits],
    )
    item_id = item["id"]
    if not isinstance(item_id, str):
        raise ValueError("id must be a string")
    record: dict[str, object] = {
        "id": item_id,
        "confidence": parsed.confidence,
        "latency_seconds": latency_seconds,
        "required_phrases": required_phrases(parsed.answer, item),
        "cited_section": cited_section(parsed, item),
        "token_f1": _token_f1_for(parsed.answer, item),
        "phrase_coverage": phrase_coverage(parsed.answer, item),
        "citation_recall": citation_recall(parsed, item),
        "distractor_citation": distractor_citation(parsed, item),
        "abstain_contamination": abstain_contamination(parsed.answer, item),
        "error": None,
    }
    scores: dict[str, float] = {}
    for name in _JUDGE_NAMES:
        try:
            score, reason = _measure(metrics[name], test_case)
        except Exception as exc:
            record[name] = None
            record[f"{name}_reason"] = None
            message = f"{name}: {exc}"
            record["error"] = message if record["error"] is None else f"{record['error']}; {message}"
            continue
        scores[name] = score
        record[name] = score
        record[f"{name}_reason"] = reason
    if len(scores) != len(_JUDGE_NAMES):
        return record, None
    row = QuestionScore(
        item_id=item_id,
        faithfulness=scores["faithfulness"],
        groundedness=scores["groundedness"],
        correctness=scores["correctness"],
        confidence=parsed.confidence,
        latency_seconds=latency_seconds,
        required_phrases=bool(record["required_phrases"]),
        cited_section=bool(record["cited_section"]),
        token_f1=_optional_float(record["token_f1"]),
        phrase_coverage=_optional_float(record["phrase_coverage"]),
        citation_recall=_optional_float(record["citation_recall"]),
        distractor_citation=float(record["distractor_citation"]),
        abstain_contamination=_optional_bool(record["abstain_contamination"]),
    )
    return record, row


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


def score_answers(answers_path: Path = ANSWERS_PATH, evals_path: Path = EVALS_PATH) -> GenerationEval | None:
    """Score the stored answers and write one eval row per question."""
    stored = json.loads(answers_path.read_text(encoding="utf-8"))
    dataset = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    gold = {item["id"]: item for item in dataset["items"]}
    metrics = geval_metrics()
    eval_items: list[dict[str, object]] = []
    scored: list[QuestionScore] = []
    for record in stored["items"]:
        item = gold[record["id"]]
        if record.get("error"):
            row_record, row = _failed_eval(record), None
        else:
            hits = [_hit_from_stored(hit) for hit in record["retrieved"]]
            row_record, row = eval_record(
                item,
                hits,
                str(record["raw"]),
                float(record["latency_seconds"]),
                metrics,
            )
        eval_items.append(row_record)
        if row is not None:
            scored.append(row)
            _print_row(row)
        else:
            print(f"{record['id']} eval failed: {row_record['error']}")
        _write_json(
            evals_path,
            {"judge": config.MODELS["gptoss"], "items": eval_items},
        )
    if not scored:
        return None
    return evaluate(scored)


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


def run_golden(k: int = K) -> GenerationEval:
    """Write every answer, then score that file."""
    generate_answers(k=k)
    result = score_answers()
    if result is None:
        raise ValueError("no scored questions")
    return result


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


def _measure(metric: GEval, test_case: LLMTestCase) -> tuple[float, str]:
    """Run one judge metric, retrying the model call."""

    def once() -> tuple[float, str]:
        metric.measure(test_case, _show_indicator=False)
        reason = getattr(metric, "reason", "")
        return float(metric.score), "" if reason is None else str(reason)

    return call_with_retry(once)


def _failed_eval(record: Mapping[str, object]) -> dict[str, object]:
    """Eval row for an answer that was not generated."""
    row: dict[str, object] = {
        "id": record["id"],
        "confidence": None,
        "latency_seconds": record.get("latency_seconds"),
        "required_phrases": False,
        "cited_section": False,
        "token_f1": None,
        "phrase_coverage": None,
        "citation_recall": None,
        "distractor_citation": 0.0,
        "abstain_contamination": None,
        "error": record.get("error"),
    }
    for name in _JUDGE_NAMES:
        row[name] = None
        row[f"{name}_reason"] = None
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


def _print_row(row: QuestionScore) -> None:
    confidence = "none" if row.confidence is None else f"{row.confidence:.4f}"
    print(
        f"{row.item_id} faithfulness={row.faithfulness:.4f} "
        f"groundedness={row.groundedness:.4f} correctness={row.correctness:.4f} "
        f"confidence={confidence} latency_seconds={row.latency_seconds:.4f} "
        f"required_phrases={int(row.required_phrases)} cited_section={int(row.cited_section)} "
        f"token_f1={_format_optional(row.token_f1)} "
        f"phrase_coverage={_format_optional(row.phrase_coverage)} "
        f"citation_recall={_format_optional(row.citation_recall)} "
        f"distractor_citation={row.distractor_citation:.4f} "
        f"abstain_contamination={row.abstain_contamination}"
    )


def _judge_text(payload: Mapping[str, object]) -> str:
    """Return the score JSON, or raise with the text Ollama actually sent."""
    response = payload.get("response")
    thinking = payload.get("thinking")
    text = response if isinstance(response, str) else ""
    if "{" not in text and isinstance(thinking, str):
        text = thinking
    start = text.find("{")
    end = text.rfind("}")
    snippet = text[start : end + 1] if start >= 0 and end > start else ""
    if not snippet:
        preview = text.strip().replace("\n", " ")[:180]
        raise ValueError(
            f"Ollama returned no JSON score (done_reason={payload.get('done_reason')!s}). {preview}"
        )
    try:
        json.loads(snippet)
    except json.JSONDecodeError as exc:
        preview = snippet.replace("\n", " ")[:180]
        raise ValueError(f"Ollama judge returned invalid JSON ({exc.msg}). {preview}") from exc
    return snippet


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


def _token_f1_for(answer: str, item: Mapping[str, object]) -> float | None:
    """Token F1 against ``expected_answer``. Abstain and missing answers are excluded."""
    expected = item.get("expected_answer")
    if not isinstance(expected, str):
        return None
    return token_f1(answer, expected)


def _optional_float(value: object) -> float | None:
    """Return ``value`` as a float, or ``None`` when the score does not apply."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_bool(value: object) -> bool | None:
    """Return ``value`` when it is a bool, otherwise ``None``."""
    if isinstance(value, bool):
        return value
    return None


def _average(values: Sequence[float]) -> tuple[float, int]:
    """Mean of ``values``. An empty sequence averages to 0 over 0 questions."""
    if not values:
        return 0.0, 0
    return sum(values) / len(values), len(values)


def _format_optional(value: float | None) -> str:
    """Print a missing score as ``none``."""
    if value is None:
        return "none"
    return f"{value:.4f}"


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
    """Write answers.json, then evals.json, and print the means."""
    generate_answers()
    print(f"answers: {ANSWERS_PATH}")
    result = score_answers()
    print(f"evals: {EVALS_PATH}")
    if result is None:
        print("no scored questions")
        return
    print(f"generator: {config.MODEL} ({config.MODELS[config.MODEL]})")
    print(f"judge: {config.MODELS['gptoss']}")
    print(f"questions: {result.questions}")
    print(f"faithfulness: {result.faithfulness:.4f}")
    print(f"groundedness: {result.groundedness:.4f}")
    print(f"correctness: {result.correctness:.4f}")
    print(f"confidence: {result.confidence:.4f} ({result.confidence_questions} reported)")
    print(f"latency_seconds: {result.latency_seconds:.4f}")
    print(f"required_phrases: {result.required_phrases:.4f}")
    print(f"cited_section: {result.cited_section:.4f}")
    print(f"token_f1: {result.token_f1:.4f} ({result.token_f1_questions} questions)")
    print(f"phrase_coverage: {result.phrase_coverage:.4f} ({result.phrase_coverage_questions} questions)")
    print(f"citation_recall: {result.citation_recall:.4f} ({result.citation_recall_questions} questions)")
    print(f"distractor_citation: {result.distractor_citation:.4f}")
    print(f"abstain_contamination: {result.abstain_contamination:.4f} ({result.abstain_questions} questions)")
    print(
        f"confidence_when_passed: {result.confidence_when_passed:.4f} "
        f"({result.confidence_passed_questions} questions)"
    )
    print(
        f"confidence_when_failed: {result.confidence_when_failed:.4f} "
        f"({result.confidence_failed_questions} questions)"
    )


if __name__ == "__main__":
    main()
