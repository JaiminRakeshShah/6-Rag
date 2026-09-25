"""Deterministic checks for the minimal-rag answers.

This is a copy of ``src/eval_generation.py`` with the generator and the
DeepEval judge removed. It reads ``answers.json`` in this folder and writes
``evals.json`` beside it. The gold file stays ``data/gold/policy_rag_golden.json``.

Scores are phrase coverage, citation recall, distractor citation, token F1
against ``expected_answer``, and abstain contamination. The two pass/fail
checks remain: every ``answer_must_include`` phrase is present, and a cited
section matches a gold supporting section.
"""

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLD_PATH = ROOT / "data" / "gold" / "policy_rag_golden.json"
ANSWERS_PATH = HERE / "answers.json"
EVALS_PATH = HERE / "evals.json"
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
    """The model's confidence, latency, and the deterministic scores.

    ``token_f1``, ``phrase_coverage``, and ``citation_recall`` are ``None``
    when that score does not apply. ``abstain_contamination`` is ``None`` on
    a scored question and a bool on an abstain question.
    """

    item_id: str
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
    hard_checks_passed: float = 0.0


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


def score_answers(answers_path: Path = ANSWERS_PATH, evals_path: Path = EVALS_PATH) -> GenerationEval | None:
    """Score the stored answers and write one eval row per question."""
    stored = json.loads(answers_path.read_text(encoding="utf-8"))
    dataset = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    gold = {item["id"]: item for item in dataset["items"]}
    eval_items: list[dict[str, object]] = []
    scored: list[QuestionScore] = []
    for record in stored["items"]:
        item = gold[record["id"]]
        if record.get("error") or not isinstance(record.get("raw"), str):
            row_record, row = _failed_eval(record), None
        else:
            row = _score_row(item, str(record["raw"]), float(record["latency_seconds"]))
            row_record = _stored_row(row)
        eval_items.append(row_record)
        if row is not None:
            scored.append(row)
            _print_row(row)
        else:
            print(f"{record['id']} eval failed: {row_record['error']}")
        _write_json(evals_path, {"items": eval_items})
    if not scored:
        return None
    result = evaluate(scored)
    _write_json(evals_path, {"summary": _summary(result), "items": eval_items})
    return result


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
        hard_checks_passed=sum(hard_checks_passed(row) for row in rows) / count,
    )


def _score_row(item: Mapping[str, object], text: str, latency_seconds: float) -> QuestionScore:
    """Score one stored reply with the deterministic checks."""
    parsed = parse_generation(text)
    item_id = item["id"]
    if not isinstance(item_id, str):
        raise ValueError("id must be a string")
    return QuestionScore(
        item_id=item_id,
        confidence=parsed.confidence,
        latency_seconds=latency_seconds,
        required_phrases=required_phrases(parsed.answer, item),
        cited_section=cited_section(parsed, item),
        token_f1=_token_f1_for(parsed.answer, item),
        phrase_coverage=phrase_coverage(parsed.answer, item),
        citation_recall=citation_recall(parsed, item),
        distractor_citation=distractor_citation(parsed, item),
        abstain_contamination=abstain_contamination(parsed.answer, item),
    )


def _failed_eval(record: Mapping[str, object]) -> dict[str, object]:
    """Eval row for an answer that was not generated."""
    return {
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
        "hard_checks_passed": False,
        "error": record.get("error") or "missing answer",
    }


def _stored_row(row: QuestionScore) -> dict[str, object]:
    return {
        "id": row.item_id,
        "confidence": row.confidence,
        "latency_seconds": row.latency_seconds,
        "required_phrases": row.required_phrases,
        "cited_section": row.cited_section,
        "token_f1": row.token_f1,
        "phrase_coverage": row.phrase_coverage,
        "citation_recall": row.citation_recall,
        "distractor_citation": row.distractor_citation,
        "abstain_contamination": row.abstain_contamination,
        "hard_checks_passed": hard_checks_passed(row),
        "error": None,
    }


def _summary(result: GenerationEval) -> dict[str, object]:
    return {
        "questions": result.questions,
        "latency_seconds": result.latency_seconds,
        "required_phrases": result.required_phrases,
        "cited_section": result.cited_section,
        "token_f1": result.token_f1,
        "token_f1_questions": result.token_f1_questions,
        "phrase_coverage": result.phrase_coverage,
        "phrase_coverage_questions": result.phrase_coverage_questions,
        "citation_recall": result.citation_recall,
        "citation_recall_questions": result.citation_recall_questions,
        "distractor_citation": result.distractor_citation,
        "abstain_contamination": result.abstain_contamination,
        "abstain_questions": result.abstain_questions,
        "confidence": result.confidence,
        "confidence_questions": result.confidence_questions,
        "confidence_when_passed": result.confidence_when_passed,
        "confidence_passed_questions": result.confidence_passed_questions,
        "confidence_when_failed": result.confidence_when_failed,
        "confidence_failed_questions": result.confidence_failed_questions,
        "hard_checks_passed": result.hard_checks_passed,
    }


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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _print_row(row: QuestionScore) -> None:
    confidence = "none" if row.confidence is None else f"{row.confidence:.4f}"
    print(
        f"{row.item_id} confidence={confidence} latency_seconds={row.latency_seconds:.4f} "
        f"required_phrases={int(row.required_phrases)} cited_section={int(row.cited_section)} "
        f"token_f1={_format_optional(row.token_f1)} "
        f"phrase_coverage={_format_optional(row.phrase_coverage)} "
        f"citation_recall={_format_optional(row.citation_recall)} "
        f"distractor_citation={row.distractor_citation:.4f} "
        f"abstain_contamination={row.abstain_contamination} "
        f"hard_checks_passed={int(hard_checks_passed(row))}"
    )


def main() -> None:
    """Score answers.json and write evals.json."""
    result = score_answers()
    print(f"evals: {EVALS_PATH}")
    if result is None:
        print("no scored questions")
        return
    print(f"questions: {result.questions}")
    print(f"latency_seconds: {result.latency_seconds:.4f}")
    print(f"required_phrases: {result.required_phrases:.4f}")
    print(f"cited_section: {result.cited_section:.4f}")
    print(f"token_f1: {result.token_f1:.4f} ({result.token_f1_questions} questions)")
    print(f"phrase_coverage: {result.phrase_coverage:.4f} ({result.phrase_coverage_questions} questions)")
    print(f"citation_recall: {result.citation_recall:.4f} ({result.citation_recall_questions} questions)")
    print(f"distractor_citation: {result.distractor_citation:.4f}")
    print(f"abstain_contamination: {result.abstain_contamination:.4f} ({result.abstain_questions} questions)")
    print(f"confidence: {result.confidence:.4f} ({result.confidence_questions} reported)")
    print(
        f"confidence_when_passed: {result.confidence_when_passed:.4f} "
        f"({result.confidence_passed_questions} questions)"
    )
    print(
        f"confidence_when_failed: {result.confidence_when_failed:.4f} "
        f"({result.confidence_failed_questions} questions)"
    )
    print(f"hard_checks_passed: {result.hard_checks_passed:.4f}")


if __name__ == "__main__":
    main()
