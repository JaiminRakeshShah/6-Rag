import json

import pytest

from src.eval_generation import (
    ATTEMPTS,
    K,
    CitedSection,
    Generation,
    QuestionScore,
    abstain_contamination,
    call_with_retry,
    citation_recall,
    cited_section,
    distractor_citation,
    evaluate,
    expected_output,
    generate_answers,
    geval_metrics,
    judge_model,
    generation_prompt,
    hard_checks_passed,
    parse_generation,
    phrase_coverage,
    required_phrases,
    score_answers,
    score_item,
    token_f1,
)
from src.retrieve import INITIAL_K, HybridHit


def _hit(
    section_id: str = "Carbon_New_2040#1",
    name: str = "Commitment to Achieving Net Zero",
    body: str = "Net Zero emission by 2040.",
) -> HybridHit:
    return HybridHit(
        chunk_id=f"{section_id}:0",
        filename="Carbon_New_2040.md",
        title="Carbon",
        page_no=None,
        section_name=name,
        parent_id=section_id,
        start_span=0,
        end_span=10,
        document_date=None,
        date_source=None,
        version=None,
        superseded=False,
        body=body,
        score=1.0,
        rrf_score=0.1,
    )


def _item(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "id": "gold-001",
        "question": "By which year?",
        "abstain": False,
        "expected_answer": "2040",
        "answer_must_include": ["2040"],
        "supporting_sections": [
            {
                "section_id": "Carbon_New_2040#1",
                "section_name": "Commitment to Achieving Net Zero",
            }
        ],
        "distractor_sections": [
            {
                "section_id": "Carbon_Old_2050#2",
                "section_name": "Commitment to Achieving Net Zero",
            }
        ],
    }
    item.update(overrides)
    return item


def _cited(section_id: str, section_name: str = "") -> Generation:
    return Generation(
        answer="2040",
        sections=(CitedSection(section_id, section_name),),
        confidence=0.5,
    )


class _Metric:
    def __init__(self, score: float) -> None:
        self.score = score
        self.case = None

    def measure(self, test_case: object, _show_indicator: bool = True) -> float:
        self.case = test_case
        return self.score


def test_prompt_asks_for_the_gold_section_fields() -> None:
    prompt = generation_prompt("By which year?", [_hit()])

    assert "section_id: Carbon_New_2040#1" in prompt
    assert "section_name: Commitment to Achieving Net Zero" in prompt
    assert "chunk_id: Carbon_New_2040#1:0" in prompt
    assert '"section_id"' in prompt
    assert '"section_name"' in prompt
    assert "confidence" in prompt
    assert "Question: By which year?" in prompt


def test_parse_generation_reads_sections_and_confidence() -> None:
    text = (
        '```json\n{"answer": "2040", "sections": '
        '[{"section_id": "Carbon_New_2040#1", "section_name": "Commitment"}], '
        '"confidence": "0.8"}\n```'
    )
    parsed = parse_generation(text)

    assert parsed.answer == "2040"
    assert parsed.sections == (
        CitedSection("Carbon_New_2040#1", "Commitment"),
    )
    assert parsed.confidence == pytest.approx(0.8)


def test_parse_generation_accepts_a_chunk_id_and_a_percent() -> None:
    parsed = parse_generation(
        '{"answer": "2050", "sections": [{"chunk_id": "Carbon_Old_2050#2:0"}], "confidence": 80}'
    )

    assert parsed.sections[0].section_id == "Carbon_Old_2050#2:0"
    assert parsed.confidence == pytest.approx(0.8)


def test_non_json_reply_has_no_citation() -> None:
    assert parse_generation("The year is 2040.") == Generation(
        answer="The year is 2040.",
        sections=(),
        confidence=None,
    )


def test_required_phrases_are_case_insensitive() -> None:
    assert required_phrases("Net zero by 2040.", _item())
    assert not required_phrases("Net zero soon.", _item())


def test_abstain_rejects_distractor_evidence() -> None:
    item = _item(
        abstain=True,
        expected_answer=None,
        answer_must_include=[],
        supporting_sections=[],
        distractor_sections=[
            {
                "section_id": "Envi_2040-1#13",
                "section_name": "John Speight",
                "evidence": [{"text": "information@coforge.com"}],
            }
        ],
    )

    assert required_phrases("The documents do not say.", item)
    assert not required_phrases("Write to information@coforge.com.", item)


def test_cited_section_matches_the_gold_id() -> None:
    assert cited_section(_cited("Carbon_New_2040#1"), _item())
    assert cited_section(_cited("Carbon_New_2040#1:0"), _item())
    assert not cited_section(_cited("Carbon_New_2040#10"), _item())
    assert not cited_section(_cited("Carbon_Old_2050#2"), _item())


def test_shared_section_name_needs_the_section_id() -> None:
    named = Generation(
        answer="2040",
        sections=(CitedSection("", "Commitment to Achieving Net Zero"),),
        confidence=None,
    )

    assert not cited_section(named, _item())


def test_a_unique_section_name_matches() -> None:
    item = _item(distractor_sections=[])
    named = Generation(
        answer="See the preamble.",
        sections=(CitedSection("", "Commitment to Achieving Net Zero"),),
        confidence=None,
    )

    assert cited_section(named, item)


def test_abstain_matches_only_when_nothing_is_cited() -> None:
    item = _item(abstain=True, supporting_sections=[], answer_must_include=[])

    assert cited_section(Generation("The documents do not say.", (), None), item)
    assert not cited_section(_cited("Envi_2040-1#13", "John Speight"), item)


def test_expected_output_for_an_abstain_declines() -> None:
    text = expected_output(_item(abstain=True, expected_answer=None, answer_must_include=[]))

    assert "do not contain the answer" in text


def test_token_f1_is_word_overlap() -> None:
    assert token_f1("Version 1.0", "1.0") == pytest.approx(0.8)
    assert token_f1("about 50% by 2030", "about 25% by 2030") == pytest.approx(0.75)
    assert token_f1("2040", "2040") == pytest.approx(1.0)
    assert token_f1("2040", "") == pytest.approx(0.0)
    assert token_f1("", "") == pytest.approx(1.0)


def test_phrase_coverage_is_the_share_of_gold_phrases() -> None:
    item = _item(answer_must_include=["half", "quarterly"])

    assert phrase_coverage("A quarterly report.", item) == pytest.approx(0.5)
    assert phrase_coverage("A half-year quarterly report.", item) == pytest.approx(1.0)
    assert phrase_coverage("The documents do not say.", _item(abstain=True)) is None


def test_citation_recall_counts_each_supporting_section() -> None:
    item = _item(
        supporting_sections=[
            {"section_id": "Carbon_New_2040#6", "section_name": "Carbon Reduction Initiatives"},
            {"section_id": "Envi_2040-1#5", "section_name": "Energy Optimization"},
        ]
    )
    parsed = Generation(
        answer="10% by 2025.",
        sections=(CitedSection("Carbon_New_2040#6:0", "Carbon Reduction Initiatives"),),
        confidence=0.9,
    )

    assert citation_recall(parsed, item) == pytest.approx(0.5)
    assert citation_recall(parsed, _item(abstain=True, supporting_sections=[])) is None


def test_distractor_citation_is_the_share_of_cited_sections() -> None:
    parsed = Generation(
        answer="10% and 25%.",
        sections=(
            CitedSection("Carbon_New_2040#6", "Carbon Reduction Initiatives"),
            CitedSection("Carbon_Old_2050#8:0", "Carbon Reduction Initiatives"),
        ),
        confidence=0.9,
    )
    item = _item(
        distractor_sections=[
            {"section_id": "Carbon_Old_2050#8", "section_name": "Carbon Reduction Initiatives"}
        ]
    )

    assert distractor_citation(parsed, item) == pytest.approx(0.5)
    assert distractor_citation(Generation("2040", (), None), item) == pytest.approx(0.0)


def test_abstain_contamination_flags_an_added_figure_or_name() -> None:
    item = _item(
        question="What were Coforge's FY25 Scope 2 emissions in the United States?",
        abstain=True,
        expected_answer=None,
        answer_must_include=[],
        supporting_sections=[],
    )

    assert abstain_contamination("The documents do not say.", item) is False
    assert abstain_contamination("The documents do not say. The UK figure is 2.90.", item) is True
    assert abstain_contamination("The ESG Committee may amend the policy.", item) is True
    assert abstain_contamination("2040", _item()) is None


def test_evaluate_averages_scores_and_reported_confidence() -> None:
    assert K == INITIAL_K == 10
    first = QuestionScore("gold-001", 1.0, 0.5, 1.0, 0.25, 1.0, True, True)
    second = QuestionScore("gold-002", 0.0, 0.5, 0.0, None, 3.0, False, False)
    result = evaluate([first, second])

    assert result.faithfulness == pytest.approx(0.5)
    assert result.groundedness == pytest.approx(0.5)
    assert result.correctness == pytest.approx(0.5)
    assert result.confidence == pytest.approx(0.25)
    assert result.confidence_questions == 1
    assert result.latency_seconds == pytest.approx(2.0)
    assert result.required_phrases == pytest.approx(0.5)
    assert result.cited_section == pytest.approx(0.5)
    assert result.questions == 2


def test_evaluate_splits_confidence_by_the_hard_checks() -> None:
    passed = QuestionScore(
        "gold-001",
        1.0,
        1.0,
        1.0,
        0.4,
        1.0,
        True,
        True,
        token_f1=1.0,
        phrase_coverage=1.0,
        citation_recall=1.0,
        distractor_citation=0.0,
        abstain_contamination=None,
    )
    failed = QuestionScore(
        "gold-022",
        0.5,
        1.0,
        0.6,
        0.9,
        2.0,
        True,
        True,
        token_f1=0.5,
        phrase_coverage=1.0,
        citation_recall=1.0,
        distractor_citation=0.5,
        abstain_contamination=None,
    )

    assert hard_checks_passed(passed)
    assert not hard_checks_passed(failed)
    result = evaluate([passed, failed])

    assert result.token_f1 == pytest.approx(0.75)
    assert result.phrase_coverage == pytest.approx(1.0)
    assert result.citation_recall == pytest.approx(1.0)
    assert result.distractor_citation == pytest.approx(0.25)
    assert result.abstain_contamination == pytest.approx(0.0)
    assert result.abstain_questions == 0
    assert result.confidence_when_passed == pytest.approx(0.4)
    assert result.confidence_when_failed == pytest.approx(0.9)


def test_empty_eval_raises() -> None:
    with pytest.raises(ValueError):
        evaluate([])


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_judge_posts_to_the_host_ollama_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    calls: list[tuple[str, dict]] = []

    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        calls.append((url, json))
        return _Response({"response": '{"score": 8, "reason": "supported"}'})

    monkeypatch.setattr("src.eval_generation.httpx.post", fake_post)
    from pydantic import BaseModel

    class _Score(BaseModel):
        score: int
        reason: str

    text = judge_model().generate("Score this.", schema=_Score)

    assert text == '{"score": 8, "reason": "supported"}'
    url, body = calls[0]
    assert url == "http://host.docker.internal:11434/api/generate"
    assert body["model"] == "gpt-oss:20b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"] == {"temperature": 0}
    assert body["prompt"] == "Score this."
    assert "format" not in body
    assert "top_k" not in body["options"]
    assert "top_p" not in body["options"]


def test_judge_reads_json_from_thinking_when_response_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        return _Response({"response": "", "thinking": 'notes {"score": 1, "reason": "no"} tail', "done_reason": "stop"})

    monkeypatch.setattr("src.eval_generation.httpx.post", fake_post)

    assert judge_model().generate("Score this.") == '{"score": 1, "reason": "no"}'


def test_judge_reports_an_empty_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        return _Response({"response": "", "thinking": "", "done_reason": "stop"})

    monkeypatch.setattr("src.eval_generation.httpx.post", fake_post)

    with pytest.raises(ValueError, match="done_reason=stop"):
        judge_model().generate("Score this.")


def test_judge_is_gptoss_at_temperature_zero() -> None:
    metrics = geval_metrics()

    assert set(metrics) == {"faithfulness", "groundedness", "correctness"}
    for metric in metrics.values():
        assert metric.model.name == "gpt-oss:20b"
        assert metric.model.temperature == 0
        assert metric.evaluation_steps
        assert metric.async_mode is False


def test_score_item_logs_the_judge_and_the_checks() -> None:
    metrics = {
        "faithfulness": _Metric(0.2),
        "groundedness": _Metric(0.4),
        "correctness": _Metric(0.6),
    }
    text = (
        '{"answer": "2040", "sections": '
        '[{"section_id": "Carbon_New_2040#1", "section_name": "Commitment to Achieving Net Zero"}], '
        '"confidence": 0.7}'
    )
    row = score_item(_item(), [_hit()], text, 1.5, metrics)

    assert row.faithfulness == pytest.approx(0.2)
    assert row.groundedness == pytest.approx(0.4)
    assert row.correctness == pytest.approx(0.6)
    assert row.confidence == pytest.approx(0.7)
    assert row.latency_seconds == pytest.approx(1.5)
    assert row.required_phrases
    assert row.cited_section
    assert row.token_f1 == pytest.approx(1.0)
    assert row.phrase_coverage == pytest.approx(1.0)
    assert row.citation_recall == pytest.approx(1.0)
    assert row.distractor_citation == pytest.approx(0.0)
    assert row.abstain_contamination is None
    case = metrics["correctness"].case
    assert case.expected_output.startswith("2040")
    assert "section_id: Carbon_New_2040#1" in case.retrieval_context[0]
    assert "Sections used:" in case.actual_output


def test_retry_runs_three_times_then_raises() -> None:
    calls = {"n": 0}

    def fail() -> None:
        calls["n"] += 1
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="down"):
        call_with_retry(fail)

    assert ATTEMPTS == 3
    assert calls["n"] == 3


def test_retry_returns_the_first_success() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("empty")
        return "ok"

    assert call_with_retry(flaky) == "ok"
    assert calls["n"] == 2


def test_answers_are_written_before_evals(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"items": [_item()]}), encoding="utf-8")
    monkeypatch.setattr("src.eval_generation.GOLD_PATH", gold)
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("empty")
        return (
            '{"answer": "2040", "sections": [{"section_id": "Carbon_New_2040#1", '
            '"section_name": "Commitment to Achieving Net Zero"}], "confidence": 0.4}'
        )

    class _Client:
        def close(self) -> None:
            return None

    monkeypatch.setattr("src.eval_generation.generate", flaky)
    monkeypatch.setattr("src.eval_generation.connect", lambda: _Client())
    monkeypatch.setattr("src.eval_generation.hybrid_search", lambda client, question, k: [_hit()])
    answers = tmp_path / "answers.json"
    generate_answers(answers)

    stored = json.loads(answers.read_text(encoding="utf-8"))
    assert calls["n"] == 2
    assert stored["items"][0]["answer"] == "2040"
    assert stored["items"][0]["error"] is None
    assert stored["items"][0]["retrieved"][0]["section_id"] == "Carbon_New_2040#1"

    def fail_generate(prompt: str) -> str:
        raise AssertionError("evals must not generate")

    monkeypatch.setattr("src.eval_generation.generate", fail_generate)
    monkeypatch.setattr(
        "src.eval_generation.geval_metrics",
        lambda model=None: {
            "faithfulness": _Metric(0.2),
            "groundedness": _Metric(0.4),
            "correctness": _Metric(0.6),
        },
    )
    evals = tmp_path / "evals.json"
    result = score_answers(answers, evals)
    saved = json.loads(evals.read_text(encoding="utf-8"))

    assert result is not None
    assert result.faithfulness == pytest.approx(0.2)
    assert saved["judge"] == "gpt-oss:20b"
    assert saved["items"][0]["correctness"] == pytest.approx(0.6)
    assert saved["items"][0]["required_phrases"] is True
    assert saved["items"][0]["cited_section"] is True
