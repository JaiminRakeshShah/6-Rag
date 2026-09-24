import math

import pytest

from src.eval_retrieval import K, QueryRanking, evaluate, relevant_chunk_ids, score_ranking
from src.chunking import Chunk


def _chunk(section_id: str, index: int) -> Chunk:
    return Chunk(
        document_id="Doc",
        document_title="Doc",
        section_id=section_id,
        section_title="Section",
        index=index,
        body="body",
        text="text",
        tokens=1,
        table=False,
    )


def test_relevant_ids_follow_the_section_or_the_chunk_index() -> None:
    grouped = {"Doc#1": [_chunk("Doc#1", 0), _chunk("Doc#1", 1)]}

    whole = relevant_chunk_ids({"supporting_sections": [{"section_id": "Doc#1"}]}, grouped)
    one = relevant_chunk_ids(
        {"supporting_sections": [{"section_id": "Doc#1", "chunk_index": 1}]},
        grouped,
    )

    assert whole == {"Doc#1:0", "Doc#1:1"}
    assert one == {"Doc#1:1"}


def test_score_ranking_counts_the_first_relevant_hit() -> None:
    scores = score_ranking(["miss", "hit", "other"], {"hit"}, k=3)

    assert scores["reciprocal_rank"] == pytest.approx(0.5)
    assert scores["precision"] == pytest.approx(1 / 3)
    assert scores["precision_at_k"] == pytest.approx(1 / 3)
    assert scores["recall_at_k"] == pytest.approx(1.0)
    assert scores["ndcg"] == pytest.approx((1 / math.log2(3)) / 1.0)


def test_precision_at_k_uses_k_when_the_list_is_short() -> None:
    scores = score_ranking(["hit"], {"hit", "other"}, k=2)

    assert scores["precision"] == pytest.approx(1.0)
    assert scores["precision_at_k"] == pytest.approx(0.5)
    assert scores["recall_at_k"] == pytest.approx(0.5)
    assert scores["reciprocal_rank"] == pytest.approx(1.0)


def test_a_miss_scores_zero() -> None:
    scores = score_ranking(["miss"], {"hit"}, k=1)

    assert scores == {
        "reciprocal_rank": 0.0,
        "precision": 0.0,
        "ndcg": 0.0,
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
    }


def test_evaluate_averages_scores_and_all_latencies() -> None:
    assert K == 10
    first = QueryRanking(("hit",), frozenset({"hit"}), 0.2)
    second = QueryRanking(("miss",), frozenset({"hit"}), 0.4)
    result = evaluate([first, second], [0.2, 0.4, 1.0], k=1)

    assert result.mrr == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.ndcg == pytest.approx(0.5)
    assert result.recall_at_k == pytest.approx(0.5)
    assert result.precision_at_k == pytest.approx(0.5)
    assert result.latency_seconds == pytest.approx(1.6 / 3)
    assert result.scored_questions == 2
    assert result.timed_questions == 3


def test_empty_relevance_is_rejected() -> None:
    with pytest.raises(ValueError):
        score_ranking(["hit"], set(), k=1)
