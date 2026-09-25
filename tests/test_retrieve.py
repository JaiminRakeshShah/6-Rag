import math
import sqlite3
from pathlib import Path

import pytest

from src.chunking import chunk_markdown
from src.embed import EmbeddedChunk
from src.retrieve import INITIAL_K, JEV_KEEP, RRF_K, HybridHit, hybrid_search, reciprocal_rank_fusion
from src.vectordb import bm25_scores, build_records, connect, replace_index


def _index(
    tmp_path: Path,
    sections: list[tuple[str, str]],
    vectors: list[tuple[float, ...]],
    client=None,
):
    lines = ["# **Policy**", ""]
    for heading, body in sections:
        lines.extend([f"## **{heading}**", "", body, ""])
    markdown = "\n".join(lines)
    chunks = chunk_markdown(markdown, document_id="Policy")
    records = build_records(
        [EmbeddedChunk(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)],
        {"Policy": markdown},
    )
    if client is None:
        client = connect(tmp_path / "chroma")
    replace_index(client, records, model_id="embeddinggemma")
    return client, records


def test_bm25_ranks_the_chunk_with_more_query_terms(tmp_path: Path) -> None:
    client, records = _index(
        tmp_path,
        [("Scope", "alpha zebra."), ("Terms", "carbon 2040."), ("Other", "carbon.")],
        [(1.0, 0.0), (0.0, 1.0), (0.0, 1.0)],
    )
    scores = bm25_scores(client, 'What is "carbon" 2040?')
    by_section = {record.section_name: record.chunk_id for record in records}

    assert set(scores) == {by_section["Terms"], by_section["Other"]}
    assert scores[by_section["Terms"]] > scores[by_section["Other"]]
    assert bm25_scores(client, "???") == {}
    client.close()


def test_replace_refreshes_the_bm25_index(tmp_path: Path) -> None:
    client, records = _index(tmp_path, [("Scope", "carbon 2040.")], [(1.0, 0.0)])
    assert records[0].chunk_id in bm25_scores(client, "carbon")

    replace_index(client, [], model_id="embeddinggemma")
    assert bm25_scores(client, "carbon") == {}

    _, records2 = _index(tmp_path, [("Scope", "water policies.")], [(1.0, 0.0)], client)
    assert bm25_scores(client, "carbon") == {}
    assert records2[0].chunk_id in bm25_scores(client, "policy")
    client.close()


def test_bm25_builds_a_missing_index(tmp_path: Path) -> None:
    client, records = _index(tmp_path, [("Scope", "carbon 2040.")], [(1.0, 0.0)])
    database = sqlite3.connect(tmp_path / "chunks.sqlite")
    database.execute("DROP TABLE chunks_fts")
    database.commit()
    database.close()

    assert records[0].chunk_id in bm25_scores(client, "carbon")
    client.close()


def test_rrf_sums_reciprocal_ranks() -> None:
    assert RRF_K == 60
    scores = reciprocal_rank_fusion(["scope", "terms"], ["terms"])

    assert scores["scope"] == pytest.approx(1 / 61)
    assert scores["terms"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["terms"] > scores["scope"]


def _mock_jev(
    monkeypatch: pytest.MonkeyPatch,
    probability,
) -> list[list[HybridHit]]:
    """Record the chunks sent to Jev and rank them with ``probability``."""
    seen: list[list[HybridHit]] = []

    def fake_jev(api_key: str, question: str, hits: list[HybridHit]) -> dict[str, object]:
        seen.append(list(hits))
        return {"probabilities": {hit.chunk_id: probability(hit) for hit in hits}}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr("src.retrieve.call_jev", fake_jev)
    return seen


def test_cross_encoder_reranks_ahead_of_rrf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_vector = (1.0, 0.0)
    monkeypatch.setattr("src.retrieve.embed_texts", lambda texts: [query_vector])
    monkeypatch.setattr(
        "src.retrieve.cross_encoder_scores",
        lambda query, passages: [1.0 if "alpha zebra" in passage else 0.2 for passage in passages],
    )
    seen = _mock_jev(
        monkeypatch,
        lambda hit: 0.8 if hit.section_name == "Terms" else 0.2,
    )
    client, records = _index(
        tmp_path,
        [("Scope", "alpha zebra."), ("Terms", "carbon 2040.")],
        [query_vector, (math.cos(0.3), math.sin(0.3))],
    )
    by_section = {record.section_name: record for record in records}

    hits = hybrid_search(client, "carbon 2040", k=2)

    assert [hit.section_name for hit in seen[0]] == ["Scope", "Terms"]
    assert [hit.section_name for hit in hits] == ["Terms", "Scope"]
    scope = next(hit for hit in hits if hit.section_name == "Scope")
    terms = next(hit for hit in hits if hit.section_name == "Terms")
    assert scope.score == pytest.approx(1.0)
    assert terms.score == pytest.approx(0.2)
    assert scope.probability == pytest.approx(0.2)
    assert terms.probability == pytest.approx(0.8)
    assert terms.rrf_score > scope.rrf_score
    assert scope.chunk_id == by_section["Scope"].chunk_id
    assert hybrid_search(client, "carbon 2040", k=1)[0].section_name == "Terms"
    client.close()


def test_initial_retrieval_keeps_ten_from_each_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert INITIAL_K == 10
    monkeypatch.setattr("src.retrieve.embed_texts", lambda texts: [(1.0, 0.0)])
    monkeypatch.setattr(
        "src.retrieve.cross_encoder_scores",
        lambda query, passages: [0.0 for _ in passages],
    )
    sections = [(f"Near {index}", f"alpha note {index}.") for index in range(INITIAL_K)]
    sections.append(("Terms", "carbon 2040."))
    sections.append(("Extra", "quartz pebble."))
    vectors = [(math.cos(0.001 * index), math.sin(0.001 * index)) for index in range(INITIAL_K)]
    vectors.extend([(math.cos(1.2), math.sin(1.2)), (math.cos(2.0), math.sin(2.0))])
    weights = {"Terms": 0.9, "Near 0": 0.8, "Near 1": 0.7}
    seen = _mock_jev(monkeypatch, lambda hit: weights.get(hit.section_name, 0.01))
    client, _records = _index(tmp_path, sections, vectors)

    hits = hybrid_search(client, "carbon 2040", k=len(sections))

    pool = {hit.section_name for hit in seen[0]}
    assert "Terms" in pool
    assert "Extra" not in pool
    assert len(seen[0]) == INITIAL_K + 1
    assert next(hit for hit in seen[0] if hit.section_name == "Terms").body == "carbon 2040."
    assert len(hits) == JEV_KEEP
    assert [hit.section_name for hit in hits] == ["Terms", "Near 0", "Near 1"]
    assert hits[0].probability == pytest.approx(0.9)
    assert hits[0].score == pytest.approx(0.0)
    client.close()


def test_blank_query_and_empty_store_do_not_embed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> list[float]:
        raise AssertionError("embed and rerank should not be called")

    monkeypatch.setattr("src.retrieve.embed_texts", fail)
    monkeypatch.setattr("src.retrieve.cross_encoder_scores", fail)
    client = connect(tmp_path / "chroma")
    assert hybrid_search(client, "carbon", k=1) == []

    _index(tmp_path, [("Scope", "carbon 2040.")], [(1.0, 0.0)], client)
    assert hybrid_search(client, "   ", k=1) == []
    with pytest.raises(ValueError):
        hybrid_search(client, "carbon", k=0)
    client.close()
