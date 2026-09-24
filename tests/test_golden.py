import json
from pathlib import Path

from src.chunking import chunk_directory

GOLD_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "policy_rag_golden.json"
REQUIRED = {
    "id",
    "question",
    "question_type",
    "difficulty",
    "abstain",
    "expected_answer",
    "answer_must_include",
    "supporting_sections",
}


def _load() -> dict:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def _chunks() -> dict[str, list]:
    grouped: dict[str, list] = {}
    for chunk in chunk_directory():
        grouped.setdefault(chunk.section_id, []).append(chunk)
    return grouped


def test_golden_dataset_matches_chunk_sections() -> None:
    dataset = _load()
    grouped = _chunks()
    items = dataset["items"]
    assert dataset["schema_version"] == "1.0.0"
    assert [item["id"] for item in items] == [f"gold-{index:03d}" for index in range(1, len(items) + 1)]
    assert set(item["question_type"] for item in items) == set(dataset["question_types"])

    for item in items:
        assert REQUIRED <= item.keys()
        assert item["question"].endswith("?")
        assert item["difficulty"] in {"easy", "medium", "hard"}
        if item["abstain"]:
            assert item["supporting_sections"] == []
            assert item["expected_answer"] is None
            continue
        assert item["supporting_sections"]
        for section in item["supporting_sections"] + item.get("distractor_sections", []):
            chunks = grouped[section["section_id"]]
            assert chunks[0].document_id == section["document_id"]
            assert chunks[0].section_title == section["section_name"]
            pool = chunks
            if "chunk_index" in section:
                pool = [chunk for chunk in chunks if chunk.index == section["chunk_index"]]
                assert pool
            for quote in section["evidence"]:
                if quote["in"] == "section_name":
                    assert quote["text"] in pool[0].section_title
                else:
                    assert any(quote["text"] in chunk.body for chunk in pool)
