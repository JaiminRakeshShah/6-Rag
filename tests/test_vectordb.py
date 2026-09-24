from pathlib import Path

import sqlite3

import pytest

from src.chunking import DATA_MD, chunk_directory, chunk_markdown
from src.embed import EmbeddedChunk
from src.vectordb import (
    COLLECTION,
    build_records,
    connect,
    parse_document,
    replace_index,
    search,
)


def _markdown_by_document() -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(DATA_MD.glob("*.md"))}


def _embedded_corpus() -> list[EmbeddedChunk]:
    return [EmbeddedChunk(chunk, (1.0, 0.0)) for chunk in chunk_directory()]


def test_parse_document_reads_publication_review_and_version() -> None:
    published = parse_document("Publication date: 10 October 2025\n")
    assert published.document_date == "2025-10-10"
    assert published.date_source == "publication"
    assert published.version is None
    assert published.superseded is False

    archived = parse_document(
        "DOCUMENT STATUS: SUPERSEDED | Version 1.0 | Publication date: 12 March 2023\n"
    )
    assert archived.document_date == "2023-03-12"
    assert archived.version == "1.0"
    assert archived.superseded is True

    reviewed = parse_document("Review Date – 10<sup>th</sup> March 2026 Last Review – 1<sup>st</sup> April 2025\n")
    assert reviewed.document_date == "2026-03-10"
    assert reviewed.date_source == "review"
    assert reviewed.version is None


def test_corpus_records_keep_spans_and_leave_page_numbers_empty() -> None:
    markdown = _markdown_by_document()
    records = build_records(_embedded_corpus(), markdown)

    assert len(records) == 55
    assert {record.page_no for record in records} == {None}
    by_document = {record.document_id: record for record in records}
    assert by_document["Carbon_New_2040"].document_date == "2025-10-10"
    assert by_document["Carbon_New_2040"].version is None
    assert by_document["Carbon_Old_2050"].document_date == "2023-03-12"
    assert by_document["Carbon_Old_2050"].version == "1.0"
    assert by_document["Carbon_Old_2050"].superseded is True
    assert by_document["Envi_2040-1"].document_date == "2026-03-10"
    assert by_document["Envi_2040-1"].date_source == "review"
    assert by_document["Water-Management-Policy"].document_date == "2026-08-10"

    for record in records:
        text = markdown[record.document_id]
        assert record.start_span is not None and record.end_span is not None
        parts = [part for part in record.body.split("\n\n") if part]
        assert text[record.start_span : record.start_span + len(parts[0])] == parts[0]
        assert text[record.end_span - len(parts[-1]) : record.end_span] == parts[-1]
        assert record.chunk_id == f"{record.parent_id}:{record.chunk_index}"
        assert record.filename == f"{record.document_id}.md"


def test_missing_source_leaves_span_and_date_empty() -> None:
    chunk = chunk_markdown("# **Policy**\n\n## **Scope**\n\nHello.\n", document_id="Policy")[0]
    record = build_records([EmbeddedChunk(chunk, (0.1, 0.2))], {})[0]

    assert record.start_span is None
    assert record.end_span is None
    assert record.document_date is None
    assert record.page_no is None
    assert record.section_name == "Scope"
    assert record.body == "Hello."


def test_search_returns_the_nearest_chunk_with_its_text(tmp_path: Path) -> None:
    markdown = "# **Policy**\n\n## **Scope**\n\nAlpha text.\n\n## **Terms**\n\nBeta text.\n"
    chunks = chunk_markdown(markdown, document_id="Policy")
    records = build_records(
        [
            EmbeddedChunk(chunks[0], (1.0, 0.0)),
            EmbeddedChunk(chunks[1], (0.0, 1.0)),
        ],
        {"Policy": markdown},
    )
    client = connect(tmp_path / "chroma")
    replace_index(client, records, model_id="embeddinggemma")

    hits = search(client, [1.0, 0.0], k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == records[0].chunk_id
    assert hits[0].section_name == "Scope"
    assert hits[0].body == "Alpha text."
    assert hits[0].page_no is None
    assert hits[0].distance == pytest.approx(0.0)
    assert hits[0].filename == "Policy.md"
    stored = client.get_collection(COLLECTION, embedding_function=None).get(
        where={"section_name": "Terms"},
        include=["documents"],
    )
    assert stored["documents"] == ["Beta text."]
    sql = sqlite3.connect(tmp_path / "chunks.sqlite")
    sql.row_factory = sqlite3.Row
    row = sql.execute("SELECT body, page_no FROM chunks WHERE section_name = 'Terms'").fetchone()
    assert row["body"] == "Beta text."
    assert row["page_no"] is None
    assert sql.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    sql.close()
    assert client.get_collection(COLLECTION, embedding_function=None).metadata["model_id"] == (
        "embeddinggemma"
    )
    client.close()


def test_replace_index_drops_a_different_dimension(tmp_path: Path) -> None:
    markdown = "# **Policy**\n\n## **Scope**\n\nAlpha text.\n"
    chunk = chunk_markdown(markdown, document_id="Policy")[0]
    client = connect(tmp_path / "chroma")
    replace_index(
        client,
        build_records([EmbeddedChunk(chunk, (1.0, 0.0))], {"Policy": markdown}),
        model_id="embeddinggemma",
    )
    with pytest.raises(ValueError):
        search(client, [1.0, 0.0, 0.0], k=1)

    replace_index(
        client,
        build_records([EmbeddedChunk(chunk, (0.0, 1.0, 0.0))], {"Policy": markdown}),
        model_id="other-model",
    )
    hits = search(client, [0.0, 1.0, 0.0], k=1)
    assert hits[0].body == "Alpha text."
    assert client.get_collection(COLLECTION, embedding_function=None).count() == 1
    client.close()


def test_empty_replace_clears_the_store(tmp_path: Path) -> None:
    markdown = "# **Policy**\n\n## **Scope**\n\nAlpha text.\n"
    chunk = chunk_markdown(markdown, document_id="Policy")[0]
    client = connect(tmp_path / "chroma")
    replace_index(
        client,
        build_records([EmbeddedChunk(chunk, (1.0, 0.0))], {"Policy": markdown}),
        model_id="embeddinggemma",
    )
    replace_index(client, [], model_id="embeddinggemma")

    assert search(client, [1.0, 0.0], k=1) == []
    assert client.list_collections() == []
    sql = sqlite3.connect(tmp_path / "chunks.sqlite")
    assert sql.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    sql.close()
    client.close()


def test_mixed_dimensions_are_rejected(tmp_path: Path) -> None:
    markdown = "# **Policy**\n\n## **Scope**\n\nAlpha.\n\n## **Terms**\n\nBeta.\n"
    chunks = chunk_markdown(markdown, document_id="Policy")
    records = build_records(
        [EmbeddedChunk(chunks[0], (1.0, 0.0)), EmbeddedChunk(chunks[1], (1.0, 0.0, 0.0))],
        {"Policy": markdown},
    )
    client = connect(tmp_path / "chroma")
    with pytest.raises(ValueError):
        replace_index(client, records, model_id="embeddinggemma")
    client.close()
