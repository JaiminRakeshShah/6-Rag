from pathlib import Path

from src.chunking import (
    MAX_TOKENS,
    OVERLAP_RATIO,
    _ENCODER,
    _table_end,
    Chunk,
    chunk_file,
    chunk_markdown,
    count_tokens,
    section_chunks,
)

DATA_MD = Path(__file__).resolve().parents[1] / "data_md"
FY25_TABLE = "\n".join(
    [
        "| **EMISSIONS** | **India** | **UK** |",
        "| --- | --- | --- |",
        "| Scope 1 | 1,265 | 0* |",
        "| Scope 2 | 6,344 | 2.90 |",
        "| Scope 3 | 29,836 | 850.05 |",
        "| **Total emissions** | **37,445** | **852.95** |",
    ]
)


def _paragraph(label: str, words: int) -> str:
    return " ".join(f"{label}{index}" for index in range(words)) + "."


def _document(sections: list[tuple[str, str]], title: str = "Q3 Report") -> str:
    parts = [f"# **{title}**", ""]
    for heading, body in sections:
        parts.extend([f"## **{heading}**", "", body, ""])
    return "\n".join(parts)


def _overlap_tokens(previous: str, following: str) -> int:
    encoded = _ENCODER.encode(previous, disallowed_special=())
    matched = 0
    for size in range(1, len(encoded) + 1):
        tail = _ENCODER.decode(encoded[-size:])
        if following.startswith(tail):
            matched = size
    return matched


def _tables(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    tables: list[str] = []
    index = 0
    while index < len(lines):
        end = _table_end(lines, index)
        if end is None:
            index += 1
            continue
        tables.append("\n".join(lines[index:end]).strip())
        index = end
    return tables


def test_short_section_stays_whole() -> None:
    markdown = _document([("2024 Revenue", "The revenue dropped by 10 percent.")])
    chunks = chunk_markdown(markdown, document_id="Q3_Report")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.tokens < 400
    assert chunk.tokens <= MAX_TOKENS
    assert chunk.body == "The revenue dropped by 10 percent."
    assert chunk.text == (
        "[Doc: Q3 Report | Section: 2024 Revenue] The revenue dropped by 10 percent."
    )
    assert section_chunks(chunks, chunk.section_id) == [chunk]


def test_long_prose_splits_inside_the_token_window_with_overlap() -> None:
    body = "\n\n".join(_paragraph(f"p{index}", 90) for index in range(8))
    chunks = chunk_markdown(_document([("2024 Revenue", body)]), document_id="Q3_Report")

    assert len(chunks) >= 2
    assert all(chunk.tokens <= MAX_TOKENS for chunk in chunks)
    assert all(not chunk.table for chunk in chunks)
    assert len({chunk.section_id for chunk in chunks}) == 1
    for previous, following in zip(chunks, chunks[1:]):
        overlap = _overlap_tokens(previous.body, following.body)
        ratio = overlap / previous.tokens
        assert OVERLAP_RATIO - 0.03 <= ratio <= OVERLAP_RATIO + 0.03


def test_sentence_longer_than_the_limit_is_cut_on_tokens() -> None:
    sentence = " ".join(f"w{index}" for index in range(800))
    assert count_tokens(sentence) > MAX_TOKENS
    chunks = chunk_markdown(_document([("Notes", sentence)]), document_id="Q3_Report")

    assert len(chunks) >= 2
    assert all(chunk.tokens <= MAX_TOKENS for chunk in chunks)
    for previous, following in zip(chunks, chunks[1:]):
        ratio = _overlap_tokens(previous.body, following.body) / previous.tokens
        assert 0.10 <= ratio <= 0.15


def test_paragraph_over_the_limit_splits_on_sentences() -> None:
    sentences = " ".join(_paragraph(f"s{index}", 40) for index in range(20))
    assert "\n\n" not in sentences
    assert count_tokens(sentences) > MAX_TOKENS
    chunks = chunk_markdown(
        _document([("Notes", sentences)]),
        document_id="Q3_Report",
    )

    assert len(chunks) >= 2
    assert all(chunk.tokens <= MAX_TOKENS for chunk in chunks)
    assert all(chunk.section_title == "Notes" for chunk in chunks)


def test_oversized_table_stays_one_chunk() -> None:
    rows = ["| Item | Value |", "| --- | --- |"]
    for index in range(40):
        rows.append("| item-" + str(index) + " | " + " ".join(["gamma"] * 20) + " |")
    table = "\n".join(rows)
    assert count_tokens(table) > MAX_TOKENS

    chunks = chunk_markdown(_document([("Numbers", table)]), document_id="Q3_Report")

    assert len(chunks) == 1
    assert chunks[0].table
    assert chunks[0].tokens > MAX_TOKENS
    assert chunks[0].body == table


def test_table_in_a_long_section_is_not_sliced() -> None:
    rows = ["| Item | Value |", "| --- | --- |"]
    for index in range(30):
        rows.append("| item-" + str(index) + " | " + " ".join(["delta"] * 12) + " |")
    table = "\n".join(rows)
    intro = "\n\n".join(_paragraph("before", 80) for _ in range(4))
    outro = "\n\n".join(_paragraph("after", 80) for _ in range(4))
    chunks = chunk_markdown(
        _document([("Numbers", f"{intro}\n\n{table}\n\n{outro}")]),
        document_id="Q3_Report",
    )

    tables = [chunk for chunk in chunks if chunk.table]
    assert len(tables) == 1
    assert tables[0].body == table
    assert all(table not in chunk.body for chunk in chunks if chunk is not tables[0])
    assert all(chunk.tokens <= MAX_TOKENS for chunk in chunks if not chunk.table)


def test_small_section_keeps_its_table_with_the_prose() -> None:
    body = "A short note.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"
    chunks = chunk_markdown(_document([("Small", body)]), document_id="Q3_Report")

    assert len(chunks) == 1
    assert not chunks[0].table
    assert "| 1 | 2 |" in chunks[0].body
    assert "A short note." in chunks[0].body


def test_matched_chunk_expands_to_the_whole_section() -> None:
    long_body = "\n\n".join(_paragraph(f"rev{index}", 90) for index in range(8))
    markdown = _document(
        [
            ("2024 Revenue", long_body),
            ("Outlook", "Growth continues."),
        ]
    )
    chunks = chunk_markdown(markdown, document_id="Q3_Report")
    matched = chunks[0]
    expanded = section_chunks(chunks, matched.section_id)

    assert len(expanded) >= 2
    assert expanded == [chunk for chunk in chunks if chunk.section_id == matched.section_id]
    assert all(chunk.section_title == "2024 Revenue" for chunk in expanded)
    outlook = [chunk for chunk in chunks if chunk.section_title == "Outlook"]
    assert section_chunks(chunks, outlook[0].section_id) == outlook
    assert matched.section_id not in {chunk.section_id for chunk in outlook}


def test_policy_markdown_keeps_tables_and_token_limits() -> None:
    titles = {
        "Carbon_New_2040": "Carbon Reduction Plan",
        "Carbon_Old_2050": "Carbon Reduction Plan (v1.0, superseded)",
        "Envi_2040-1": "Environmental Sustainability Policy",
        "Water-Management-Policy": "Water Management Policy",
    }
    seen_titles: set[str] = set()
    for path in sorted(DATA_MD.glob("*.md")):
        chunks = chunk_file(path)
        assert chunks
        seen_titles.add(path.stem)
        assert chunks[0].document_title == titles[path.stem]
        _assert_measured(chunks)
        source = path.read_text(encoding="utf-8")
        tables = _tables(source)
        assert tables
        for table in tables:
            holders = [chunk for chunk in chunks if table in chunk.body]
            assert len(holders) == 1
        if path.stem == "Carbon_New_2040":
            holders = [chunk for chunk in chunks if FY25_TABLE in chunk.body]
            assert len(holders) == 1

    assert seen_titles == set(titles)


def _assert_measured(chunks: list[Chunk]) -> None:
    assert chunks
    for chunk in chunks:
        assert chunk.text == (
            f"[Doc: {chunk.document_title} | Section: {chunk.section_title}] {chunk.body}"
        )
        assert chunk.tokens == count_tokens(chunk.body)
        if not chunk.table:
            assert chunk.tokens <= MAX_TOKENS
        if chunk.table:
            assert "| --- |" in chunk.body or "|---|" in chunk.body

    for section_id in {chunk.section_id for chunk in chunks}:
        group = section_chunks(chunks, section_id)
        prose = [chunk for chunk in group if not chunk.table]
        for previous, following in zip(prose, prose[1:]):
            if previous.index + 1 != following.index:
                continue
            overlap = _overlap_tokens(previous.body, following.body)
            ratio = overlap / previous.tokens
            assert 0.10 <= ratio <= 0.15
