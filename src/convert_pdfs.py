"""Convert PDFs in data/ to Markdown in data_md/."""

import re
from pathlib import Path

import pymupdf
import pymupdf4llm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "data_md"

_MARK = re.compile(r"</?mark>")
_PICTURE_START = re.compile(r"<!-- Start of picture text -->\s*")
_PICTURE_END = re.compile(r"<br>\s*<!-- End of picture text -->|<!-- End of picture text -->")
_LIST_ITEM = re.compile(r"^(\s*[-*]\s+)(.*)$")


def line_break_hyphens(pdf_path: Path) -> list[tuple[str, str]]:
    """Pairs of (joined word, hyphenated word) split across PDF lines."""
    pairs: list[tuple[str, str]] = []
    document = pymupdf.open(pdf_path)
    for page in document:
        lines = page.get_text("text").splitlines()
        for line, nxt in zip(lines, lines[1:]):
            broken = re.search(r"([A-Za-z]+)-\s*$", line)
            continued = re.match(r"([A-Za-z]+)", nxt.lstrip())
            if not broken or not continued:
                continue
            left, right = broken.group(1), continued.group(1)
            pairs.append((left + right, f"{left}-{right}"))
    return pairs


def restore_line_break_hyphens(markdown: str, pdf_path: Path) -> str:
    for joined, hyphenated in line_break_hyphens(pdf_path):
        markdown = re.sub(
            rf"\b{re.escape(joined)}\b",
            hyphenated,
            markdown,
        )
    return markdown


def drop_stray_bullets(markdown: str) -> str:
    cleaned: list[str] = []
    for line in markdown.splitlines():
        match = _LIST_ITEM.match(line)
        if match and "•" in match.group(2):
            body = re.sub(r"\s*•\s*", " ", match.group(2))
            body = re.sub(r" {2,}", " ", body).strip()
            line = match.group(1) + body
        cleaned.append(line)
    return "\n".join(cleaned)


def _ordinals(text: str) -> str:
    return re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1<sup>\2</sup>", text)


def restore_review_dates(markdown: str, pdf_path: Path) -> str:
    if "Review Date" in markdown:
        return markdown
    document = pymupdf.open(pdf_path)
    lines = [
        line.strip()
        for page in document
        for line in page.get_text("text").splitlines()
        if line.strip().startswith(("Review Date", "Last Review"))
    ]
    if not lines:
        return markdown
    paragraph = " ".join(_ordinals(line) for line in lines)
    heading = re.search(r"^# .*$", markdown, flags=re.M)
    if heading is None:
        return paragraph + "\n\n" + markdown
    return markdown[: heading.end()] + "\n\n" + paragraph + markdown[heading.end() :]


def restore_rights_notice(markdown: str, pdf_path: Path) -> str:
    notice = "© 2026 Coforge. All rights reserved."
    if notice in markdown:
        return markdown
    document = pymupdf.open(pdf_path)
    source = "\n".join(page.get_text("text") for page in document)
    if notice not in source:
        return markdown
    return markdown.rstrip() + "\n\n" + notice + "\n"


def _column_bounds(header: list[tuple]) -> list[tuple[str, float]]:
    named = [(word[4], word[0]) for word in header if word[4] in {"EMISSIONS", "India", "UK"}]
    named.sort(key=lambda item: item[1])
    bounds: list[tuple[str, float]] = []
    for index, (label, x_pos) in enumerate(named):
        if index + 1 < len(named):
            edge = (x_pos + named[index + 1][1]) / 2
        else:
            edge = float("inf")
        bounds.append((label, edge))
    return bounds


def _line_cells(line: list[tuple], bounds: list[tuple[str, float]]) -> dict[str, str]:
    cells = {label: [] for label, _edge in bounds}
    for word in line:
        for label, edge in bounds:
            if word[0] < edge:
                cells[label].append(word[4])
                break
    return {label: " ".join(words) for label, words in cells.items()}


def fy25_table(pdf_path: Path) -> str | None:
    """Build the FY25 India/UK table from word positions.

    Layout mode scrambles find_tables() cell text, so the cells come from
    the page's words instead.
    """
    document = pymupdf.open(pdf_path)
    for page in document:
        lines: list[list[tuple]] = []
        for word in page.get_text("words"):
            if lines and abs(word[1] - lines[-1][0][1]) < 2:
                lines[-1].append(word)
            else:
                lines.append([word])
        header_at = next(
            (
                index
                for index, line in enumerate(lines)
                if {word[4] for word in line} >= {"EMISSIONS", "India", "UK"}
            ),
            None,
        )
        if header_at is None:
            continue
        bounds = _column_bounds(lines[header_at])
        if [label for label, _edge in bounds] != ["EMISSIONS", "India", "UK"]:
            continue
        body: list[tuple[str, str, str]] = []
        for line in lines[header_at + 1 :]:
            cells = _line_cells(line, bounds)
            label = cells["EMISSIONS"]
            if label not in {"Scope 1", "Scope 2", "Scope 3", "Total emissions"}:
                if body:
                    break
                continue
            body.append((label, cells["India"], cells["UK"]))
            if label == "Total emissions":
                break
        if [row[0] for row in body] != ["Scope 1", "Scope 2", "Scope 3", "Total emissions"]:
            continue
        rendered = [
            "| **EMISSIONS** | **India** | **UK** |",
            "| --- | --- | --- |",
        ]
        for label, india, uk in body:
            if label == "Total emissions":
                rendered.append(f"| **{label}** | **{india}** | **{uk}** |")
            else:
                rendered.append(f"| {label} | {india} | {uk} |")
        return "\n".join(rendered)
    return None


_FY25_TABLE = re.compile(
    r"\|.*Reporting Year: 1 April 2024 – 31 March 2025.*\n(?:\|.*\n)+"
)


def repair_fy25_table(markdown: str, table: str | None) -> str:
    if table is None or not _FY25_TABLE.search(markdown):
        return markdown
    replacement = (
        "**Reporting Year: 1 April 2024 – 31 March 2025**\n\n"
        "**Current year emissions – Total (tCO2e)**\n\n"
        f"{table}\n"
    )
    return _FY25_TABLE.sub(replacement, markdown, count=1)


def clean_markdown(markdown: str, pdf_path: Path, current_year: str | None) -> str:
    markdown = _MARK.sub("", markdown)
    markdown = _PICTURE_START.sub("", markdown)
    markdown = _PICTURE_END.sub("", markdown)
    markdown = drop_stray_bullets(markdown)
    markdown = restore_line_break_hyphens(markdown, pdf_path)
    markdown = restore_review_dates(markdown, pdf_path)
    markdown = restore_rights_notice(markdown, pdf_path)
    return repair_fy25_table(markdown, current_year)


def convert_pdf(pdf_path: Path, output_dir: Path) -> Path:
    # Read table cells before to_markdown; that call rewrites page text.
    current_year = fy25_table(pdf_path)
    markdown = pymupdf4llm.to_markdown(
        str(pdf_path),
        header=False,
        footer=False,
    )
    markdown = clean_markdown(markdown, pdf_path, current_year)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{pdf_path.stem}.md"
    destination.write_text(markdown, encoding="utf-8")
    return destination


def convert_all(data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    pdfs = sorted(path for path in data_dir.glob("*.pdf") if path.is_file())
    return [convert_pdf(pdf, output_dir) for pdf in pdfs]


def main() -> None:
    written = convert_all()
    if not written:
        raise SystemExit(f"No PDFs found in {DATA_DIR}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
