import re
from collections import Counter
from pathlib import Path

import pymupdf

from src.convert_pdfs import convert_all

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_each_pdf_becomes_nonempty_markdown(tmp_path: Path) -> None:
    written = convert_all(DATA_DIR, tmp_path)
    stems = sorted(path.stem for path in DATA_DIR.glob("*.pdf"))

    assert stems
    assert [path.stem for path in written] == stems
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert path.suffix == ".md"
        assert text.strip()
        assert "<mark>" not in text
        assert "picture text" not in text
        body, _, _about = text.partition("About Coforge")
        assert "© 2026 Coforge" not in body


def test_line_break_hyphens_and_review_dates(tmp_path: Path) -> None:
    written = {path.stem: path.read_text(encoding="utf-8") for path in convert_all(DATA_DIR, tmp_path)}

    envi = written["Envi_2040-1"]
    assert "third-party" in envi
    assert "sub-targets" in envi
    assert "10<sup>th</sup>" in envi
    assert "1<sup>st</sup>" in envi

    water = written["Water-Management-Policy"]
    assert "third-party" in water
    assert "water-stressed" in water
    assert "10<sup>th</sup>" in water


def test_fy25_columns_and_carbon_tokens(tmp_path: Path) -> None:
    written = {path.stem: path.read_text(encoding="utf-8") for path in convert_all(DATA_DIR, tmp_path)}
    carbon = written["Carbon_New_2040"]

    assert "| Scope 1 | 1,265 | 0* |" in carbon
    assert "| Scope 2 | 6,344 | 2.90 |" in carbon
    assert "| Scope 3 | 29,836 | 850.05 |" in carbon
    assert "| **Total emissions** | **37,445** | **852.95** |" in carbon
    assert "|Scope 1|413|" in carbon
    assert "|Scope 2|22.38|" in carbon

    pdf = pymupdf.open(DATA_DIR / "Carbon_New_2040.pdf")
    source = "\n".join(page.get_text("text") for page in pdf)
    assert _tokens(source) == _tokens(carbon)


def _tokens(text: str) -> Counter[str]:
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[|*_`#]", " ", text)
    return Counter(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower()))
