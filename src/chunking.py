"""Token-aware Markdown chunking for the policy documents in data_md/.

A section at or under 512 tokens stays whole. A longer section splits on
blank lines, then on ". ", until each prose chunk fits. Tables are never
split. Every chunk is prefixed with the document title and section heading
so it can be embedded on its own. Chunks from the same section share a
section_id; section_chunks() returns that whole set for a reranker.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

MAX_TOKENS = 512
OVERLAP_RATIO = 0.125

_ENCODER = tiktoken.get_encoding("cl100k_base")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_SEPARATOR = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")

ROOT = Path(__file__).resolve().parents[1]
DATA_MD = ROOT / "data_md"


def count_tokens(text: str) -> int:
    """Count tokens with cl100k_base. The 512 limit uses this same count."""
    return len(_ENCODER.encode(text, disallowed_special=()))


def _overlap_count(tokens: int) -> int:
    """How many tokens to copy from the end of one chunk onto the next.

    That is about 12.5% of ``tokens``, the middle of the 10–15% overlap range.
    """
    return int(tokens * OVERLAP_RATIO + 0.5)


def _tail(text: str, tokens: int) -> str:
    """Return the last ``tokens`` tokens of ``text`` as a string."""
    if tokens <= 0:
        return ""
    ids = _ENCODER.encode(text, disallowed_special=())
    if tokens >= len(ids):
        return text
    return _ENCODER.decode(ids[-tokens:])


def _plain(text: str) -> str:
    """Turn a Markdown heading into plain text by dropping bold markers."""
    text = re.sub(r"^\*\*(.*)\*\*$", r"\1", text.strip())
    text = text.replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Chunk:
    """One retrieval piece of a document.

    ``body`` is the chunk itself. ``text`` is what gets embedded: the document
    title and section heading, then the body. ``tokens`` counts the body only.
    ``table`` is true when this chunk is a whole Markdown table.
    """

    document_id: str
    document_title: str
    section_id: str
    section_title: str
    index: int
    body: str
    text: str
    tokens: int
    table: bool


def section_chunks(chunks: list[Chunk], section_id: str) -> list[Chunk]:
    """Return every chunk in one section, in order, for the reranker.

    After a search hit, pass the hit's ``section_id`` here. The list includes
    the hit and the other chunks from that same heading.
    """
    return [chunk for chunk in chunks if chunk.section_id == section_id]


def chunk_file(path: Path) -> list[Chunk]:
    """Read one Markdown file and chunk it. The document id is the file stem."""
    return chunk_markdown(path.read_text(encoding="utf-8"), document_id=path.stem)


def chunk_directory(directory: Path = DATA_MD) -> list[Chunk]:
    """Chunk every ``.md`` file in ``directory``, in filename order."""
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_file(path))
    return chunks


def chunk_markdown(markdown: str, document_id: str) -> list[Chunk]:
    """Split one Markdown document into chunks.

    ``document_id`` is stored on each chunk and used to build ``section_id``.
    The first ``#`` heading becomes the document title in the context prefix.
    """
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    document_title, sections = _sections(markdown)
    if not document_title:
        document_title = document_id

    chunks: list[Chunk] = []
    for section_index, section in enumerate(sections):
        section_id = f"{document_id}#{section_index}"
        for index, (body, is_table) in enumerate(_section_pieces(section.blocks)):
            chunks.append(
                Chunk(
                    document_id=document_id,
                    document_title=document_title,
                    section_id=section_id,
                    section_title=section.title,
                    index=index,
                    body=body,
                    text=f"[Doc: {document_title} | Section: {section.title}] {body}",
                    tokens=count_tokens(body),
                    table=is_table,
                )
            )
    return chunks


@dataclass
class _Section:
    """One heading and the prose paragraphs and tables under it."""

    title: str
    blocks: list[tuple[str, str]]


def _sections(markdown: str) -> tuple[str, list[_Section]]:
    """Split Markdown into the document title and its sections.

    The first ``#`` heading is the title. Text before the next heading is
    called Preamble. Each later heading starts a new section. A heading with
    no body is skipped.
    """
    title = ""
    current = "Preamble"
    buf: list[str] = []
    sections: list[_Section] = []

    def flush() -> None:
        """Save the lines gathered since the previous heading, if any."""
        blocks = _blocks("\n".join(buf))
        buf.clear()
        if blocks:
            sections.append(_Section(current, blocks))

    for line in markdown.split("\n"):
        heading = _HEADING.match(line)
        if heading is None:
            buf.append(line)
            continue
        flush()
        level = len(heading.group(1))
        text = _plain(heading.group(2))
        if level == 1 and not title:
            title = text
            current = "Preamble"
        else:
            current = text
    flush()
    return title, sections


def _blocks(body: str) -> list[tuple[str, str]]:
    """Split a section body into ``("prose", text)`` and ``("table", text)`` blocks."""
    lines = body.split("\n")
    blocks: list[tuple[str, str]] = []
    prose: list[str] = []

    def flush_prose() -> None:
        """Turn buffered lines into paragraphs, splitting on blank lines."""
        text = "\n".join(prose).strip()
        prose.clear()
        if not text:
            return
        for paragraph in re.split(r"\n[ \t]*\n", text):
            paragraph = paragraph.strip()
            if paragraph:
                blocks.append(("prose", paragraph))

    index = 0
    while index < len(lines):
        table_end = _table_end(lines, index)
        if table_end is None:
            prose.append(lines[index])
            index += 1
            continue
        flush_prose()
        blocks.append(("table", "\n".join(lines[index:table_end]).strip()))
        index = table_end
    flush_prose()
    return blocks


def _table_end(lines: list[str], start: int) -> int | None:
    """Return the index just after a Markdown table that starts at ``start``.

    Returns None when that line is not a table. A table is a run of ``|``
    lines that includes a ``| --- |`` separator.
    """
    if not lines[start].lstrip().startswith("|"):
        return None
    end = start
    saw_separator = False
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        if _SEPARATOR.match(lines[end]):
            saw_separator = True
        end += 1
    if saw_separator and end > start + 1:
        return end
    return None


def _section_pieces(blocks: list[tuple[str, str]]) -> list[tuple[str, bool]]:
    """Turn one section into ``(body, is_table)`` pieces.

    A section of 512 tokens or fewer stays one piece, table included. A longer
    section keeps each table as its own piece and splits the surrounding prose.
    """
    whole = "\n\n".join(text for _kind, text in blocks)
    only_table = len(blocks) == 1 and blocks[0][0] == "table"
    if count_tokens(whole) <= MAX_TOKENS:
        return [(whole, only_table)]

    pieces: list[tuple[str, bool]] = []
    prose: list[str] = []

    def flush_prose() -> None:
        """Split the buffered paragraphs and append each piece."""
        if not prose:
            return
        for part in _split_prose("\n\n".join(prose)):
            pieces.append((part, False))
        prose.clear()

    for kind, text in blocks:
        if kind == "table":
            flush_prose()
            pieces.append((text, True))
            continue
        prose.append(text)
    flush_prose()
    return pieces


@dataclass
class _Atom:
    """A prose piece small enough to place into a chunk.

    ``joiner`` is inserted before ``text`` when this piece follows other text.
    ``final`` means the piece is already a 512-token window and must not be
    merged with its neighbors.
    """

    text: str
    joiner: str
    final: bool = False


def _split_prose(text: str) -> list[str]:
    """Return ``text`` unchanged when it fits, otherwise pieces of at most 512 tokens."""
    if count_tokens(text) <= MAX_TOKENS:
        return [text]
    return _merge(_atoms(text, ""))


def _atoms(text: str, joiner: str) -> list[_Atom]:
    """Break text into packable pieces: paragraphs, then sentences, then tokens.

    ``joiner`` is the separator used when the first piece is appended to
    whatever came before it.
    """
    if count_tokens(text) <= MAX_TOKENS:
        return [_Atom(text, joiner)]
    paragraphs = _paragraphs(text)
    if len(paragraphs) > 1:
        atoms: list[_Atom] = []
        for index, paragraph in enumerate(paragraphs):
            atoms.extend(_atoms(paragraph, "\n\n" if index else joiner))
        return atoms
    sentences = _sentences(text)
    if len(sentences) > 1:
        atoms = []
        for index, sentence in enumerate(sentences):
            atoms.extend(_atoms(sentence, " " if index else joiner))
        return atoms
    return [_Atom(window, "", final=True) for window in _token_windows(text)]


def _paragraphs(text: str) -> list[str]:
    """Split text on blank lines and drop empty pieces."""
    return [part.strip() for part in re.split(r"\n[ \t]*\n", text) if part.strip()]


def _sentences(text: str) -> list[str]:
    """Split text on a period followed by a space."""
    return [part for part in re.split(r"(?<=\.) ", text) if part]


def _token_windows(text: str) -> list[str]:
    """Cut text with no paragraph or sentence break into 512-token windows.

    Each window after the first repeats about 12.5% of the previous window.
    """
    ids = _ENCODER.encode(text, disallowed_special=())
    overlap = _overlap_count(MAX_TOKENS)
    step = max(1, MAX_TOKENS - overlap)
    windows: list[str] = []
    start = 0
    while start < len(ids):
        windows.append(_ENCODER.decode(ids[start : start + MAX_TOKENS]))
        if start + MAX_TOKENS >= len(ids):
            break
        start += step
    return windows


def _merge(atoms: list[_Atom]) -> list[str]:
    """Pack atoms into chunks of at most 512 tokens, with a 12.5% overlap.

    When the next atom does not fit, the chunk is closed and its tail is
    carried onto the next one. A single sentence that still does not fit with
    that tail is placed on its own.
    """
    chunks: list[str] = []
    current = ""
    fresh = False
    index = 0
    while index < len(atoms):
        atom = atoms[index]
        if atom.final:
            if fresh:
                chunks.append(current)
            chunks.append(atom.text)
            current = _tail(atom.text, _overlap_count(count_tokens(atom.text)))
            fresh = False
            index += 1
            continue

        base = current
        addition = atom.text if not base else atom.joiner + atom.text
        if count_tokens(base + addition) <= MAX_TOKENS:
            current = base + addition
            fresh = True
            index += 1
            continue

        if fresh:
            chunks.append(current)
            current = _tail(current, _overlap_count(count_tokens(current)))
            fresh = False
            continue

        finer = _sentences(atom.text)
        if len(finer) > 1:
            replacement = []
            for part_index, sentence in enumerate(finer):
                joiner = atom.joiner if part_index == 0 else " "
                replacement.extend(_atoms(sentence, joiner))
            atoms = atoms[:index] + replacement + atoms[index + 1 :]
            continue

        current = atom.text
        fresh = True
        index += 1

    if fresh:
        chunks.append(current)
    return chunks
