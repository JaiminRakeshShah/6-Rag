"""Permanent Chroma store for embeddings, plus a SQLite copy of the chunk text.

Chroma holds each embedding. ``data/chunks.sqlite`` holds the same chunks in
ordinary tables: ``documents`` and ``chunks``, including ``chunks.body``.
``chunks_fts`` is an FTS5 index of that body, rebuilt on each replace, and
``bm25_scores`` reads it.

Chroma rejects null metadata, so empty fields are omitted there. SQLite stores
those fields as NULL. ``page_no`` stays null. ``parent_id`` is the section id
shared by the chunks of one heading.
"""

import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from src.chunking import DATA_MD, chunk_directory
from src.embed import EmbeddedChunk, embed_chunks

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "chroma"
SQL_PATH = ROOT / "data" / "chunks.sqlite"
COLLECTION = "chunks"
_DEFAULT_MODEL_ID = "embeddinggemma"

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


@dataclass(frozen=True)
class DocumentFacts:
    """Date and version read from one Markdown source."""

    document_date: str | None
    date_source: str | None
    version: str | None
    superseded: bool


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk, its metadata, and the embedding of ``embedded_text``."""

    chunk_id: str
    document_id: str
    filename: str
    title: str
    page_no: int | None
    section_name: str
    parent_id: str
    start_span: int | None
    end_span: int | None
    chunk_index: int
    body: str
    embedded_text: str
    token_count: int
    is_table: bool
    document_date: str | None
    date_source: str | None
    version: str | None
    superseded: bool
    vector: tuple[float, ...]


@dataclass(frozen=True)
class SearchHit:
    """A nearest-chunk result. ``distance`` is cosine distance, lower is nearer."""

    chunk_id: str
    filename: str
    title: str
    page_no: int | None
    section_name: str
    parent_id: str
    start_span: int | None
    end_span: int | None
    document_date: str | None
    date_source: str | None
    version: str | None
    superseded: bool
    body: str
    distance: float


def connect(path: Path = DEFAULT_PATH) -> ClientAPI:
    """Open a persistent Chroma client, creating the directory if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def parse_document(markdown: str) -> DocumentFacts:
    """Read the document date and version from the Markdown itself.

    A publication date wins over a review date. ``version`` stays null when
    the source never states one. ``superseded`` is true only for a document
    that labels itself ``DOCUMENT STATUS: SUPERSEDED``.
    """
    superseded = re.search(r"DOCUMENT STATUS:\s*SUPERSEDED", markdown) is not None
    version_match = re.search(r"\bVersion\s+(\d+(?:\.\d+)*)\b", markdown)
    version = version_match.group(1) if version_match else None
    publication = re.search(
        r"Publication date:\**\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
        markdown,
        flags=re.IGNORECASE,
    )
    if publication:
        return DocumentFacts(_iso_date(*publication.groups()), "publication", version, superseded)
    review = re.search(
        r"Review Date\s*[–—-]\s*(\d{1,2})(?:<sup>[^<]*</sup>)?\s+([A-Za-z]+)\s+(\d{4})",
        markdown,
    )
    if review:
        return DocumentFacts(_iso_date(*review.groups()), "review", version, superseded)
    return DocumentFacts(None, None, version, superseded)


def locate_span(markdown: str, body: str, cursor: int) -> tuple[int, int] | None:
    """Return the character span of ``body`` in ``markdown``, searching from ``cursor``.

    Chunk bodies join paragraphs with a blank line, while the file keeps its
    own whitespace between those paragraphs. The span runs from the first
    paragraph to the last, original whitespace included.
    """
    start: int | None = None
    end: int | None = None
    position = cursor
    for part in body.split("\n\n"):
        if not part:
            continue
        at = markdown.find(part, position)
        if at < 0:
            return None
        if start is None:
            start = at
        end = at + len(part)
        position = at + 1
    if start is None or end is None:
        return None
    return start, end


def build_records(
    embedded: Sequence[EmbeddedChunk],
    markdown_by_document: dict[str, str],
) -> list[ChunkRecord]:
    """Pair each embedding with the metadata of its source chunk.

    ``chunk_id`` is ``{parent_id}:{chunk_index}``. ``parent_id`` is the
    section id. ``page_no`` is always null.
    """
    facts: dict[str, DocumentFacts] = {}
    cursors: dict[str, int] = {}
    records: list[ChunkRecord] = []
    for item in embedded:
        chunk = item.chunk
        document_id = chunk.document_id
        markdown = markdown_by_document.get(document_id, "")
        if document_id not in facts:
            facts[document_id] = parse_document(markdown)
            cursors[document_id] = 0
        span = locate_span(markdown, chunk.body, cursors[document_id]) if markdown else None
        if span is None and markdown and cursors[document_id] != 0:
            span = locate_span(markdown, chunk.body, 0)
        if span is not None:
            cursors[document_id] = span[0] + 1
        fact = facts[document_id]
        parent_id = chunk.section_id
        records.append(
            ChunkRecord(
                chunk_id=f"{parent_id}:{chunk.index}",
                document_id=document_id,
                filename=f"{document_id}.md",
                title=chunk.document_title,
                page_no=None,
                section_name=chunk.section_title,
                parent_id=parent_id,
                start_span=None if span is None else span[0],
                end_span=None if span is None else span[1],
                chunk_index=chunk.index,
                body=chunk.body,
                embedded_text=chunk.text,
                token_count=chunk.tokens,
                is_table=chunk.table,
                document_date=fact.document_date,
                date_source=fact.date_source,
                version=fact.version,
                superseded=fact.superseded,
                vector=item.vector,
            )
        )
    return records


def replace_index(client: ClientAPI, records: Sequence[ChunkRecord], model_id: str) -> None:
    """Replace the Chroma collection and the SQLite chunk tables with ``records``.

    Every vector must share one dimension. The collection is recreated on each
    replace, so two embedding models never share an index. The SQLite file sits
    next to the Chroma directory.
    """
    if not records:
        _drop_collection(client)
        _replace_sql(_sql_path(client), [], model_id)
        return

    dimension = len(records[0].vector)
    if dimension == 0 or any(len(record.vector) != dimension for record in records):
        raise ValueError("every embedding must share one non-zero dimension")
    _drop_collection(client)

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine", "model_id": model_id, "dimension": dimension},
        embedding_function=None,
    )
    collection.add(
        ids=[record.chunk_id for record in records],
        embeddings=[list(record.vector) for record in records],
        documents=[record.body for record in records],
        metadatas=[_metadata(record) for record in records],
    )
    _replace_sql(_sql_path(client), records, model_id)


def bm25_scores(client: ClientAPI, query: str) -> dict[str, float]:
    """Score ``query`` against chunk bodies with FTS5 BM25.

    The result maps ``chunk_id`` to a higher-is-better score. FTS5's ``bm25()``
    is negative and lower is better, so the value stored here is ``-bm25()``.
    Alphanumeric tokens are stemmed and combined with OR. A query with none
    matches nothing. Chunks that do not match are absent.
    """
    match = _fts_match(query)
    path = _sql_path(client)
    if match is None or not path.exists():
        return {}
    connection = sqlite3.connect(path)
    try:
        created = _ensure_fts(connection, rebuild=False)
        if created:
            connection.commit()
        rows = connection.execute(
            """
            SELECT c.chunk_id, -bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks AS c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            """,
            (match,),
        ).fetchall()
    finally:
        connection.close()
    return {str(chunk_id): float(score) for chunk_id, score in rows}


def get_chunks(client: ClientAPI, chunk_ids: Sequence[str]) -> list[SearchHit]:
    """Return stored chunks for ``chunk_ids``.

    ``distance`` is 0. This read is not a neighbor query, so that field is
    not a rank.
    """
    if not chunk_ids:
        return []
    collection = _open_collection(client)
    if collection is None:
        return []
    found = collection.get(ids=list(chunk_ids), include=["documents", "metadatas"])
    documents = found["documents"] or []
    metadatas = found["metadatas"] or []
    hits: list[SearchHit] = []
    for chunk_id, document, metadata in zip(found["ids"], documents, metadatas, strict=True):
        if document is None or metadata is None:
            continue
        hits.append(_hit(chunk_id, document, metadata, 0.0))
    return hits


def search(client: ClientAPI, vector: Sequence[float], k: int = 5) -> list[SearchHit]:
    """Return the ``k`` nearest chunks by cosine distance."""
    if k < 1:
        raise ValueError("k must be at least 1")
    collection = _open_collection(client)
    if collection is None or collection.count() == 0:
        return []
    stored = None if collection.metadata is None else collection.metadata.get("dimension")
    if stored is None or int(stored) != len(vector):
        found = "missing" if stored is None else stored
        raise ValueError(f"query dimension is {len(vector)} and the index dimension is {found}")
    result = collection.query(
        query_embeddings=[list(vector)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = result["ids"][0]
    documents = result["documents"] or [[]]
    metadatas = result["metadatas"] or [[]]
    distances = result["distances"] or [[]]
    return [
        _hit(chunk_id, document, metadata, distance)
        for chunk_id, document, metadata, distance in zip(
            ids, documents[0], metadatas[0], distances[0], strict=True
        )
        if document is not None and metadata is not None and distance is not None
    ]


def main() -> None:
    """Embed data_md into data/chroma and data/chunks.sqlite."""
    markdown = {
        path.stem: path.read_text(encoding="utf-8") for path in sorted(DATA_MD.glob("*.md"))
    }
    records = build_records(embed_chunks(chunk_directory()), markdown)
    model_id = os.getenv("EMBED_MODEL", _DEFAULT_MODEL_ID)
    client = connect()
    try:
        replace_index(client, records, model_id=model_id)
    finally:
        client.close()
    print(f"stored {len(records)} embeddings in {DEFAULT_PATH} and {SQL_PATH}")


def _iso_date(day: str, month: str, year: str) -> str:
    month_number = _MONTHS.get(month.lower())
    if month_number is None:
        raise ValueError(f"unknown month {month}")
    return f"{int(year):04d}-{month_number:02d}-{int(day):02d}"


def _metadata(record: ChunkRecord) -> dict[str, str | int | float | bool]:
    """Chroma metadata for one chunk. Null fields are left out."""
    metadata: dict[str, str | int | float | bool] = {
        "document_id": record.document_id,
        "filename": record.filename,
        "title": record.title,
        "section_name": record.section_name,
        "parent_id": record.parent_id,
        "chunk_index": record.chunk_index,
        "token_count": record.token_count,
        "is_table": record.is_table,
        "superseded": record.superseded,
        "embedded_text": record.embedded_text,
    }
    optional: dict[str, str | int | None] = {
        "page_no": record.page_no,
        "start_span": record.start_span,
        "end_span": record.end_span,
        "document_date": record.document_date,
        "date_source": record.date_source,
        "version": record.version,
    }
    for key, value in optional.items():
        if value is not None:
            metadata[key] = value
    return metadata


def _hit(
    chunk_id: str,
    document: str,
    metadata: Mapping[str, str | int | float | bool],
    distance: float,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        filename=str(metadata["filename"]),
        title=str(metadata["title"]),
        page_no=_optional_int(metadata, "page_no"),
        section_name=str(metadata["section_name"]),
        parent_id=str(metadata["parent_id"]),
        start_span=_optional_int(metadata, "start_span"),
        end_span=_optional_int(metadata, "end_span"),
        document_date=_optional_str(metadata, "document_date"),
        date_source=_optional_str(metadata, "date_source"),
        version=_optional_str(metadata, "version"),
        superseded=bool(metadata["superseded"]),
        body=document,
        distance=float(distance),
    )


def _optional_int(metadata: Mapping[str, str | int | float | bool], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    return int(value)


def _optional_str(metadata: Mapping[str, str | int | float | bool], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    return str(value)


def _open_collection(client: ClientAPI) -> Collection | None:
    try:
        return client.get_collection(COLLECTION, embedding_function=None)
    except NotFoundError:
        return None


def _drop_collection(client: ClientAPI) -> None:
    try:
        client.delete_collection(COLLECTION)
    except NotFoundError:
        return


def _sql_path(client: ClientAPI) -> Path:
    return Path(client.get_settings().persist_directory).parent / SQL_PATH.name


def _replace_sql(path: Path, records: Sequence[ChunkRecord], model_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        _ensure_sql(connection)
        with connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM index_meta")
            seen: set[str] = set()
            for record in records:
                if record.document_id in seen:
                    continue
                seen.add(record.document_id)
                connection.execute(
                    """
                    INSERT INTO documents (
                        document_id, filename, title, document_date, date_source, version, superseded
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.document_id,
                        record.filename,
                        record.title,
                        record.document_date,
                        record.date_source,
                        record.version,
                        int(record.superseded),
                    ),
                )
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, filename, page_no, section_name, parent_id,
                    start_span, end_span, chunk_index, body, embedded_text, token_count, is_table
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.chunk_id,
                        record.document_id,
                        record.filename,
                        record.page_no,
                        record.section_name,
                        record.parent_id,
                        record.start_span,
                        record.end_span,
                        record.chunk_index,
                        record.body,
                        record.embedded_text,
                        record.token_count,
                        int(record.is_table),
                    )
                    for record in records
                ],
            )
            if records:
                connection.executemany(
                    "INSERT INTO index_meta (key, value) VALUES (?, ?)",
                    (("model_id", model_id), ("dimension", str(len(records[0].vector)))),
                )
            _ensure_fts(connection, rebuild=True)
    finally:
        connection.close()


_FTS_TERM = re.compile(r"[A-Za-z0-9]+")


def _fts_match(query: str) -> str | None:
    """Build an FTS5 OR query from the alphanumeric tokens in ``query``."""
    terms: list[str] = []
    seen: set[str] = set()
    for term in _FTS_TERM.findall(query.lower()):
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


def _ensure_fts(connection: sqlite3.Connection, *, rebuild: bool) -> bool:
    """Create ``chunks_fts`` when missing, and rebuild it when asked.

    Returns True when the index was rebuilt. An old database has the chunk
    rows but no FTS table; the first lexical query builds it from those rows.
    """
    found = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
    ).fetchone()
    if found is None:
        connection.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                body,
                content='chunks',
                content_rowid='rowid',
                tokenize='porter unicode61'
            )
            """
        )
        rebuild = True
    if not rebuild:
        return False
    connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
    return True


def _ensure_sql(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            title TEXT NOT NULL,
            document_date TEXT,
            date_source TEXT CHECK (date_source IN ('publication', 'review') OR date_source IS NULL),
            version TEXT,
            superseded INTEGER NOT NULL CHECK (superseded IN (0, 1))
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(document_id),
            filename TEXT NOT NULL,
            page_no INTEGER CHECK (page_no IS NULL OR page_no >= 1),
            section_name TEXT NOT NULL,
            parent_id TEXT NOT NULL,
            start_span INTEGER,
            end_span INTEGER,
            chunk_index INTEGER NOT NULL,
            body TEXT NOT NULL,
            embedded_text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            is_table INTEGER NOT NULL CHECK (is_table IN (0, 1)),
            CHECK (
                (start_span IS NULL AND end_span IS NULL)
                OR (start_span >= 0 AND end_span >= start_span)
            )
        );

        CREATE TABLE IF NOT EXISTS index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


if __name__ == "__main__":
    main()
