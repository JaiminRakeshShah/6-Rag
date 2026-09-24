"""Embed chunk text by calling the Ollama server over HTTP.

Ollama runs on the host. This container reaches it at OLLAMA_BASE_URL
(http://host.docker.internal:11434), the same way a client calls a remote server.
"""

import os
from dataclasses import dataclass

import httpx

from src.chunking import Chunk, chunk_directory

_DEFAULT_BASE_URL = "http://host.docker.internal:11434"
_DEFAULT_MODEL_ID = "embeddinggemma"


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk and the embedding of its prefixed ``text``."""

    chunk: Chunk
    vector: tuple[float, ...]


def embed_texts(texts: list[str]) -> list[tuple[float, ...]]:
    """Embed each string by posting it to the host Ollama server.

    Sends ``POST {OLLAMA_BASE_URL}/api/embed`` with model ``EMBED_MODEL``
    (``embeddinggemma`` unless overridden). Returns one vector per input,
    in the same order. An empty list does not call the server.
    """
    if not texts:
        return []
    base_url = os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    model_id = os.getenv("EMBED_MODEL", _DEFAULT_MODEL_ID)
    response = httpx.post(
        f"{base_url}/api/embed",
        json={"model": model_id, "input": texts},
        timeout=180.0,
    )
    response.raise_for_status()
    embeddings = response.json().get("embeddings")
    received = len(embeddings) if isinstance(embeddings, list) else 0
    if not isinstance(embeddings, list) or received != len(texts):
        raise ValueError(f"Ollama returned {received} embeddings for {len(texts)} inputs")
    return [tuple(float(value) for value in vector) for vector in embeddings]


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Embed ``chunk.text``, which already includes the document and section prefix."""
    vectors = embed_texts([chunk.text for chunk in chunks])
    return [EmbeddedChunk(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)]


def main() -> None:
    """Embed every Markdown file in data_md and print the count and vector length."""
    embedded = embed_chunks(chunk_directory())
    dimension = len(embedded[0].vector) if embedded else 0
    print(f"{len(embedded)} embeddings, dimension {dimension}")


if __name__ == "__main__":
    main()
