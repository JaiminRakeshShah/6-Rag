import pytest

from src.chunking import chunk_markdown
from src.embed import embed_chunks, embed_texts


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_embed_texts_posts_to_the_host_ollama_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    calls: list[tuple[str, dict]] = []

    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        calls.append((url, json))
        return _Response({"embeddings": [[0.25, 0.5], [0.75, 1.0]]})

    monkeypatch.setattr("src.embed.httpx.post", fake_post)

    assert embed_texts(["alpha", "beta"]) == [(0.25, 0.5), (0.75, 1.0)]
    assert calls == [
        (
            "http://host.docker.internal:11434/api/embed",
            {"model": "embeddinggemma", "input": ["alpha", "beta"]},
        )
    ]


def test_embed_chunks_sends_prefixed_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    seen: dict[str, list[str]] = {}

    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        seen["input"] = json["input"]
        return _Response({"embeddings": [[1.0], [2.0]]})

    monkeypatch.setattr("src.embed.httpx.post", fake_post)
    markdown = "# **Q3 Report**\n\n## **Revenue**\n\nDown 10 percent.\n\n## **Outlook**\n\nSteady.\n"
    chunks = chunk_markdown(markdown, document_id="Q3_Report")
    embedded = embed_chunks(chunks)

    assert seen["input"] == [chunk.text for chunk in chunks]
    assert all(item.startswith("[Doc: Q3 Report | Section: ") for item in seen["input"])
    assert [item.chunk.section_id for item in embedded] == [chunk.section_id for chunk in chunks]
    assert embedded[0].vector == (1.0,)


def test_empty_input_does_not_call_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        raise AssertionError("server should not be called")

    monkeypatch.setattr("src.embed.httpx.post", fake_post)
    assert embed_texts([]) == []


def test_mismatched_embedding_count_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        return _Response({"embeddings": [[1.0]]})

    monkeypatch.setattr("src.embed.httpx.post", fake_post)
    with pytest.raises(ValueError):
        embed_texts(["alpha", "beta"])
