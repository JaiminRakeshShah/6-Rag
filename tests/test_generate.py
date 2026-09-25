import pytest

from src import config
from src.generate import generate


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_generate_uses_the_config_model_and_temperature_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr(config, "MODEL", "gptoss")
    calls: list[tuple[str, dict]] = []

    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        calls.append((url, json))
        return _Response({"response": "2040"})

    monkeypatch.setattr("src.generate.httpx.post", fake_post)

    assert generate("By which year?") == "2040"
    assert calls == [
        (
            "http://host.docker.internal:11434/api/generate",
            {
                "model": "gpt-oss:20b",
                "prompt": "By which year?",
                "stream": False,
                "options": {"temperature": 0},
            },
        )
    ]


def test_generate_defaults_to_gemma(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        seen["model"] = json["model"]
        return _Response({"response": "ok"})

    monkeypatch.setattr("src.generate.httpx.post", fake_post)

    assert config.MODEL == "gemma"
    assert generate("hello") == "ok"
    assert seen["model"] == "gemma3:12b"


def test_unknown_model_does_not_call_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MODEL", "mistral")

    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        raise AssertionError("server should not be called")

    monkeypatch.setattr("src.generate.httpx.post", fake_post)

    with pytest.raises(ValueError, match="gemma"):
        generate("hello")


def test_missing_response_text_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> _Response:
        return _Response({"done": True})

    monkeypatch.setattr("src.generate.httpx.post", fake_post)

    with pytest.raises(ValueError):
        generate("hello")
