"""Generate text with the Ollama model selected in ``config.MODEL``.

Temperature is fixed at 0. ``top_k`` and ``top_p`` are not sent, so each
model keeps its own defaults. Ollama runs on the host. This container
reaches it at OLLAMA_BASE_URL (http://host.docker.internal:11434).
"""

import os
import sys

import httpx

from src import config

_DEFAULT_BASE_URL = "http://host.docker.internal:11434"
TEMPERATURE = 0


def generate(prompt: str) -> str:
    """Send ``prompt`` to the model named by ``config.MODEL``.

    Posts ``POST {OLLAMA_BASE_URL}/api/generate`` with ``stream`` off and
    ``options.temperature`` set to 0. Returns the ``response`` text.
    """
    try:
        model_id = config.MODELS[config.MODEL]
    except KeyError as exc:
        choices = ", ".join(config.MODELS)
        raise ValueError(f"MODEL must be one of: {choices}") from exc

    base_url = os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    response = httpx.post(
        f"{base_url}/api/generate",
        json={
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": TEMPERATURE},
        },
        timeout=180.0,
    )
    response.raise_for_status()
    text = response.json().get("response")
    if not isinstance(text, str):
        raise ValueError("Ollama returned no response text")
    return text


def main() -> None:
    """Generate from the command-line prompt and print the text."""
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        raise SystemExit("usage: python -m src.generate PROMPT")
    print(generate(prompt))


if __name__ == "__main__":
    main()
