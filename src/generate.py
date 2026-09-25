"""Generate text with the Ollama model selected in ``config.MODEL``.

Temperature is fixed at 0. ``top_k`` and ``top_p`` are not sent, so each
model keeps its own defaults. Ollama runs on the host. This container
reaches it at OLLAMA_BASE_URL (http://host.docker.internal:11434).
"""

import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from src import config
from src.retrieve import HybridHit

_DEFAULT_BASE_URL = "http://host.docker.internal:11434"
TEMPERATURE = 0
_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class MissingPromptVariableError(ValueError):
    """A required prompt variable was not supplied."""


def load_prompt(prompt_id: str | None = None, version: str | None = None) -> str:
    """Read ``src/prompts/{prompt_id}.{version}.md``. Defaults come from config."""
    prompt_id = config.PROMPT_ID if prompt_id is None else prompt_id
    version = config.PROMPT_VERSION if version is None else version
    path = _PROMPT_DIR / f"{prompt_id}.{version}.md"
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, variables: dict[str, str]) -> str:
    """Fill ``{name}`` placeholders. Literal JSON braces stay as written."""
    required = set(_PLACEHOLDER.findall(template))
    missing = sorted(name for name in required if name not in variables)
    if missing:
        raise MissingPromptVariableError(f"missing prompt variables: {', '.join(missing)}")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            return match.group(0)
        return variables[name]

    return _PLACEHOLDER.sub(replace, template)


def format_chunks(hits: Sequence[HybridHit]) -> str:
    """Chunk block the generation prompt inserts at ``{chunks}``."""
    blocks: list[str] = []
    for hit in hits:
        status = "superseded" if hit.superseded else "current"
        blocks.append(
            "\n".join(
                (
                    f"chunk_id: {hit.chunk_id}",
                    f"section_id: {hit.parent_id}",
                    f"section_name: {hit.section_name}",
                    f"filename: {hit.filename}",
                    f"status: {status}",
                    "body:",
                    hit.body,
                )
            )
        )
    return "\n\n".join(blocks) if blocks else "(no chunks retrieved)"


def generation_prompt(question: str, hits: Sequence[HybridHit]) -> str:
    """Render the configured generation prompt for one question."""
    return render_prompt(
        load_prompt(),
        {"question": question, "chunks": format_chunks(hits)},
    )


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
