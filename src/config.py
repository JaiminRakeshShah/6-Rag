"""Gate for the generator.

Set ``MODEL`` to ``gemma``, ``qwen``, or ``gptoss``. The string on the
right is the Ollama tag ``generate`` sends.
"""

MODEL = "gemma"

MODELS = {
    "gemma": "gemma3:12b",
    "qwen": "qwen3:8b",
    "gptoss": "gpt-oss:20b",
}
