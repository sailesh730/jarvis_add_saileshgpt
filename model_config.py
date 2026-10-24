"""Model choices shared by the chat API and LangGraph agents."""

DEFAULT_MODEL = "gemini-3.6-flash"

MODEL_OPTIONS = [
    ("gemini-3.8-flash", "Gemini 3.8 Flash"),
    (DEFAULT_MODEL, "Gemini 3.6 Flash"),
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"),
    ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite"),
    ("gemini-3.1-pro-preview", "Gemini 3.1 Pro (Preview)"),
    ("gemini-3-flash-preview", "Gemini 3 Flash (Preview)"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("qwen2.5:7b", "Qwen 2.5 7B (Ollama)"),
    ("llama3.1:8b", "Llama 3.1 8B (Ollama)"),
]

MODEL_IDS = {model_id for model_id, _ in MODEL_OPTIONS}

LEGACY_MODEL_ALIASES = {
    "gemini-1.5-flash": DEFAULT_MODEL,
    "gemini-1.5-pro": "gemini-2.5-pro",
    "qwen2.5": "qwen2.5:7b",
    "llama3.1": "llama3.1:8b",
}
