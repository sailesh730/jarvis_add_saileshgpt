"""
agent.py - builds the LangGraph agents (general + travel) and streams their replies.

Main entry points used by the web server:
    get_reply_events(message, thread_id, model_id, agent_type)
        yields dicts: {"type": "delta", "text"} | {"type": "tool_start", "tool", "tool_call_id"}
                      | {"type": "tool_complete", "tool", "tool_call_id"}
    get_reply_chunks(...)  same stream but plain text only (old interface)
    get_agent(model_name, agent_type)
    delete_thread_state(thread_id)

Agent types: "general", "travel", and "jarvis" (controls this computer, see system_tools.py).
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402

from model_config import DEFAULT_MODEL, LEGACY_MODEL_ALIASES, MODEL_IDS, MODEL_OPTIONS  # noqa: E402
from tools import GENERAL_TOOLS, TRAVEL_TOOLS, web_search  # noqa: E402
from tools import SYSTEM_PROMPT as TRAVEL_SYSTEM_PROMPT  # noqa: E402
from system_tools import JARVIS_SYSTEM_PROMPT, SYSTEM_TOOLS  # noqa: E402

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ALLOWED_MODELS = MODEL_IDS | LEGACY_MODEL_ALIASES.keys()
AGENT_TYPES = {"general", "travel", "jarvis"}

# Tools per agent. web_search is None when TAVILY_API_KEY is missing, so it is filtered out.
# Jarvis gets the PC-control tools plus a few light general tools. Small local models struggle
# with 30+ tool schemas, so only these general tools are carried over (matched by tool name).
_JARVIS_EXTRA_TOOL_NAMES = {
    "web_search", "tavily_search", "wed_search", "calculator", "get_weather", "get_today",
    "remember_this", "rember_this", "recall_memory",
}
_jarvis_extras = [t for t in GENERAL_TOOLS if getattr(t, "name", "") in _JARVIS_EXTRA_TOOL_NAMES]
if web_search is not None and all(getattr(t, "name", "") != getattr(web_search, "name", "") for t in _jarvis_extras):
    _jarvis_extras.append(web_search)

AGENT_TOOLS = {
    "general": list(GENERAL_TOOLS),
    "travel": list(TRAVEL_TOOLS) + ([web_search] if web_search is not None else []),
    "jarvis": list(SYSTEM_TOOLS) + _jarvis_extras,
}

GENERAL_SYSTEM_PROMPT = """You are SaileshGPT, a helpful, accurate assistant.

- Answer simple questions directly. Never use a tool just because it exists.
- Use web search for current or online information (latest, today, news, prices, versions).
- Use the calculator for arithmetic that needs accuracy.
- Use get_weather for current weather conditions.
- Use memory tools only when the user asks you to remember or recall something.
- Use search_uploaded_document for questions about files the user uploaded. If the answer is
  not in the document, say so instead of guessing.
- Use browser_read when the user asks you to open or read a specific public URL.
- When you used web search or a document, say so, and mention sources when available.
- Never claim a tool was used unless you called it, and never invent tool results, sources,
  links or memories. If a tool fails, say so briefly and continue with what you know.
- Keep answers clear, well structured and no longer than needed."""

SYSTEM_PROMPTS = {
    "general": GENERAL_SYSTEM_PROMPT,
    "travel": TRAVEL_SYSTEM_PROMPT,
    "jarvis": JARVIS_SYSTEM_PROMPT,
}


# ==========================================================================
# Models
# ==========================================================================
def normalize_model(model_name: Optional[str]) -> str:
    """Validate the selected model and fall back to the default."""
    if not model_name:
        return DEFAULT_MODEL
    model_name = model_name.strip()
    if model_name in LEGACY_MODEL_ALIASES:
        return LEGACY_MODEL_ALIASES[model_name]
    if model_name in ALLOWED_MODELS:
        return model_name
    return DEFAULT_MODEL


def _is_gemini(model_name: str) -> bool:
    return model_name.startswith("gemini-")


def _build_llm(selected_model: str, num_ctx: int = 4096):
    if _is_gemini(selected_model):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY. Add it to the project .env file to use Gemini.")
        return ChatGoogleGenerativeAI(
            model=selected_model,
            api_key=google_api_key,
            temperature=0.4,
            streaming=True,
        )
    return ChatOllama(
        model=selected_model,
        temperature=0.4,
        num_ctx=num_ctx,
        num_predict=1024,  # was 128, which cut every answer off after a few sentences
    )


# ==========================================================================
# Graph
# ==========================================================================
_CHECKPOINTERS: dict = {}


def _get_checkpointer(agent_type: str) -> SqliteSaver:
    """One history database per agent type, shared by all models, so switching models
    in the middle of a chat keeps the conversation context."""
    if agent_type not in _CHECKPOINTERS:
        conn = sqlite3.connect(
            DATA_DIR / f"langgraph_checkpoints_{agent_type}.sqlite",
            check_same_thread=False,
        )
        _CHECKPOINTERS[agent_type] = SqliteSaver(conn)
    return _CHECKPOINTERS[agent_type]


def build_graph(llm, agent_type: str):
    tools = AGENT_TOOLS[agent_type]
    system_prompt = SYSTEM_PROMPTS[agent_type]
    llm_with_tools = llm.bind_tools(tools)

    def chat_node(state: MessagesState):
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    return graph.compile(checkpointer=_get_checkpointer(agent_type))


def build_agent(model_name: str):
    selected = normalize_model(model_name)
    return build_graph(_build_llm(selected), "general")


def build_travel_agent(model_name: str):
    selected = normalize_model(model_name)
    return build_graph(_build_llm(selected), "travel")


def build_jarvis_agent(model_name: str):
    selected = normalize_model(model_name)
    # ~25 tool schemas need more room than the default 4096-token window on local Ollama models.
    return build_graph(_build_llm(selected, num_ctx=8192), "jarvis")


_AGENT_CACHE: dict = {}


def get_agent(model_name: Optional[str] = None, agent_type: str = "general"):
    """Return a cached agent for (model, agent type), building it on first use."""
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    selected = normalize_model(model_name)
    key = (selected, agent_type)
    if key not in _AGENT_CACHE:
        builder = {
            "general": build_agent,
            "travel": build_travel_agent,
            "jarvis": build_jarvis_agent,
        }[agent_type]
        _AGENT_CACHE[key] = builder(selected)
    return _AGENT_CACHE[key]


def delete_thread_state(thread_id: str) -> None:
    """Remove a chat's saved agent context from every agent's checkpoint database."""
    for agent_type in AGENT_TYPES:
        saver = _get_checkpointer(agent_type)
        deleter = getattr(saver, "delete_thread", None)
        if deleter is not None:
            deleter(thread_id)


# ==========================================================================
# Errors and retry
# ==========================================================================
def get_retry_model_candidates(model_name: Optional[str]) -> list:
    """The selected model first, then other models from the same provider as fallbacks."""
    selected = normalize_model(model_name)
    candidates = [selected]
    for model_id, _ in MODEL_OPTIONS:
        if model_id not in candidates and _is_gemini(model_id) == _is_gemini(selected):
            candidates.append(model_id)
    return candidates


def is_retryable_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "503", "429", "rate limit", "too many requests", "unavailable",
            "high demand", "overloaded", "resource_exhausted",
        )
    )


def explain_model_error(exc: Exception, model: str) -> str:
    """Turn raw provider errors into a message the user can act on."""
    raw = str(exc)
    low = raw.lower()
    if not _is_gemini(model) and ("refused" in low or "connect" in low):
        return f"Could not reach Ollama. Run 'ollama serve' and 'ollama pull {model}', then try again."
    if "does not support tools" in low:
        return f"The model '{model}' does not support tool calling. Pick another model."
    if "not found" in low or "404" in low:
        return f"The model '{model}' was not found. Choose a different model or check its name."
    if "api key" in low or "api_key" in low or "permission" in low or "401" in low or "403" in low:
        return "The API key was rejected. Check GOOGLE_API_KEY in your .env file."
    if "429" in low or "quota" in low or "resource_exhausted" in low:
        return f"Rate limit or quota reached for '{model}'. Wait a bit or pick another model."
    return f"{type(exc).__name__}: {raw[:300]}"


# ==========================================================================
# Streaming
# ==========================================================================
def _chunk_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type", "text") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _tool_events(update):
    """Turn a LangGraph 'updates' payload into tool_start / tool_complete events."""
    if not isinstance(update, dict):
        return
    for payload in update.values():
        if not isinstance(payload, dict):
            continue
        for msg in payload.get("messages") or []:
            if isinstance(msg, AIMessage):
                for call in msg.tool_calls or []:
                    yield {"type": "tool_start", "tool": call["name"], "tool_call_id": call.get("id") or ""}
            elif isinstance(msg, ToolMessage):
                yield {"type": "tool_complete", "tool": msg.name or "", "tool_call_id": msg.tool_call_id or ""}


def _stream_input(agent, config: dict, message: str, resume: bool):
    """On a retry, resume the interrupted run instead of adding the user message twice."""
    if resume:
        try:
            if agent.get_state(config).next:
                return None
        except Exception:
            pass
    return {"messages": [HumanMessage(content=message)]}


def get_reply_events(message: str, thread_id: str, model_id: str, agent_type: str = "general"):
    candidates = get_retry_model_candidates(model_id)
    config = {"configurable": {"thread_id": thread_id}}
    emitted = False
    last_error = None

    for attempt, candidate in enumerate(candidates):
        try:
            agent = get_agent(candidate, agent_type)
            payload = _stream_input(agent, config, message, resume=attempt > 0)
            for mode, data in agent.stream(payload, config=config, stream_mode=["messages", "updates"]):
                if mode == "messages":
                    chunk, _meta = data
                    if isinstance(chunk, AIMessageChunk):
                        text = _chunk_text(chunk.content)
                        if text:
                            emitted = True
                            yield {"type": "delta", "text": text}
                elif mode == "updates":
                    for event in _tool_events(data):
                        emitted = True
                        yield event
            return
        except Exception as exc:
            last_error = exc
            # Never retry after output has started: the user would see duplicated text.
            if emitted or not is_retryable_model_error(exc):
                raise RuntimeError(explain_model_error(exc, candidate)) from exc

    raise RuntimeError(
        f"All available models failed. Last error: {explain_model_error(last_error, candidates[-1])}"
    ) from last_error


def get_reply_chunks(message: str, thread_id: str, model_id: str, agent_type: str = "general"):
    """Text-only version of get_reply_events (kept for the old interface)."""
    for event in get_reply_events(message, thread_id, model_id, agent_type):
        if event["type"] == "delta":
            yield event["text"]


# --------------------------------------------------------------------------
# Backward-compatible names
# --------------------------------------------------------------------------
normilize_model = normalize_model
DEFFAULT_MODEL = DEFAULT_MODEL
DEFFULT_MODEL = DEFAULT_MODEL
ALLOWED_MODEL = ALLOWED_MODELS