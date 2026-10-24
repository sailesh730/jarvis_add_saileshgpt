"""
app.py - FastAPI server for SaileshGPT.

Run:  python app.py      (then open http://127.0.0.1:8080)
Needs: templates/index.html (the front-end page), agent.py, rag.py, tools.py,
       database.py, model_config.py, system_tools.py, jarvis_routes.py in the same folder.
"""

import json
import logging
import os
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402

import database  # noqa: E402
from database import (  # noqa: E402
    SessionLocal,
    create_or_update_conversation,
    get_chat_history,
    get_user_conversation,
    get_user_conversations,
    init_db,
    save_chat_message,
)
from jarvis_routes import build_router, is_local_request  # noqa: E402
  # noqa: E402
from model_config import DEFAULT_MODEL,LEGACY_MODEL_ALIASES,MODEL_IDS,MODEL_OPTIONS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("saileshgpt")

SESSION_COOKIE = "sailesh_session"
SESSION_PATTERN = re.compile(r"[A-Za-z0-9_-]{40,50}")
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".py", ".csv"}
AGENT_TYPES = {"general", "travel", "jarvis"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_MESSAGE_CHARS = 10_000

Path("upload").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="SaileshGPT", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# The front-end reads data.error, so return every HTTP error in that shape.
@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def get_browser_session_id(request: Request) -> str:
    session_id = request.cookies.get(SESSION_COOKIE, "")
    if not SESSION_PATTERN.fullmatch(session_id):
        raise HTTPException(status_code=401, detail="Refresh the page to start a session.")
    return session_id


def error_text(exc: Exception) -> str:
    return str(exc) or "Chat request failed. Check the server logs for details."


# Voice (ElevenLabs) and PC-control approval endpoints live in jarvis_routes.py.
app.include_router(build_router(get_browser_session_id))


# ==========================================================================
# Pages and models
# ==========================================================================
@app.get("/")
async def home(request: Request):
    response = templates.TemplateResponse(request=request, name="index.html", context={})
    if not SESSION_PATTERN.fullmatch(request.cookies.get(SESSION_COOKIE, "")):
        response.set_cookie(
            key=SESSION_COOKIE,
            value=secrets.token_urlsafe(32),
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
        )
    return response


@app.get("/models")
async def list_models():
    return JSONResponse(
        {
            "default_model": DEFAULT_MODEL,
            "models": [{"id": model_id, "label": label} for model_id, label in MODEL_OPTIONS],
        }
    )


# ==========================================================================
# Conversations (plain `def` endpoints run in the threadpool, so DB calls don't block)
# ==========================================================================
@app.get("/conversations")
def list_conversations(request: Request):
    session_id = get_browser_session_id(request)
    db = SessionLocal()
    try:
        return JSONResponse(
            [
                {
                    "thread_id": c.thread_id,
                    "title": c.title or "New conversation",
                    "updated_at": c.updated_at.isoformat(),
                }
                for c in get_user_conversations(db, session_id)
            ]
        )
    finally:
        db.close()


@app.get("/conversations/{thread_id}")
def load_conversation(thread_id: str, request: Request):
    session_id = get_browser_session_id(request)
    db = SessionLocal()
    try:
        conversation = get_user_conversation(db, thread_id, session_id)
        if conversation is None:
            return JSONResponse({"error": "Conversation not found."}, status_code=404)
        return JSONResponse(
            {
                "thread_id": conversation.thread_id,
                "title": conversation.title or "New conversation",
                "messages": [
                    {"role": m.role, "content": m.content} for m in get_chat_history(db, thread_id)
                ],
            }
        )
    finally:
        db.close()


def _cleanup_thread_data(thread_id: str) -> None:
    """Best-effort removal of a chat's uploaded-document chunks and agent context."""
    try:
        from rag import delete_thread_documents

        delete_thread_documents(thread_id)
    except Exception:
        logger.warning("Could not delete documents for %s", thread_id, exc_info=True)
    try:
        from agent import delete_thread_state

        delete_thread_state(thread_id)
    except Exception:
        logger.warning("Could not delete agent state for %s", thread_id, exc_info=True)


def _require_db_function(name: str):
    func = getattr(database, name, None)
    if func is None:
        raise HTTPException(
            status_code=501,
            detail=f"Deleting is not set up yet: add '{name}' to database.py.",
        )
    return func


@app.delete("/conversations/{thread_id}")
def delete_conversation(thread_id: str, request: Request):
    session_id = get_browser_session_id(request)
    delete_one = _require_db_function("delete_user_conversation")
    db = SessionLocal()
    try:
        if get_user_conversation(db, thread_id, session_id) is None:
            return JSONResponse({"error": "Conversation not found."}, status_code=404)
        delete_one(db, thread_id, session_id)
    finally:
        db.close()
    _cleanup_thread_data(thread_id)
    return JSONResponse({"ok": True})


@app.delete("/conversations")
def delete_all_conversations(request: Request):
    session_id = get_browser_session_id(request)
    delete_all = _require_db_function("delete_user_conversations")
    db = SessionLocal()
    try:
        thread_ids = [c.thread_id for c in get_user_conversations(db, session_id)]
        delete_all(db, session_id)
    finally:
        db.close()
    for thread_id in thread_ids:
        _cleanup_thread_data(thread_id)
    return JSONResponse({"ok": True, "deleted": len(thread_ids)})


# ==========================================================================
# Chat
# ==========================================================================
def thread_belongs_to_session(thread_id: str, session_id: str) -> bool:
    db = SessionLocal()
    try:
        return get_user_conversation(db, thread_id, session_id) is not None
    finally:
        db.close()


async def prepare_chat_request(request: Request):
    """Validate the request. Returns a tuple on success or a JSONResponse on error."""
    session_id = get_browser_session_id(request)
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    message = payload.get("message")
    message = message.strip() if isinstance(message, str) else ""
    if not message:
        return JSONResponse({"error": "Message is required."}, status_code=400)
    if len(message) > MAX_MESSAGE_CHARS:
        return JSONResponse(
            {"error": f"Message is too long (max {MAX_MESSAGE_CHARS} characters)."},
            status_code=413,
        )

    thread_id = payload.get("thread_id")
    if thread_id is not None and not isinstance(thread_id, str):
        return JSONResponse({"error": "Invalid conversation ID."}, status_code=400)

    is_new = not thread_id
    if thread_id:
        if not await run_in_threadpool(thread_belongs_to_session, thread_id, session_id):
            return JSONResponse({"error": "Conversation not found."}, status_code=404)
    else:
        thread_id = f"chat_{uuid.uuid4().hex}"

    model_id = payload.get("model_id") or DEFAULT_MODEL
    if not isinstance(model_id, str) or model_id not in MODEL_IDS:
        return JSONResponse({"error": "Select a supported model."}, status_code=400)

    agent_type = payload.get("agent") or "general"
    if not isinstance(agent_type, str) or agent_type not in AGENT_TYPES:
        return JSONResponse({"error": "Select a supported assistant."}, status_code=400)
    if agent_type == "jarvis" and not is_local_request(request):
        return JSONResponse({"error": "Jarvis can only be used from this computer."}, status_code=403)

    return session_id, message, thread_id, is_new, model_id, agent_type


def save_chat_turn(session_id: str, thread_id: str, message: str, reply: str, is_new: bool) -> str:
    db = SessionLocal()
    try:
        conversation = create_or_update_conversation(
            db,
            thread_id=thread_id,
            user_id=session_id,
            title=message[:80] if is_new else None,
        )
        save_chat_message(db, thread_id, "user", message)
        save_chat_message(db, thread_id, "assistant", reply)
        return conversation.title or "New conversation"
    finally:
        db.close()


def run_chat(session_id, message, thread_id, is_new, model_id, agent_type="general"):
    from agent import get_reply_chunks  # imported here so a setup problem shows up as a chat error

    reply = "".join(get_reply_chunks(message, thread_id, model_id, agent_type)).strip()
    reply = reply or "I did not receive a response."
    return reply, save_chat_turn(session_id, thread_id, message, reply, is_new)


@app.post("/chat")
async def chat(request: Request):
    prepared = await prepare_chat_request(request)
    if isinstance(prepared, JSONResponse):
        return prepared
    session_id, message, thread_id, is_new, model_id, agent_type = prepared

    try:
        reply, title = await run_in_threadpool(
            run_chat, session_id, message, thread_id, is_new, model_id, agent_type
        )
        return JSONResponse({"reply": reply, "thread_id": thread_id, "title": title})
    except Exception as exc:
        logger.exception("Chat request failed")
        return JSONResponse({"error": error_text(exc)}, status_code=500)


@app.post("/chat/stream")
async def chat_stream(request: Request):
    prepared = await prepare_chat_request(request)
    if isinstance(prepared, JSONResponse):
        return prepared
    session_id, message, thread_id, is_new, model_id, agent_type = prepared

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # A sync generator is iterated in a worker thread, so one streaming reply
    # doesn't block other requests.
    def event_stream():
        reply_parts = []
        try:
            from agent import get_reply_events

            for event in get_reply_events(message, thread_id, model_id, agent_type):
                if event["type"] == "delta":
                    reply_parts.append(event["text"])
                yield sse(event)

            reply = "".join(reply_parts).strip() or "I did not receive a response."
            title = save_chat_turn(session_id, thread_id, message, reply, is_new)
            yield sse(
                {
                    "type": "complete",
                    "thread_id": thread_id,
                    "title": title,
                    "reply": reply,
                    "model_id": model_id,
                }
            )
        except Exception as exc:
            logger.exception("Streaming chat request failed")
            yield sse({"type": "error", "message": error_text(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ==========================================================================
# Upload
# ==========================================================================
def get_or_create_upload_thread(thread_id, session_id, original_name):
    """Returns (thread_id, title), or None if the thread isn't this session's."""
    if not thread_id:
        return f"chat_{uuid.uuid4().hex}", original_name[:80]
    db = SessionLocal()
    try:
        conversation = get_user_conversation(db, thread_id, session_id)
        if conversation is None:
            return None
        return thread_id, conversation.title or original_name[:80]
    finally:
        db.close()


def save_upload_messages(session_id, thread_id, title, original_name, confirmation):
    db = SessionLocal()
    try:
        create_or_update_conversation(db, thread_id=thread_id, user_id=session_id, title=title)
        save_chat_message(db, thread_id, "user", f"Uploaded file: {original_name}")
        save_chat_message(db, thread_id, "assistant", confirmation)
    finally:
        db.close()


@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(default=None),
):
    session_id = get_browser_session_id(request)
    original_name = Path((file.filename or "").replace("\\", "/")).name
    extension = Path(original_name).suffix.lower()
    if not original_name or extension not in ALLOWED_UPLOAD_EXTENSIONS:
        await file.close()
        return JSONResponse(
            {"error": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV files."},
            status_code=400,
        )

    thread = await run_in_threadpool(get_or_create_upload_thread, thread_id, session_id, original_name)
    if thread is None:
        await file.close()
        return JSONResponse({"error": "Conversation not found."}, status_code=404)
    thread_id, title = thread

    stored_path = Path("upload") / f"{uuid.uuid4().hex}{extension}"
    total_bytes = 0
    try:
        with stored_path.open("xb") as destination:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    return JSONResponse(
                        {"error": "File is too large. Maximum upload size is 20 MB."},
                        status_code=413,
                    )
                destination.write(chunk)

        from rag import add_document_to_rag  # imported here so a setup problem shows up as an upload error

        result = await run_in_threadpool(
            add_document_to_rag, str(stored_path), thread_id, source_name=original_name
        )
        confirmation = f"Uploaded and indexed {original_name} ({result['chunks']} text chunks)."
        await run_in_threadpool(
            save_upload_messages, session_id, thread_id, title, original_name, confirmation
        )
        return JSONResponse(
            {
                "thread_id": thread_id,
                "title": title,
                "filename": original_name,
                "chunks": result["chunks"],
                "message": confirmation,
            }
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        logger.exception("Upload failed")
        return JSONResponse({"error": f"Upload failed: {str(exc)[:200]}"}, status_code=500)
    finally:
        stored_path.unlink(missing_ok=True)  # the text is indexed; the raw file isn't needed
        await file.close()


if __name__ == "__main__":
    # 127.0.0.1 keeps the dev server private; use 0.0.0.0 only to share it on your network.
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)