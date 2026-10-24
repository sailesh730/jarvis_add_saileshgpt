"""
jarvis_routes.py - HTTP endpoints for the voice (ElevenLabs) and PC-control features.

    GET  /voice/status       which voice features are configured
    POST /voice/speak        {text}  -> audio/mpeg   (ElevenLabs text-to-speech)
    POST /voice/transcribe   audio file -> {text}    (ElevenLabs Scribe speech-to-text)
    GET  /system/pending     actions waiting for your Approve / Deny click
    POST /system/approve     {id, approved}

All of these require your browser session cookie, and the PC-control ones only accept
requests that come from this same computer (127.0.0.1).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable

import requests
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from system_tools import approvals

logger = logging.getLogger("saileshgpt.jarvis")

ELEVEN_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George"; change with ELEVENLABS_VOICE_ID in .env
MAX_TTS_CHARS = int(os.getenv("ELEVENLABS_MAX_CHARS", "1200"))  # keeps your character quota safe
MAX_AUDIO_BYTES = 15 * 1024 * 1024
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def is_local_request(request: Request) -> bool:
    """True only when the request comes from this same machine."""
    client = request.client
    return bool(client and client.host in LOCAL_HOSTS)


def require_local(request: Request) -> None:
    if not is_local_request(request):
        raise HTTPException(status_code=403, detail="PC control is only available from this computer.")


def _api_key() -> str:
    return os.getenv("ELEVENLABS_API_KEY", "").strip()


def _voice_id() -> str:
    return os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID).strip() or DEFAULT_VOICE_ID


def _clean_for_speech(text: str) -> str:
    """Remove markdown/URLs so the voice doesn't read symbols, then cap the length."""
    text = re.sub(r"```.*?```", " (code omitted) ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[#*_>~|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_TTS_CHARS:
        cut = text[:MAX_TTS_CHARS]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[: end + 1] if end > MAX_TTS_CHARS * 0.5 else cut
    return text


def _eleven_error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("status")
        message = str(detail) if detail else response.text[:200]
    except Exception:
        message = response.text[:200]
    if response.status_code in (401, 403):
        return "ElevenLabs rejected the API key. Check ELEVENLABS_API_KEY in .env."
    if response.status_code == 429:
        return "ElevenLabs rate limit or quota reached."
    return f"ElevenLabs error {response.status_code}: {message}"


def build_router(session_dependency: Callable) -> APIRouter:
    router = APIRouter()
    session = [Depends(session_dependency)]

    # ---------------------------------------------------------------- voice
    @router.get("/voice/status", dependencies=session)
    def voice_status():
        configured = bool(_api_key())
        return JSONResponse(
            {
                "elevenlabs": configured,
                "tts": configured,
                "stt": configured,
                "voice_id": _voice_id() if configured else None,
            }
        )

    @router.post("/voice/speak", dependencies=session)
    async def voice_speak(request: Request):
        if not _api_key():
            return JSONResponse({"error": "ELEVENLABS_API_KEY is not set in .env."}, status_code=503)
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body."}, status_code=400)
        text = payload.get("text") if isinstance(payload, dict) else None
        text = _clean_for_speech(text) if isinstance(text, str) else ""
        if not text:
            return JSONResponse({"error": "Nothing to speak."}, status_code=400)

        def call():
            return requests.post(
                f"{ELEVEN_BASE}/text-to-speech/{_voice_id()}",
                params={"output_format": "mp3_44100_64"},
                headers={"xi-api-key": _api_key(), "Content-Type": "application/json"},
                json={
                    "text": text,
                    "model_id": os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"),
                },
                timeout=45,
            )

        from fastapi.concurrency import run_in_threadpool

        try:
            response = await run_in_threadpool(call)
        except requests.RequestException as exc:
            logger.warning("ElevenLabs TTS request failed: %s", exc)
            return JSONResponse({"error": "Could not reach ElevenLabs."}, status_code=502)
        if response.status_code != 200:
            return JSONResponse({"error": _eleven_error(response)}, status_code=502)
        return Response(content=response.content, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})

    @router.post("/voice/transcribe", dependencies=session)
    async def voice_transcribe(audio: UploadFile = File(...)):
        if not _api_key():
            return JSONResponse({"error": "ELEVENLABS_API_KEY is not set in .env."}, status_code=503)
        data = await audio.read(MAX_AUDIO_BYTES + 1)
        await audio.close()
        if not data:
            return JSONResponse({"error": "No audio received."}, status_code=400)
        if len(data) > MAX_AUDIO_BYTES:
            return JSONResponse({"error": "Recording is too long."}, status_code=413)

        def call():
            form = {"model_id": os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")}
            language = os.getenv("ELEVENLABS_STT_LANGUAGE", "").strip()
            if language:
                form["language_code"] = language
            return requests.post(
                f"{ELEVEN_BASE}/speech-to-text",
                headers={"xi-api-key": _api_key()},
                data=form,
                files={"file": (audio.filename or "speech.webm", data, audio.content_type or "audio/webm")},
                timeout=60,
            )

        from fastapi.concurrency import run_in_threadpool

        try:
            response = await run_in_threadpool(call)
        except requests.RequestException as exc:
            logger.warning("ElevenLabs STT request failed: %s", exc)
            return JSONResponse({"error": "Could not reach ElevenLabs."}, status_code=502)
        if response.status_code != 200:
            return JSONResponse({"error": _eleven_error(response)}, status_code=502)
        text = (response.json().get("text") or "").strip()
        return JSONResponse({"text": text})

    # ------------------------------------------------------ approval dialog
    @router.get("/system/pending", dependencies=session)
    def system_pending(request: Request):
        require_local(request)
        return JSONResponse({"pending": approvals.pending()})

    @router.post("/system/approve", dependencies=session)
    async def system_approve(request: Request):
        require_local(request)
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body."}, status_code=400)
        approval_id = payload.get("id") if isinstance(payload, dict) else None
        approved = payload.get("approved") if isinstance(payload, dict) else None
        if not isinstance(approval_id, str) or not isinstance(approved, bool):
            return JSONResponse({"error": "Send {id: string, approved: boolean}."}, status_code=400)
        if not approvals.resolve(approval_id, approved):
            return JSONResponse({"error": "That request already expired."}, status_code=404)
        return JSONResponse({"ok": True})

    return router
