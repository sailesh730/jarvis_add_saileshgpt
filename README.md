# SaileshGPT + Jarvis (ElevenLabs voice and PC control)

Adds a third agent, **Jarvis · PC control**, plus ElevenLabs speech to your existing project.

## What you get
- **Talk to it:** mic button (tap, speak, it stops when you pause) uses ElevenLabs Scribe.
  The waveform button turns on **hands-free mode**: listen, think, speak, listen again.
- **It talks back:** the speaker button reads replies aloud with your ElevenLabs voice.
- **Controls this laptop (Jarvis agent):** open/close apps, open websites, volume, mute,
  media keys, brightness, screenshots, clipboard, notifications, list/search/read/write/move/delete
  files, run shell commands, type text, press shortcuts, lock/sleep/restart/shutdown, system status.
- Also fixes two things your code already referenced but did not have: `delete_user_conversation(s)`
  in `database.py` and `delete_thread_state` in `agent.py` (the delete buttons now work).

## Install (5 minutes)
1. Copy this folder's files into your project, overwriting `app.py`, `agent.py`, `database.py`
   and `templates/index.html`. New files include `system_tools.py`, `jarvis_routes.py`,
   `tools.py`, `rag.py` and `model_config.py`. Keep your own `.env` and `chroma.sqlite3` as they are.
2. `pip install -r requirements-jarvis.txt`
3. Add the lines from `.env.example` to your `.env` (at least `ELEVENLABS_API_KEY`).
   Create the key at elevenlabs.io > Developers > API Keys; if you restrict it, allow
   Text to Speech and Speech to Text.
4. `python app.py`, open http://127.0.0.1:8080 and pick **Jarvis · PC control** in the sidebar.
   Use `127.0.0.1` or `localhost` so the browser allows the microphone.

## Travel tools
The travel agent can look up country information, airports, recent flight status, forecasts,
exchange rates and trip budgets. Country, forecast and exchange-rate lookups use public APIs;
airport and flight lookups require `AVIATIONSTACK_API_KEY` in `.env` and an Aviationstack plan
that supports HTTPS. Flight results are live or recent operational data, not fares or future
schedules.

Uploaded-document search uses Google embeddings when `GOOGLE_API_KEY` is set. Otherwise it uses
Ollama's `nomic-embed-text` model; install/pull that model with Ollama, or set
`RAG_EMBEDDING_PROVIDER=google` or `RAG_EMBEDDING_PROVIDER=ollama` in `.env` to choose explicitly.

## Try saying
"How's my battery?" - "Open Spotify" - "Set volume to 30" - "Pause the music" -
"What's on my Desktop?" - "Take a screenshot" - "Close Chrome" (asks approval) -
"Create a file notes.txt on my Desktop saying buy milk" (asks approval)

## Safety (please read)
Giving an AI control of your computer is risky, so these guard rails are built in:
- **Approval dialog:** closing apps, shell commands, writing/moving/deleting files, typing,
  keyboard shortcuts and restart/shutdown pause until you click **Approve** on the page.
  The model cannot approve for itself. No answer in 90 s = denied. Deny is focused by default.
- **Local only:** Jarvis chat and the approval endpoints reject any request not from 127.0.0.1.
  Do not change the server host to 0.0.0.0 while using Jarvis.
- **Files:** limited to your home folder (or `JARVIS_ALLOWED_ROOTS`); `.ssh`, `.aws`, `.env`,
  browser profiles and similar are blocked. Deletes go to the Recycle Bin/Trash.
- **Shell filter:** obviously destructive commands (rm -rf, format, curl | bash, ...) are refused
  even if you approve. This is best effort; the approval dialog shows the exact command, so read it.
- **Prompt injection:** web pages/files the agent reads can contain hidden instructions. The
  approval step is your protection, so never approve something you did not ask for.
- **Hands-free mode:** the mic hears everything in the room (a video, a guest). Risky actions
  still need a click, but turn voice mode off when you step away.
- **Audit log:** every action, allowed or denied, is written to `data/jarvis_audit.log`.
- Shutdown/restart wait 30 s; say "cancel shutdown" to stop them. Move the mouse to a screen
  corner to abort any keyboard/mouse automation (pyautogui fail-safe).

## Which model?
Jarvis needs reliable tool calling. **Gemini** models do this well. Small local Ollama models
(for example 3-4B) often pick the wrong tool or skip the tool call; use 7B+ if you stay local.

## Notes by operating system
- **Windows:** best supported. `pip install pycaw comtypes` gives exact volume control
  (without it, volume uses keyboard volume keys).
- **macOS:** grant your terminal/Python Accessibility and Screen Recording permission
  (System Settings > Privacy & Security) or typing, shortcuts and screenshots fail.
- **Linux:** needs a desktop session (X11 works best). Screenshots may need `scrot`;
  volume uses `pactl`; media uses `playerctl`.

## Honest testing note
The code was tested here on Linux with a simulated ElevenLabs and a fake model: path blocking,
command filter, approve/deny/timeout, API endpoints, database deletes, and a full agent run that
waits for approval and then executes. The actions that touch a real desktop (opening Windows apps,
volume, notifications, keystrokes) could not be run in my environment, so try them once
yourself, starting with harmless ones like "open notepad".

## Tuning
All options are in `.env` (see `.env.example`). Add or remove abilities by editing the
`SYSTEM_TOOLS` list at the bottom of `system_tools.py`.
