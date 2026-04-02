"""FastAPI WebSocket server for broadcasting transcription results."""
import asyncio
import json
from contextlib import asynccontextmanager
from queue import Queue as ThreadQueue
from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from plugins.session_store import Session

# Connected WebSocket clients
connected_clients: set[WebSocket] = set()

# Thread-safe queue for receiving messages from the processing thread
_message_queue: ThreadQueue = ThreadQueue()

# Session and summarizer references (set from main.py)
_session: Optional[Session] = None
_summarizer = None  # Optional[Summarizer]

# Additional startup tasks registered by main.py
_extra_startup_coros: list = []


def register_startup_task(coro_func):
    """Register an async function to run at startup."""
    _extra_startup_coros.append(coro_func)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(_broadcast_worker())
    for coro_func in _extra_startup_coros:
        asyncio.create_task(coro_func())
    yield
    # Shutdown (nothing needed)


app = FastAPI(lifespan=lifespan)


def set_session(session: Session):
    global _session
    _session = session


def set_summarizer(summarizer):
    global _summarizer
    _summarizer = summarizer


def enqueue_message(message: dict):
    """Thread-safe: push a message to be broadcast (called from any thread)."""
    _message_queue.put(message)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        connected_clients.discard(websocket)


async def _broadcast(message: dict):
    """Send a message to all connected clients."""
    data = json.dumps(message, ensure_ascii=False)
    dead: set[WebSocket] = set()
    for client in connected_clients:
        try:
            await client.send_text(data)
        except Exception:
            dead.add(client)
    connected_clients.difference_update(dead)


async def _broadcast_worker():
    """Background task: poll the thread-safe queue and broadcast messages."""
    while True:
        while not _message_queue.empty():
            try:
                message = _message_queue.get_nowait()
                await _broadcast(message)
            except Exception:
                break
        await asyncio.sleep(0.05)


@app.get("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# --- Summary API ---

class SummarizeRequest(BaseModel):
    style: str = "overview"


@app.post("/api/summarize")
async def api_summarize(request: SummarizeRequest):
    if _session is None:
        return {"error": "No active session"}
    if _summarizer is None:
        return {"error": "Summarizer not enabled (set GEMINI_API_KEY and use --summarize)"}

    transcript = _session.get_full_text(include_translation=True)
    if not transcript.strip():
        return {"error": "No transcript data yet"}

    try:
        summary = await _summarizer.summarize_long_session(
            transcript, style=request.style
        )
        return {
            "summary": summary,
            "style": request.style,
            "duration_minutes": round(_session.get_duration_minutes(), 1),
            "entries": _session.get_entry_count(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/session/save")
async def api_save_session():
    if _session is None:
        return {"error": "No active session"}
    path = _session.save()
    return {"saved": _session.session_id, "path": path}


@app.get("/api/session/stats")
async def api_session_stats():
    if _session is None:
        return {"error": "No active session"}
    return {
        "session_id": _session.session_id,
        "entries": _session.get_entry_count(),
        "duration_minutes": round(_session.get_duration_minutes(), 1),
    }
