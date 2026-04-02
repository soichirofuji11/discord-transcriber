"""FastAPI WebSocket server for broadcasting transcription results."""
import asyncio
import json
from queue import Queue as ThreadQueue

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

# Connected WebSocket clients
connected_clients: set[WebSocket] = set()

# Thread-safe queue for receiving messages from the processing thread
_message_queue: ThreadQueue = ThreadQueue()


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
        # Non-blocking poll with async sleep to avoid blocking the event loop
        while not _message_queue.empty():
            try:
                message = _message_queue.get_nowait()
                await _broadcast(message)
            except Exception:
                break
        await asyncio.sleep(0.05)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(_broadcast_worker())


@app.get("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())
