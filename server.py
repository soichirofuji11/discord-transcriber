"""FastAPI WebSocket server for broadcasting transcription results."""
import json

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Connected WebSocket clients
connected_clients: set[WebSocket] = set()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except Exception:
        connected_clients.discard(websocket)


async def broadcast(message: dict):
    """Send a message to all connected clients."""
    data = json.dumps(message, ensure_ascii=False)
    dead: set[WebSocket] = set()
    for client in connected_clients:
        try:
            await client.send_text(data)
        except Exception:
            dead.add(client)
    connected_clients -= dead


@app.get("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())
