"""FastAPI app: serves the page, the library API and the camera WebSocket.

Run with:  uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .landmarks import ROOT, HolisticTracker, ensure_model
from .library import Library
from .session import Session, empty_landmarks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("libras")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(data_dir: Path | None = None, tracker_factory=HolisticTracker) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("LIBRAS_DATA_DIR", ROOT / "data"))
    if tracker_factory is HolisticTracker:
        ensure_model()
    library = Library(data_dir)
    log.info("Library: %d signs, %d examples (%s)", len(library.signs),
             len(library.classifier.examples), data_dir)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        port = os.environ.get("LIBRAS_PORT", "8000")
        log.info("READY -> open http://localhost:%s in Chrome", port)
        yield

    app = FastAPI(title="Libras Recognizer", lifespan=lifespan)
    app.state.library = library
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    def health():
        return {"ok": True, "signs": len(library.signs),
                "examples": len(library.classifier.examples)}

    @app.get("/api/signs")
    def list_signs():
        return {"signs": library.summary()}

    @app.delete("/api/signs/{label}")
    def delete_sign(label: str):
        if not library.delete_sign(label):
            raise HTTPException(404, f"No sign called {label}")
        return {"ok": True}

    @app.get("/api/samples/{sid}")
    def get_sample(sid: str):
        try:
            return library.sample_json(sid)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "Sample not found")

    @app.delete("/api/samples/{sid}")
    def delete_sample(sid: str):
        if not library.delete_sample(sid):
            raise HTTPException(404, "Sample not found")
        return {"ok": True}

    @app.get("/api/export")
    def export_library():
        data = library.export_zip()
        name = f"libras-library-{datetime.now():%Y%m%d-%H%M}.zip"
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.post("/api/import")
    async def import_library(request: Request):
        data = await request.body()
        if not data:
            raise HTTPException(400, "Empty upload")
        try:
            result = await asyncio.to_thread(library.import_zip, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return result

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_running_loop()
        # One thread per connection: MediaPipe is always called from the same thread.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="session")
        session = None
        try:
            session = await loop.run_in_executor(executor, Session, library, tracker_factory)
            await ws.send_json({"type": "hello", "signs": len(library.signs),
                                "settings": session.settings})
            while True:
                text = await ws.receive_text()
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "warning", "message": "Bad message"})
                    continue
                try:
                    out = await loop.run_in_executor(executor, session.handle, msg)
                except Exception as e:  # keep the connection alive on errors
                    log.exception("error handling %s", msg.get("type"))
                    out = [{"type": "error", "message": f"{type(e).__name__}: {e}"}]
                    if msg.get("type") == "frame":
                        out.append(empty_landmarks())
                for m in out:
                    await ws.send_json(m)
        except WebSocketDisconnect:
            pass
        finally:
            if session is not None:
                executor.submit(session.close)
            executor.shutdown(wait=False)

    return app


app = create_app()
