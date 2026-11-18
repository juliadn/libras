"""HTTP + WebSocket tests with the real MediaPipe model and a real photo."""
import base64
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.landmarks import MODEL_PATH
from app.library import Library
from app.main import create_app
from tests.synth import static_take

ASSET = Path(__file__).parent / "assets" / "person.jpg"
ASSET_URL = "https://storage.googleapis.com/mediapipe-assets/male_full_height_hands.jpg"


def person_jpeg() -> bytes:
    if not ASSET.exists():
        try:
            ASSET.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(ASSET_URL, ASSET)
        except OSError:
            pytest.skip("test photo not available (offline)")
    return ASSET.read_bytes()


@pytest.fixture
def client(tmp_path):
    if not MODEL_PATH.exists():
        pytest.skip("MediaPipe model not downloaded")
    lib = Library(tmp_path)
    lib.add_samples("A", "static", [static_take("A") for _ in range(2)])
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as c:
        yield c


def test_page_and_library_api(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
    assert client.get("/api/health").json()["signs"] == 1
    signs = client.get("/api/signs").json()["signs"]
    assert signs[0]["label"] == "A" and signs[0]["count"] == 2
    sid = signs[0]["samples"][0]["id"]
    assert client.get(f"/api/samples/{sid}").json()["frames"]

    exported = client.get("/api/export")
    assert exported.headers["content-type"] == "application/zip"
    assert client.delete(f"/api/samples/{sid}").json()["ok"]
    assert client.post("/api/import", content=exported.content).json()["added"] == 1
    assert client.post("/api/import", content=b"junk").status_code == 400
    assert client.delete("/api/signs/A").json()["ok"]
    assert client.delete("/api/signs/A").status_code == 404
    assert client.get("/api/samples/..%2F..%2Fetc").status_code == 404


def test_websocket_frames_return_landmarks(client):
    jpeg = base64.b64encode(person_jpeg()).decode()
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "settings", "threshold": 0.6})
        assert ws.receive_json()["threshold"] == 0.6
        def exchange(msg):
            """Send one message, return everything the server answered."""
            ws.send_json(msg)
            ws.send_json({"type": "ping"})
            got = []
            while True:
                m = ws.receive_json()
                if m["type"] == "pong":
                    return got
                got.append(m)

        for _ in range(3):
            replies = exchange({"type": "frame", "jpeg": jpeg})
            assert replies[0]["type"] == "landmarks"
        msg = replies[0]
        assert msg["hands"]["left"] and msg["hands"]["right"]
        assert len(msg["pose"]) == 11 and len(msg["face"]) == 40
        assert msg["rest_y"] is not None

        replies = exchange({"type": "frame", "jpeg": "bm90IGFuIGltYWdl"})   # not an image
        assert [m["type"] for m in replies][:2] == ["error", "landmarks"]
