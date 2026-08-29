import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from api import main as api_main
from src.model import SimpleCNN


def _make_app(tmp_path, monkeypatch):
    """Point the already-imported api.main app at a fresh untrained checkpoint.

    Metrics (Counter/Histogram) are registered once at module import time, so
    tests reuse the single imported module instead of reloading it (reloading
    would re-register the same metric names and raise a Prometheus
    CollectorRegistry duplicate-timeseries error).
    """
    model_path = tmp_path / "model.pt"
    torch.save(SimpleCNN().state_dict(), model_path)
    monkeypatch.setattr(api_main, "MODEL_PATH", str(model_path))
    return api_main.app


def test_health_endpoint(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


def test_predict_endpoint(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        image = Image.new("RGB", (300, 300), color=(50, 120, 200))
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/predict", files={"file": ("test.jpg", buf, "image/jpeg")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["label"] in ("cat", "dog")
        assert set(body["probabilities"].keys()) == {"cat", "dog"}


def test_predict_rejects_non_image(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == 400


def test_metrics_endpoint(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"http_requests_total" in response.content
