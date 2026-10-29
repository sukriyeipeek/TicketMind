"""API boundary tests use a stub; model quality is tested separately."""

from pathlib import Path
import pickle

from fastapi.testclient import TestClient
import pytest

import ticketmind.api as api


@pytest.fixture
def ready_app(monkeypatch):
    class StubService:
        seen: list[str] = []

        def __init__(self, model_path: Path):
            self.model_path = model_path

        def analyze(self, text: str) -> dict:
            self.seen.append(text)
            return {
                "category": "iade",
                "model_score": 0.4,
                "score_type": "uncalibrated_model_score",
                "needs_review": True,
                "review_reasons": ["synthetic_demo_model"],
                "entities": [],
                "model_version": "test-stub",
            }

    monkeypatch.setattr(api, "TicketService", StubService)
    return api.create_app(Path("not-loaded-by-stub.joblib")), StubService


def test_ready_health_and_preserved_original_text(ready_app):
    app, stub = ready_app
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ready",
            "model_loaded": True,
            "dataset_type": "synthetic_demo",
        }
        text = "  İADE: kulaklığımı geri göndermek istiyorum.  "
        response = client.post("/analyze", json={"text": text})
        assert response.status_code == 200
        assert stub.seen == [text]
        assert response.json()["needs_review"] is True
        assert response.json()["score_type"] == "uncalibrated_model_score"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": " \t\n "},
        {"text": 42},
        {"text": True},
        {"text": None},
        {"text": ["iade"]},
        {"text": "x" * 4001},
        {"text": "İade istiyorum", "model_path": "untrusted.joblib"},
    ],
)
def test_invalid_requests_do_not_reach_model(ready_app, payload):
    app, stub = ready_app
    with TestClient(app) as client:
        response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    assert stub.seen == []


def test_missing_model_is_not_ready_without_implicit_training(monkeypatch, tmp_path):
    calls: list[Path] = []

    class MissingService:
        def __init__(self, model_path: Path):
            calls.append(model_path)
            raise FileNotFoundError("private path should not be exposed")

    monkeypatch.setattr(api, "TicketService", MissingService)
    model_path = tmp_path / "missing.joblib"
    with TestClient(api.create_app(model_path)) as client:
        health = client.get("/health")
        response = client.post("/analyze", json={"text": "Kargom gelmedi."})
    assert health.status_code == 503
    assert health.json()["status"] == "not_ready"
    assert health.json()["model_loaded"] is False
    assert response.status_code == 503
    assert "python -m ticketmind.train" in response.json()["detail"]
    assert "private path" not in response.text
    assert calls == [model_path]
    assert not model_path.exists()


def test_environment_model_path_is_resolved_at_app_creation(monkeypatch, ready_app):
    _, stub = ready_app
    paths: list[Path] = []
    original_init = stub.__init__

    def capture_init(self, model_path):
        paths.append(model_path)
        original_init(self, model_path)

    monkeypatch.setattr(stub, "__init__", capture_init)
    monkeypatch.setenv("TICKETMIND_MODEL_PATH", "custom/baseline.joblib")
    with TestClient(api.create_app()) as client:
        assert client.get("/health").status_code == 200
    assert paths == [Path("custom/baseline.joblib")]


def test_corrupt_local_model_is_not_ready(monkeypatch, tmp_path):
    class CorruptService:
        def __init__(self, model_path):
            raise pickle.UnpicklingError("artifact is invalid")

    monkeypatch.setattr(api, "TicketService", CorruptService)
    with TestClient(api.create_app(tmp_path / "corrupt.joblib")) as client:
        assert client.get("/health").status_code == 503
        response = client.post("/analyze", json={"text": "Ödeme alınmadı."})
        assert response.status_code == 503
        assert "artifact is invalid" not in response.text
