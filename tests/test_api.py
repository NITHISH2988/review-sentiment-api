import joblib
import pytest
from fastapi.testclient import TestClient

from review_sentiment import api
from review_sentiment.model import create_pipeline


@pytest.fixture
def client_with_model(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model = create_pipeline().fit(
        ["excellent delivery", "great product", "broken item", "terrible support"],
        [1, 1, 0, 0],
    )
    joblib.dump(model, model_path)
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    api.get_model.cache_clear()
    yield TestClient(api.app)
    api.get_model.cache_clear()


def test_health(client_with_model):
    response = client_with_model.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_label_and_confidence(client_with_model):
    response = client_with_model.post("/predict", json={"text": "excellent delivery"})
    assert response.status_code == 200
    assert response.json()["label"] in {"positive", "negative"}
    assert 0 <= response.json()["confidence"] <= 1


@pytest.mark.parametrize("text", ["", "   \n", "x" * 5001])
def test_predict_rejects_invalid_text(client_with_model, text):
    assert client_with_model.post("/predict", json={"text": text}).status_code == 422


def test_predict_without_model_returns_service_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    api.get_model.cache_clear()
    response = TestClient(api.app).post("/predict", json={"text": "a review"})
    assert response.status_code == 503
    api.get_model.cache_clear()


def test_corrupt_model_returns_service_unavailable(tmp_path, monkeypatch):
    model_path = tmp_path / "corrupt.joblib"
    model_path.write_text("not a joblib file", encoding="utf-8")
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    api.get_model.cache_clear()
    response = TestClient(api.app).post("/predict", json={"text": "a review"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Sentiment model is unavailable"
    api.get_model.cache_clear()


def test_optional_transformer_backend_is_configurable(monkeypatch):
    class FakeTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict_sentiment(self, text):
            return {"label": "positive", "confidence": 0.75}

    monkeypatch.setenv("SENTIMENT_BACKEND", "transformer")
    monkeypatch.setenv("SENTIMENT_TRANSFORMER_MODEL", "test/model")
    monkeypatch.setattr(api, "TransformerPredictor", FakeTransformer)
    api.get_model.cache_clear()
    response = TestClient(api.app).post("/predict", json={"text": "a review"})
    assert response.status_code == 200
    assert response.json() == {"label": "positive", "confidence": 0.75}
    api.get_model.cache_clear()


def test_unknown_backend_returns_service_unavailable(monkeypatch):
    monkeypatch.setenv("SENTIMENT_BACKEND", "transfomer")
    api.get_model.cache_clear()

    response = TestClient(api.app).post("/predict", json={"text": "a review"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Sentiment model is unavailable"
    api.get_model.cache_clear()
