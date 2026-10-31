import joblib
import pytest

from review_sentiment.model import create_pipeline
from review_sentiment.predictor import load_model, predict


@pytest.fixture
def fitted_model():
    return create_pipeline().fit(
        ["excellent delivery", "great product", "broken item", "terrible support"],
        [1, 1, 0, 0],
    )


def test_prediction_maps_model_class_and_confidence(fitted_model):
    result = predict(fitted_model, "excellent delivery")
    assert result["label"] == "positive"
    assert 0 <= result["confidence"] <= 1


def test_model_loader_round_trips_joblib_artifact(tmp_path, fitted_model):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_model, path)
    assert predict(load_model(path), "broken item")["label"] == "negative"


def test_model_loader_reports_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


def test_model_loader_rejects_invalid_artifact(tmp_path):
    path = tmp_path / "not-a-model.joblib"
    joblib.dump({"weights": []}, path)
    with pytest.raises(TypeError, match="predict"):
        load_model(path)
