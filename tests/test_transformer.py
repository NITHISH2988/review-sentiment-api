import pytest

from review_sentiment.transformer import TransformerPredictor


def fake_pipeline(task, model):
    assert task == "text-classification"
    assert model == "test/model"

    def classify(value, **kwargs):
        if isinstance(value, list):
            return [
                {"label": "POSITIVE", "score": 0.9},
                {"label": "NEGATIVE", "score": 0.8},
            ]
        return [{"label": "POSITIVE", "score": 0.9}]

    return classify


def test_transformer_prediction_maps_labels_and_scores():
    predictor = TransformerPredictor("test/model", pipeline_factory=fake_pipeline)
    assert predictor.predict_sentiment("fine") == {
        "label": "positive",
        "confidence": 0.9,
    }
    assert predictor.predict_batch(["good", "bad"]) == [1, 0]


def test_transformer_adapter_rejects_unknown_labels():
    predictor = TransformerPredictor(
        "test/model",
        pipeline_factory=lambda *args, **kwargs: (
            lambda *a, **k: [{"label": "neutral", "score": 0.7}]
        ),
    )
    with pytest.raises(ValueError, match="unsupported label"):
        predictor.predict_sentiment("mixed")
