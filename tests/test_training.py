import pandas as pd
import pytest

from review_sentiment.predictor import load_model
from review_sentiment.train import train


@pytest.fixture
def reviews():
    return pd.DataFrame(
        {
            "text": [
                "excellent product",
                "great quality",
                "love this",
                "works very well",
                "highly recommend",
                "happy with purchase",
                "terrible item",
                "poor quality",
                "hate this",
                "does not work",
                "would not recommend",
                "very disappointed",
                "excellent service",
                "great experience",
                "broken on arrival",
                "awful service",
                "good value",
                "pleasant surprise",
                "waste of money",
                "arrived damaged",
            ],
            "label": [1] * 10 + [0] * 10,
        }
    )


def test_train_writes_model_and_held_out_metrics(tmp_path, reviews):
    data_path = tmp_path / "reviews.csv"
    reviews.to_csv(data_path, index=False)
    output_dir = tmp_path / "artifacts"

    metrics = train(data_path, output_dir)

    assert (output_dir / "model.joblib").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert metrics["rows_after_cleaning"] == 20
    assert metrics["train_rows"] + metrics["test_rows"] == 20
    assert len(metrics["confusion_matrix"]) == 2
    assert callable(load_model(output_dir / "model.joblib").predict)


def test_train_requires_both_classes(tmp_path):
    data_path = tmp_path / "reviews.csv"
    pd.DataFrame({"text": ["fine", "nice"], "label": [1, 1]}).to_csv(
        data_path, index=False
    )
    with pytest.raises(ValueError, match="both numeric classes"):
        train(data_path, tmp_path / "artifacts")
