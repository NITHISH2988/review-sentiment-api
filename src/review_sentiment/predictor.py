"""Model artifact loading and prediction validation."""

from pathlib import Path

import joblib


def load_model(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    model = joblib.load(path)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Artifact does not expose predict()")
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Artifact does not expose predict_proba()")
    return model


def predict(model, text: str) -> dict[str, str | float]:
    probabilities = model.predict_proba([text])[0]
    labels = list(model.classes_)
    if len(labels) != len(probabilities) or len(labels) != 2:
        raise ValueError("Expected a binary classifier with two class probabilities")
    index = int(probabilities.argmax())
    label = labels[index]
    if label not in (0, 1):
        raise ValueError("Model classes must be encoded as 0 and 1")
    return {
        "label": "positive" if int(label) == 1 else "negative",
        "confidence": float(probabilities[index]),
    }
