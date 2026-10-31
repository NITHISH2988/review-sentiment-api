"""FastAPI service for a locally trained binary sentiment classifier."""

import logging
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .predictor import load_model, predict
from .transformer import DEFAULT_TRANSFORMER_MODEL, TransformerPredictor

logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.getenv("SENTIMENT_MODEL_PATH", "artifacts/model.joblib"))
app = FastAPI(title="Review Sentiment API", version="0.2.0")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def text_must_contain_non_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must contain a non-whitespace character")
        return value


@lru_cache(maxsize=1)
def get_model():
    if os.getenv("SENTIMENT_BACKEND", "tfidf").casefold() == "transformer":
        return TransformerPredictor(
            os.getenv("SENTIMENT_TRANSFORMER_MODEL", DEFAULT_TRANSFORMER_MODEL)
        )
    return load_model(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_sentiment(request: PredictionRequest) -> dict[str, str | float]:
    try:
        model = get_model()
        if isinstance(model, TransformerPredictor):
            return model.predict_sentiment(request.text)
        return predict(model, request.text)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Train it with the documented CSV workflow first.",
        ) from error
    except Exception as error:
        logger.exception("Sentiment model could not produce a prediction")
        raise HTTPException(
            status_code=503, detail="Sentiment model is unavailable"
        ) from error
