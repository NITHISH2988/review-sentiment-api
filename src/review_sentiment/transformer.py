"""Optional inference-only adapter for a pretrained transformer classifier."""

from collections.abc import Callable

DEFAULT_TRANSFORMER_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class TransformerPredictor:
    def __init__(
        self,
        model_name: str = DEFAULT_TRANSFORMER_MODEL,
        pipeline_factory: Callable | None = None,
    ):
        if pipeline_factory is None:
            try:
                from transformers import pipeline
            except ImportError as error:
                raise RuntimeError(
                    "Install the optional transformer dependencies with "
                    "python -m pip install -e '.[transformer]'"
                ) from error
            pipeline_factory = pipeline
        self.model_name = model_name
        self._pipeline = pipeline_factory("text-classification", model=model_name)

    @staticmethod
    def _convert(result: dict) -> dict[str, str | float]:
        label = str(result["label"]).casefold()
        if label not in {"positive", "negative"}:
            raise ValueError(
                f"Transformer returned unsupported label {result['label']!r}"
            )
        return {"label": label, "confidence": float(result["score"])}

    def predict_sentiment(self, text: str) -> dict[str, str | float]:
        result = self._pipeline(text, truncation=True, max_length=512)[0]
        return self._convert(result)

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[int]:
        results = self._pipeline(
            texts, truncation=True, max_length=512, batch_size=batch_size
        )
        converted = [self._convert(result) for result in results]
        return [int(result["label"] == "positive") for result in converted]
