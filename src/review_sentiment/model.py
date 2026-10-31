"""Traditional NLP baseline and its small, serializable preprocessing step."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def normalize_text(text: str) -> str:
    """Normalize whitespace and case; preserve punctuation for TF-IDF tokenization."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    min_df=1,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
