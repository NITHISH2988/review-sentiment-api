"""Train and evaluate a TF-IDF/logistic-regression model from a labeled CSV."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .model import create_pipeline


def train(
    data_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
    transformer_model: str | None = None,
) -> dict:
    data = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("CSV must include 'text' and 'label' columns")
    data = data.loc[:, ["text", "label"]].dropna()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"].str.len() > 0]
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError(
            "label must contain both numeric classes 0 (negative) and 1 (positive)"
        )
    if data["label"].value_counts().min() < 2:
        raise ValueError("Need at least two examples per class for a stratified split")

    train_text, test_text, train_labels, test_labels = train_test_split(
        data["text"],
        data["label"].astype(int),
        test_size=test_size,
        random_state=seed,
        stratify=data["label"],
    )
    model = create_pipeline().fit(train_text, train_labels)
    predictions = model.predict(test_text)
    metrics = {
        "rows_after_cleaning": len(data),
        "train_rows": len(train_text),
        "test_rows": len(test_text),
        "seed": seed,
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=[0, 1],
            target_names=["negative", "positive"],
            output_dict=True,
            zero_division=0,
        ),
    }
    if transformer_model:
        from .transformer import TransformerPredictor

        transformer = TransformerPredictor(transformer_model)
        transformer_predictions = transformer.predict_batch(test_text.tolist())
        metrics["transformer_comparison"] = {
            "model": transformer_model,
            "evaluation": "same held-out split; inference only",
            "accuracy": float(accuracy_score(test_labels, transformer_predictions)),
            "confusion_matrix": confusion_matrix(
                test_labels, transformer_predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                test_labels,
                transformer_predictions,
                labels=[0, 1],
                target_names=["negative", "positive"],
                output_dict=True,
                zero_division=0,
            ),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="CSV with text,label columns"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--transformer-model",
        help="Optional pretrained model ID; inference-only comparison on the same held-out split",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.data,
                args.output_dir,
                args.test_size,
                args.seed,
                args.transformer_model,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
