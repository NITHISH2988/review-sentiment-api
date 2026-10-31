# Developer Guide: Review Sentiment API

This repository compares a trainable TF-IDF/logistic-regression baseline with an optional pretrained transformer inference adapter, then serves predictions with FastAPI.

## Architecture and data flow

```text
reviews.csv → train.train → clean/validate labels → stratified holdout split
                 ├─ TF-IDF + logistic regression fit on training rows
                 ├─ optional transformer inference on the same test rows
                 └─ metrics.json + model.joblib
FastAPI request → PredictionRequest validation → cached backend
                 ├─ TF-IDF artifact: predictor.predict
                 └─ TransformerPredictor → normalized label/score
```

`model.normalize_text()` collapses whitespace and case-folds while retaining punctuation for scikit-learn’s tokenizer. `create_pipeline()` fits TF-IDF word unigrams and bigrams, then balanced logistic regression. Keeping vectorizer and classifier in one scikit-learn pipeline ensures inference applies exactly the learned vocabulary and weighting. `train.train()` reads the user-provided CSV, removes empty rows, requires both numeric classes, makes a seeded stratified split, and evaluates accuracy and per-class metrics. The optional `TransformerPredictor` compares a pretrained model on that same held-out text; it does not fine-tune on the CSV.

`api.py` validates and trims request text, lazily loads one configured backend through `get_model()` and caches it. `SENTIMENT_BACKEND` accepts only `tfidf` or `transformer`. The TF-IDF path calls `predictor.load_model()` and `predict()`. The transformer path wraps a Hugging Face text-classification pipeline. Missing artifacts or model failures produce 503; malformed request data is rejected by FastAPI/Pydantic with 422. `/health` reports process health, not model readiness.

## Directory map and entry points

- `src/review_sentiment/model.py`: `normalize_text` and `create_pipeline`.
- `src/review_sentiment/train.py`: CSV cleaning, split, metrics, persistence, and CLI; run `python -m review_sentiment.train`.
- `src/review_sentiment/predictor.py`: trusted artifact loading and binary label/probability mapping.
- `src/review_sentiment/transformer.py`: optional `TransformerPredictor` and default model ID.
- `src/review_sentiment/api.py`: request schema, cached backend selection, `/health`, and `/predict`; run Uvicorn.
- `tests/`: preprocessing/training, artifact and prediction validation, transformer adapter with fake model, and API validation/error behavior.
- `Dockerfile`: serves a previously trained baseline artifact mounted at runtime.
- `.github/workflows/ci.yml`: Python 3.11, Ruff, tests without model downloads, and runtime dependency auditing.

## Reproduce a baseline experiment and serve it

Create your own CSV with `text,label` columns, numeric labels 0/1, and enough examples from both classes. First verify that you have permission to use the corpus:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m review_sentiment.train --data path\to\reviews.csv --seed 42
python -m pytest -q
uvicorn review_sentiment.api:app --reload
```

Training writes `artifacts/model.joblib` and `artifacts/metrics.json`; it prints the observed report. Since no corpus is bundled, there is no universal expected score. The metric is valid only for the selected dataset, its label quality, and this one split. Make a request at `/docs` or send `{"text":"The product arrived quickly and works well"}` to `/predict`. The response is a label and winning class score, not a calibrated probability.

For a pretrained transformer comparison, install `python -m pip install -e ".[dev,transformer]` and rerun training with `--transformer-model distilbert/distilbert-base-uncased-finetuned-sst-2-english`. It downloads model files and requires internet access on first use; it is inference-only and English/SST-2 oriented. To serve it, set `SENTIMENT_BACKEND=transformer` and optionally `SENTIMENT_TRANSFORMER_MODEL` before starting the API. To serve TF-IDF, use `SENTIMENT_BACKEND=tfidf` and ensure `SENTIMENT_MODEL_PATH` points to a trusted compatible artifact.

## Concepts to be ready to explain

1. **Normalization consistency:** training and serving both pass raw text through the serialized TF-IDF pipeline’s `normalize_text` preprocessor.
2. **TF-IDF:** term frequency is downweighted by document frequency; n-grams provide local phrase information but not deep semantic context.
3. **Logistic regression:** a linear classifier on sparse features; class balancing changes class-weighted loss, not the data distribution.
4. **Stratified split:** preserves approximate class ratios but does not prevent near-duplicate leakage between partitions.
5. **Transformer attention:** self-attention builds context-dependent token representations; the project consumes a pretrained classifier rather than implementing attention itself.
6. **Fine-tuning versus inference:** the transformer comparison only predicts the held-out examples; no weights are adapted to the uploaded dataset.
7. **Evaluation:** confusion matrix and class report expose errors that accuracy can hide, especially under imbalance.
8. **Lazy model lifecycle:** first request initializes and caches the model, reducing repeat load latency but leaving `/health` distinct from readiness.
9. **Model artifact trust:** joblib deserialization can execute code; only load artifacts you created or otherwise trust.
10. **Score interpretation:** predicted class scores are not calibrated certainty.

## Tests, CI, and extension points

Run `python -m pytest -q`, `python -m ruff check .`, and `python -m ruff format --check .`. CI uses fake transformer adapters and never downloads transformer weights. Tests prove contract handling and pipeline behavior, not real-world sentiment accuracy. Docker build/run instructions are in README; they require a trained artifact. Container execution should not be claimed until Docker Engine is available and the commands have been tested.

High-value future work includes grouped splitting when multiple reviews come from the same product/user, duplicate detection before splitting, calibrated probabilities, and a clearly licensed representative dataset. Preserve comparable splits if evaluating the two approaches and document that the transformer model’s training domain may not match product reviews.

## Current limits and troubleshooting

- No review dataset is included and no empirical sentiment quality claim is made.
- A pretrained transformer is English and domain-limited, and costs more memory/time than TF-IDF.
- If `/predict` returns 503 on TF-IDF, train the artifact and check `SENTIMENT_MODEL_PATH`.
- If backend selection fails, set exactly `tfidf` or `transformer`; other values fail explicitly.
- A 422 means input schema/length validation failed. Blank and over-5000-character text is rejected.
- A transformer startup/download error may mean the optional extra is missing, network is unavailable, or the configured model ID cannot be loaded.

