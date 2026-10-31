# Review Sentiment API

Compare a TF-IDF sentiment baseline with optional pretrained transformer inference, then serve predictions through FastAPI.

## Problem

The project demonstrates a reproducible binary text-classification workflow and a small prediction API. The dataset is supplied by the user because no review corpus with suitable documented terms is bundled.

## Features

- TF-IDF unigrams/bigrams and balanced logistic-regression baseline.
- Stratified held-out evaluation and saved metrics/artifact.
- Optional pretrained transformer inference comparison on the same test examples (no fine-tuning).
- Validated FastAPI endpoint with lazy model loading, explicit backend selection and error responses.
- Docker image definition for serving a previously trained baseline artifact.

## Architecture

```mermaid
flowchart LR
  A[User CSV] --> B[Clean rows and validate binary labels]
  B --> C[Stratified train/test split]
  C --> D[TF-IDF + logistic regression fit]
  C --> E[Optional pretrained transformer inference]
  D --> F[Metrics + joblib pipeline]
  E --> G[Comparable held-out metrics]
  F --> H[FastAPI cached TF-IDF model]
  I[Transformer configuration] --> J[FastAPI cached transformer]
  H --> K[/predict]
  J --> K
```

**Stack:** Python, pandas, scikit-learn, Hugging Face Transformers/PyTorch optional extra, FastAPI/Pydantic, joblib, Docker, pytest and Ruff.

## Models and data

For module-by-module walkthroughs, reproduction steps, and interview concepts, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

The baseline is a scikit-learn pipeline: whitespace/case normalization, TF-IDF word unigrams and bigrams, then balanced logistic regression. Training data is supplied by the user as CSV; no review corpus is bundled. Verify the dataset license and terms before use.

CSV columns:

```csv
text,label
"The product arrived quickly and works well",1
"The item broke immediately",0
```

Labels are numeric `0` (negative) and `1` (positive). The training command makes a stratified held-out split, records accuracy and per-class metrics, and persists the fitted TF-IDF model. Tests use small synthetic fixtures and do not claim quality on actual reviews.

## Train/evaluate the baseline

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m review_sentiment.train --data path\to\reviews.csv --seed 42
```

To compare with a pretrained transformer on the **same held-out split**, install the optional extra and pass a model ID:

```powershell
python -m pip install -e ".[dev,transformer]"
python -m review_sentiment.train --data path\to\reviews.csv --seed 42 --transformer-model distilbert/distilbert-base-uncased-finetuned-sst-2-english
```

The transformer comparison is inference-only; it is not fine-tuned on your CSV. The configured model downloads to the Hugging Face cache and runs locally after download. Its model card identifies it as an English SST-2 fine-tune under Apache-2.0 and documents known bias/limitations; confirm those terms and suitability for your use. [Model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

## Serve predictions

```powershell
uvicorn review_sentiment.api:app --reload
```

Open `http://127.0.0.1:8000/docs`. `GET /health` checks process health. `POST /predict` accepts `{"text":"..."}` and returns `label` plus the winning class score. That score is not calibrated. Blank and overlong inputs are rejected. The baseline artifact defaults to `artifacts/model.joblib`; set `SENTIMENT_MODEL_PATH` to change it.

The optional transformer backend is selected with `SENTIMENT_BACKEND=transformer` and `SENTIMENT_TRANSFORMER_MODEL`; otherwise the API uses the TF-IDF artifact. Model initialization is lazy and cached. Missing or invalid model state returns HTTP 503.

## Docker

Build after training the baseline model, then mount its artifact directory read-only:

```powershell
docker build -t review-sentiment-api .
docker run --rm -p 8000:8000 -v "${PWD}/artifacts:/app/artifacts:ro" review-sentiment-api
```

Docker Engine was unavailable during this quality pass, so the image build and container run instructions remain unverified.

## Tests and CI

```powershell
python -m pytest -q
```

Tests cover preprocessing, both class labels, prediction confidence, joblib artifact loading, missing/corrupt models, request validation, held-out training output, optional transformer adapter behavior using a fake pipeline, and API response handling. CI on Python 3.11 runs Ruff lint and formatting checks, this suite without downloading transformer weights, and a pip-audit scan of runtime dependencies.

## Trade-offs and limitations

TF-IDF with logistic regression is compact, fast, and comparatively interpretable, but relies on sparse n-gram overlap and has limited contextual understanding. A transformer can model richer context but uses more memory, takes longer to load, and inherits its training data's domain and bias limits. No real review dataset results are documented here because no licensed corpus has been selected. Sentiment is subjective and the endpoint is not suitable for consequential decisions.

## Roadmap

- Select a licensed review dataset and report a reproducible baseline result.
- Add duplicate/group-aware evaluation when the dataset has product or reviewer IDs.
- Measure probability calibration and startup latency before proposing serving optimizations.

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for source walkthroughs and the baseline/transformer distinctions.

## License

MIT for this code. Training datasets and pretrained models have separate terms and are not included.
