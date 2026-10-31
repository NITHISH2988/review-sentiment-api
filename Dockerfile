FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd -m app
USER app
EXPOSE 8000
CMD ["uvicorn","review_sentiment.api:app","--host","0.0.0.0","--port","8000"]
