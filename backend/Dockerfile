FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SCRIBEPROOF_DATA_DIR=/data \
    SCRIBEPROOF_ARTIFACT_ROOT=/data/artifacts \
    SCRIBEPROOF_DATABASE_URL=sqlite:////data/db/scribeproof.db \
    SCRIBEPROOF_OCR_MODE=trocr \
    SCRIBEPROOF_ALLOW_MOCK_FALLBACK=false \
    SCRIBEPROOF_USE_PADDLE=true \
    SCRIBEPROOF_WARMUP_MODELS=true \
    SCRIBEPROOF_TROCR_MODEL=microsoft/trocr-large-handwritten \
    SCRIBEPROOF_VERIFIER_ENABLED=false \
    HF_HOME=/data/hf_cache \
    TRANSFORMERS_CACHE=/data/hf_cache \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p /data/db /data/artifacts /data/hf_cache

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=8 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
