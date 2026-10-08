"""ScribeProof FastAPI entrypoint."""

from __future__ import annotations

# IMPORTANT (Windows): import torch before paddle to avoid DLL conflicts.
import torch  # noqa: F401

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from application.pipeline import DocumentProcessor
from infrastructure.storage.artifact_store import ArtifactStore
from infrastructure.storage.repository import Repository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scribeproof")

DATA_DIR = Path(os.getenv("SCRIBEPROOF_DATA_DIR", Path(__file__).parent / "data"))
DB_URL = os.getenv(
    "SCRIBEPROOF_DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'db' / 'scribeproof.db'}",
)
ARTIFACT_ROOT = Path(os.getenv("SCRIBEPROOF_ARTIFACT_ROOT", DATA_DIR / "artifacts"))

_repo: Repository | None = None
_store: ArtifactStore | None = None
_processor: DocumentProcessor | None = None


def get_repo() -> Repository:
    assert _repo is not None
    return _repo


def get_store() -> ArtifactStore:
    assert _store is not None
    return _store


def get_processor() -> DocumentProcessor:
    assert _processor is not None
    return _processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _repo, _store, _processor
    (DATA_DIR / "db").mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    _repo = Repository(DB_URL)
    _store = ArtifactStore(ARTIFACT_ROOT)
    _processor = DocumentProcessor(_repo, _store)
    logger.info("ScribeProof ready — db=%s artifacts=%s", DB_URL, ARTIFACT_ROOT)

    warmup = os.getenv("SCRIBEPROOF_WARMUP_MODELS", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    if warmup:
        import threading

        def _warm() -> None:
            logger.info(
                "Warming up PaddleOCR + TrOCR (first run may download weights)…"
            )
            try:
                status = _processor.warmup_models()
                logger.info("Model warmup status: %s", status)
            except Exception as e:
                logger.exception("Model warmup failed: %s", e)

        threading.Thread(target=_warm, name="model-warmup", daemon=True).start()

    yield


app = FastAPI(
    title="ScribeProof",
    description="Provenance-preserving handwriting intelligence for HNX26EPS04",
    version="1.0.0",
    lifespan=lifespan,
)

origins = os.getenv(
    "SCRIBEPROOF_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Artifact-SHA256", "X-Transformation", "X-Source-Artifact-Id"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
