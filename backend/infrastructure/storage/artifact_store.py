"""Content-addressed local filesystem artifact store (SHA-256)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import aiofiles

from domain.entities import Artifact, utcnow


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "blobs").mkdir(exist_ok=True)
        (self.root / "meta").mkdir(exist_ok=True)

    @staticmethod
    def sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _blob_path(self, sha256: str) -> Path:
        return self.root / "blobs" / sha256[:2] / sha256

    async def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        transformation_name: str,
        settings: Optional[dict[str, Any]] = None,
        source_artifact_id: Optional[str] = None,
        page_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Artifact:
        digest = self.sha256_bytes(data)
        path = self._blob_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            async with aiofiles.open(path, "wb") as f:
                await f.write(data)

        artifact = Artifact(
            sha256=digest,
            path=str(path),
            media_type=media_type,
            source_artifact_id=source_artifact_id,
            transformation_name=transformation_name,
            settings=settings or {},
            created_at=utcnow(),
            page_id=page_id,
            document_id=document_id,
        )
        meta_path = self.root / "meta" / f"{artifact.id}.json"
        async with aiofiles.open(meta_path, "w") as f:
            await f.write(artifact.model_dump_json())
        return artifact

    def put_bytes_sync(
        self,
        data: bytes,
        *,
        media_type: str,
        transformation_name: str,
        settings: Optional[dict[str, Any]] = None,
        source_artifact_id: Optional[str] = None,
        page_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Artifact:
        digest = self.sha256_bytes(data)
        path = self._blob_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)

        artifact = Artifact(
            sha256=digest,
            path=str(path),
            media_type=media_type,
            source_artifact_id=source_artifact_id,
            transformation_name=transformation_name,
            settings=settings or {},
            created_at=utcnow(),
            page_id=page_id,
            document_id=document_id,
        )
        meta_path = self.root / "meta" / f"{artifact.id}.json"
        meta_path.write_text(artifact.model_dump_json())
        return artifact

    def get_path(self, artifact: Artifact) -> Path:
        return Path(artifact.path)

    def read_bytes(self, artifact: Artifact) -> bytes:
        return Path(artifact.path).read_bytes()

    def write_meta(self, artifact: Artifact) -> None:
        meta_path = self.root / "meta" / f"{artifact.id}.json"
        meta_path.write_text(artifact.model_dump_json())
