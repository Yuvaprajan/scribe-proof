"""Domain entities for ScribeProof provenance-preserving handwriting intelligence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DecisionState(str, Enum):
    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    ILLEGIBLE = "ILLEGIBLE"
    CROSSED_OUT = "CROSSED_OUT"


class RegionType(str, Enum):
    MAIN_HANDWRITING = "main_handwriting"
    PRINTED_TEXT = "printed_text"
    MARGIN_NOTE = "margin_note"
    TABLE = "table"
    CROSSED_OUT_CANDIDATE = "crossed_out_candidate"
    SIGNATURE = "signature"
    UNKNOWN = "unknown"


class DocumentCategory(str, Enum):
    PRESCRIPTION = "prescription"
    MEDICAL_REPORT = "medical_report"
    LAB_REPORT = "lab_report"
    FORM = "form"
    LETTER = "letter"
    ARCHITECTURE_DIAGRAM = "architecture_diagram"
    FLOWCHART = "flowchart"
    SKETCH = "sketch"
    RECEIPT = "receipt"
    ID_CARD = "id_card"
    HANDWRITTEN_NOTE = "handwritten_note"
    MIXED = "mixed"
    OTHER = "other"
    UNKNOWN = "unknown"


class QualityClass(str, Enum):
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


class Artifact(BaseModel):
    id: str = Field(default_factory=new_id)
    sha256: str
    path: str
    media_type: str
    source_artifact_id: Optional[str] = None
    transformation_name: str
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    page_id: Optional[str] = None
    document_id: Optional[str] = None


class Document(BaseModel):
    id: str = Field(default_factory=new_id)
    filename: str
    media_type: str
    source_artifact_id: str
    status: JobStatus = JobStatus.PENDING
    page_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    id: str = Field(default_factory=new_id)
    document_id: str
    page_index: int
    width: int
    height: int
    raw_artifact_id: str
    normalized_artifact_id: Optional[str] = None
    blur_score: Optional[float] = None
    contrast_score: Optional[float] = None
    skew_angle: Optional[float] = None
    resolution: Optional[float] = None
    noise_score: Optional[float] = None
    quality_class: Optional[QualityClass] = None
    artifact_ids: dict[str, str] = Field(default_factory=dict)
    preprocessing_settings: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Region(BaseModel):
    id: str = Field(default_factory=new_id)
    page_id: str
    region_type: RegionType
    bbox: BoundingBox
    linked_region_id: Optional[str] = None
    crop_artifact_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Line(BaseModel):
    id: str = Field(default_factory=new_id)
    page_id: str
    region_id: str
    bbox: BoundingBox
    reading_order: int
    polygon: list[list[float]] = Field(default_factory=list)
    crop_artifact_ids: dict[str, str] = Field(default_factory=dict)
    is_crossed_out_candidate: bool = False
    crossed_out_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Hypothesis(BaseModel):
    id: str = Field(default_factory=new_id)
    recognition_run_id: str
    text: str
    normalized_text: str
    rank: int
    sequence_score: float
    visual_score: float
    model_name: str
    model_version: str
    source_crop_id: str
    image_variant: str
    token_confidences: list[float] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class RecognitionRun(BaseModel):
    id: str = Field(default_factory=new_id)
    line_id: str
    provider: str
    model_name: str
    model_version: str
    image_variant: str
    source_crop_id: str
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class WordEvidence(BaseModel):
    id: str = Field(default_factory=new_id)
    line_id: str
    page_id: str
    document_id: str
    selected_hypothesis_id: Optional[str] = None
    text: str
    decision_state: DecisionState
    confidence: float
    uncertainty: float
    final_score: float
    alternatives: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    crop_artifact_id: Optional[str] = None
    decision_reason: str = ""
    is_high_risk_entity: bool = False
    reading_order: int = 0
    bbox: Optional[BoundingBox] = None
    region_type: RegionType = RegionType.MAIN_HANDWRITING
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Decision(BaseModel):
    id: str = Field(default_factory=new_id)
    word_evidence_id: str
    state: DecisionState
    confidence: float
    uncertainty: float
    reason: str
    thresholds_used: dict[str, float] = Field(default_factory=dict)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class ReviewTask(BaseModel):
    id: str = Field(default_factory=new_id)
    document_id: str
    word_evidence_id: str
    status: str = "open"
    priority: int = 0
    created_at: datetime = Field(default_factory=utcnow)


class CorrectionRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    word_evidence_id: str
    previous_text: str
    new_text: str
    previous_state: DecisionState
    new_state: DecisionState
    actor: str = "user"
    created_at: datetime = Field(default_factory=utcnow)


class AuditEvent(BaseModel):
    id: str = Field(default_factory=new_id)
    document_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
