from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class DocumentCreateResponse(BaseModel):
    document_id: str
    status: str
    filename: str
    source_artifact_id: str
    sha256: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    stage: str
    progress: float
    message: str
    error_message: Optional[str] = None
    page_count: int = 0


class WordPatchRequest(BaseModel):
    text: Optional[str] = None
    decision_state: Optional[str] = None
    actor: str = "user"


class WordPatchResponse(BaseModel):
    word_id: str
    text: str
    decision_state: str
    correction_id: str


class BoundingBoxSchema(BaseModel):
    x: float
    y: float
    width: float
    height: float


class WordResultSchema(BaseModel):
    id: str
    text: str
    decision_state: str
    confidence: float
    uncertainty: float
    final_score: float
    alternatives: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    crop_artifact_id: Optional[str] = None
    decision_reason: str = ""
    is_high_risk_entity: bool = False
    reading_order: int = 0
    bbox: Optional[BoundingBoxSchema] = None
    region_type: str = "main_handwriting"
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    line_id: str
    page_id: str
    selected_hypothesis_id: Optional[str] = None


class HypothesisSchema(BaseModel):
    id: str
    text: str
    normalized_text: str
    rank: int
    sequence_score: float
    visual_score: float
    model_name: str
    model_version: str
    source_crop_id: str
    image_variant: str


class LineSchema(BaseModel):
    id: str
    reading_order: int
    bbox: BoundingBoxSchema
    polygon: list[list[float]] = Field(default_factory=list)
    region_id: str
    is_crossed_out_candidate: bool = False
    crossed_out_score: float = 0.0
    crop_artifact_ids: dict[str, str] = Field(default_factory=dict)
    region_type: Optional[str] = None


class RegionSchema(BaseModel):
    id: str
    region_type: str
    bbox: BoundingBoxSchema
    linked_region_id: Optional[str] = None
    crop_artifact_id: Optional[str] = None


class PageSchema(BaseModel):
    id: str
    page_index: int
    width: int
    height: int
    quality: dict[str, Any]
    artifact_ids: dict[str, str]
    preprocessing_settings: dict[str, Any]
    regions: list[RegionSchema] = Field(default_factory=list)
    lines: list[LineSchema] = Field(default_factory=list)


class DocumentGraphSchema(BaseModel):
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class LayoutRegionSchema(BaseModel):
    region_type: str
    description: str = ""
    bbox_norm: Optional[list[float]] = None
    raw_type: Optional[str] = None


class LineItemSchema(BaseModel):
    id: str
    label: str
    value: str
    kind: str = "field"
    source: str = "ocr"
    confidence: float = 0.0
    decision_state: Optional[str] = None
    region_type: Optional[str] = None
    script_mode: Optional[str] = None


class DocumentResultResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    pages: list[PageSchema]
    words: list[WordResultSchema]
    crossed_out_words: list[WordResultSchema]
    active_text: str
    markdown: str
    graph: DocumentGraphSchema
    hypotheses_by_line: dict[str, list[HypothesisSchema]] = Field(default_factory=dict)
    category: str = "unknown"
    category_confidence: float = 0.0
    summary: str = ""
    layout_summary: list[LayoutRegionSchema] = Field(default_factory=list)
    line_items: list[LineItemSchema] = Field(default_factory=list)


class DocumentMetaResponse(BaseModel):
    id: str
    filename: str
    media_type: str
    status: str
    page_count: int
    source_artifact_id: str
    created_at: str
    updated_at: str
    error_message: Optional[str] = None
    category: Optional[str] = None
    category_confidence: Optional[float] = None
