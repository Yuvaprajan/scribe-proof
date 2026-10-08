export type DecisionState =
  | "ACCEPTED"
  | "REVIEW_REQUIRED"
  | "ILLEGIBLE"
  | "CROSSED_OUT";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface WordResult {
  id: string;
  text: string;
  decision_state: DecisionState;
  confidence: number;
  uncertainty: number;
  final_score: number;
  alternatives: string[];
  evidence_ids: string[];
  crop_artifact_id?: string | null;
  decision_reason: string;
  is_high_risk_entity: boolean;
  reading_order: number;
  bbox?: BoundingBox | null;
  region_type: string;
  score_breakdown: Record<string, number>;
  line_id: string;
  page_id: string;
  selected_hypothesis_id?: string | null;
}

export interface Hypothesis {
  id: string;
  text: string;
  normalized_text: string;
  rank: number;
  sequence_score: number;
  visual_score: number;
  model_name: string;
  model_version: string;
  source_crop_id: string;
  image_variant: string;
}

export interface LineResult {
  id: string;
  reading_order: number;
  bbox: BoundingBox;
  polygon: number[][];
  region_id: string;
  is_crossed_out_candidate: boolean;
  crossed_out_score: number;
  crop_artifact_ids: Record<string, string>;
  region_type?: string | null;
}

export interface RegionResult {
  id: string;
  region_type: string;
  bbox: BoundingBox;
  linked_region_id?: string | null;
  crop_artifact_id?: string | null;
}

export interface PageResult {
  id: string;
  page_index: number;
  width: number;
  height: number;
  quality: {
    blur_score?: number | null;
    contrast_score?: number | null;
    skew_angle?: number | null;
    resolution?: number | null;
    noise_score?: number | null;
    quality_class?: string | null;
  };
  artifact_ids: Record<string, string>;
  preprocessing_settings: Record<string, unknown>;
  regions: RegionResult[];
  lines: LineResult[];
}

export interface LayoutRegion {
  region_type: string;
  description: string;
  bbox_norm?: number[] | null;
  raw_type?: string | null;
}

export interface LineItem {
  id: string;
  label: string;
  value: string;
  kind: string;
  source: string;
  confidence: number;
  decision_state?: string | null;
  region_type?: string | null;
  script_mode?: string | null;
}

export interface DocumentResult {
  document_id: string;
  filename: string;
  status: string;
  pages: PageResult[];
  words: WordResult[];
  crossed_out_words: WordResult[];
  active_text: string;
  markdown: string;
  graph: {
    nodes: Array<Record<string, unknown>>;
    edges: Array<Record<string, unknown>>;
  };
  hypotheses_by_line: Record<string, Hypothesis[]>;
  category?: string;
  category_confidence?: number;
  summary?: string;
  layout_summary?: LayoutRegion[];
  line_items?: LineItem[];
}

export interface DocumentMeta {
  id: string;
  filename: string;
  media_type: string;
  status: string;
  page_count: number;
  source_artifact_id: string;
  created_at: string;
  updated_at: string;
  error_message?: string | null;
  category?: string | null;
  category_confidence?: number | null;
}

export interface DocumentStatus {
  document_id: string;
  status: string;
  stage: string;
  progress: number;
  message: string;
  error_message?: string | null;
  page_count: number;
}

export interface DocumentCreateResponse {
  document_id: string;
  status: string;
  filename: string;
  source_artifact_id: string;
  sha256: string;
}
