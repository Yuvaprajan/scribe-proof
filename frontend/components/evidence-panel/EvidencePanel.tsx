"use client";

import { artifactUrl } from "@/lib/api-client";
import type { DocumentResult, WordResult } from "@/lib/types";

export function EvidencePanel({
  result,
  word,
}: {
  result: DocumentResult;
  word: WordResult | null;
}) {
  if (!word) {
    return (
      <div className="rounded-lg border border-dashed border-slateink-200 bg-canvas-50 p-5 text-sm text-slateink-500">
        Select a field or bounding box to inspect the crop, OCR candidates, and
        decision reason.
      </div>
    );
  }

  const hyps = result.hypotheses_by_line[word.line_id] || [];
  const selected = hyps.find((h) => h.id === word.selected_hypothesis_id);

  return (
    <div className="space-y-4">
      <div>
        <p className="font-mono text-[10px] text-slateink-400">{word.id}</p>
      </div>

      {word.crop_artifact_id && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={artifactUrl(word.crop_artifact_id)}
          alt="Source crop"
          className="max-h-28 w-full rounded-lg border border-slateink-200 bg-white object-contain"
        />
      )}

      <div className="rounded-lg border border-slateink-100 bg-canvas-50 p-3 text-sm">
        <div className="flex items-center justify-between gap-2">
          <span
            className="text-xs font-bold uppercase tracking-wide"
            style={{ color: stateColor(word.decision_state) }}
          >
            {word.decision_state}
          </span>
          {word.is_high_risk_entity && (
            <span className="rounded bg-amber-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-700">
              high-risk
            </span>
          )}
        </div>
        <p className="mt-2 text-xs leading-relaxed text-slateink-700">
          {word.decision_reason}
        </p>
        <dl className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
          <Metric label="Confidence" value={word.confidence} />
          <Metric label="Uncertainty" value={word.uncertainty} />
          <Metric label="Final score" value={word.final_score} />
          <Metric label="Region" value={word.region_type} raw />
        </dl>
      </div>

      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slateink-500">
          Selected vs OCR
        </h3>
        <p className="mt-1 text-sm">
          <span className="text-slateink-400">Value · </span>
          <span className="font-medium text-slateink-900">{word.text}</span>
        </p>
        {selected && (
          <p className="mt-0.5 font-mono text-[11px] text-slateink-500">
            hyp · {selected.text} · {selected.image_variant} · rank {selected.rank}
          </p>
        )}
      </div>

      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slateink-500">
          Candidates ({hyps.length})
        </h3>
        <ul className="mt-2 max-h-48 space-y-1 overflow-auto">
          {hyps.slice(0, 20).map((h) => (
            <li
              key={h.id}
              className={`rounded-md border px-2 py-1.5 text-xs ${
                h.id === word.selected_hypothesis_id
                  ? "border-brand-400 bg-brand-50"
                  : "border-slateink-100 bg-white"
              }`}
            >
              <div className="flex justify-between gap-2">
                <span className="font-medium text-slateink-900">
                  {h.text || "∅"}
                </span>
                <span className="font-mono text-slateink-400">
                  {h.visual_score.toFixed(3)}
                </span>
              </div>
              <div className="mt-0.5 text-[10px] text-slateink-400">
                {h.image_variant} · {h.model_name.split("/").pop()}
              </div>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slateink-500">
          Score breakdown
        </h3>
        <ul className="mt-2 space-y-1 font-mono text-[10px] text-slateink-600">
          {Object.entries(word.score_breakdown).map(([k, v]) => (
            <li key={k} className="flex justify-between gap-2">
              <span>{k}</span>
              <span>{typeof v === "number" ? v.toFixed(3) : String(v)}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  raw,
}: {
  label: string;
  value: number | string;
  raw?: boolean;
}) {
  return (
    <div>
      <dt className="uppercase tracking-wide text-slateink-400">{label}</dt>
      <dd className="font-mono text-slateink-900">
        {raw || typeof value === "string" ? value : Number(value).toFixed(3)}
      </dd>
    </div>
  );
}

function stateColor(state: string): string {
  switch (state) {
    case "ACCEPTED":
      return "#059669";
    case "REVIEW_REQUIRED":
      return "#d97706";
    case "ILLEGIBLE":
      return "#dc2626";
    case "CROSSED_OUT":
      return "#64748b";
    default:
      return "#334155";
  }
}
