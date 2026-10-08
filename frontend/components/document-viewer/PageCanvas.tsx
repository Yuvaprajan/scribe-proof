"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { artifactUrl, decisionColor } from "@/lib/api-client";
import type { LineResult, PageResult, WordResult } from "@/lib/types";

const VARIANTS = [
  "raw",
  "normalized",
  "deskewed",
  "clahe",
  "adaptive_binarized",
  "denoised",
] as const;

export function PageCanvas({
  page,
  words,
  selectedWordId,
  onSelectWord,
  onSelectLine,
  highlightLowConfidence = false,
}: {
  page: PageResult;
  words: WordResult[];
  selectedWordId: string | null;
  onSelectWord: (wordId: string) => void;
  onSelectLine: (lineId: string) => void;
  highlightLowConfidence?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [variant, setVariant] = useState<(typeof VARIANTS)[number]>("deskewed");
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const wordByLine = useMemo(() => {
    const m = new Map<string, WordResult>();
    for (const w of words) m.set(w.line_id, w);
    return m;
  }, [words]);

  useEffect(() => {
    const id = page.artifact_ids[variant] || page.artifact_ids.raw;
    if (!id) return;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => setImg(image);
    image.src = artifactUrl(id);
  }, [page.artifact_ids, variant]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const maxW = canvas.parentElement?.clientWidth || 720;
    const scale = Math.min(1, maxW / img.width);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    for (const line of page.lines) {
      const word = wordByLine.get(line.id);
      const isLow =
        !!word &&
        (word.decision_state !== "ACCEPTED" || word.uncertainty > 0.28);
      if (highlightLowConfidence && word && !isLow) {
        continue;
      }

      const color = word
        ? decisionColor(word.decision_state)
        : line.is_crossed_out_candidate
          ? "#64748b"
          : line.region_type === "margin_note"
            ? "#0284c7"
            : "#0891b2";
      const selected = !!(word && word.id === selectedWordId);

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = selected ? 3 : highlightLowConfidence && isLow ? 2.5 : 1.5;
      ctx.globalAlpha = selected ? 0.95 : highlightLowConfidence && isLow ? 0.9 : 0.7;
      if (line.region_type === "margin_note") {
        ctx.setLineDash([6, 4]);
      }
      const { x, y, width, height } = line.bbox;
      ctx.strokeRect(x * scale, y * scale, width * scale, height * scale);
      if (selected || (highlightLowConfidence && isLow)) {
        ctx.fillStyle = color + (selected ? "28" : "14");
        ctx.fillRect(x * scale, y * scale, width * scale, height * scale);
      }
      ctx.restore();
    }
  }, [img, page.lines, selectedWordId, wordByLine, highlightLowConfidence]);

  function hitTest(clientX: number, clientY: number): LineResult | null {
    const canvas = canvasRef.current;
    if (!canvas || !img) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const maxW = canvas.parentElement?.clientWidth || 720;
    const scale = Math.min(1, maxW / img.width);
    const x = ((clientX - rect.left) * scaleX) / scale;
    const y = ((clientY - rect.top) * scaleY) / scale;
    for (let i = page.lines.length - 1; i >= 0; i--) {
      const L = page.lines[i];
      const b = L.bbox;
      if (x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height) {
        return L;
      }
    }
    return null;
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        {VARIANTS.map((v) => (
          <button
            key={v}
            type="button"
            onClick={() => setVariant(v)}
            className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition ${
              variant === v
                ? "bg-slateink-900 text-white"
                : "bg-slateink-100 text-slateink-600 hover:bg-slateink-200"
            }`}
          >
            {v}
          </button>
        ))}
      </div>
      <div className="overflow-auto rounded-lg border border-slateink-200 bg-canvas-50">
        <canvas
          ref={canvasRef}
          className="max-w-full cursor-crosshair"
          onClick={(e) => {
            const line = hitTest(e.clientX, e.clientY);
            if (!line) return;
            onSelectLine(line.id);
            const word = wordByLine.get(line.id);
            if (word) onSelectWord(word.id);
          }}
        />
      </div>
      <dl className="grid grid-cols-2 gap-2 text-[11px] text-slateink-500 sm:grid-cols-3">
        <Metric label="Quality" value={page.quality.quality_class ?? "—"} />
        <Metric label="Blur" value={page.quality.blur_score?.toFixed(1) ?? "—"} />
        <Metric label="Skew°" value={page.quality.skew_angle?.toFixed(2) ?? "—"} />
        <Metric
          label="Contrast"
          value={page.quality.contrast_score?.toFixed(1) ?? "—"}
        />
        <Metric label="Noise" value={page.quality.noise_score?.toFixed(1) ?? "—"} />
        <Metric label="Lines" value={String(page.lines.length)} />
      </dl>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="uppercase tracking-wide">{label}</dt>
      <dd className="font-mono text-slateink-900">{value}</dd>
    </div>
  );
}
