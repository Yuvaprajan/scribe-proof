"use client";

import { decisionColor } from "@/lib/api-client";
import type { WordResult } from "@/lib/types";

export function TranscriptEditor({
  words,
  selectedWordId,
  onSelectWord,
  onEditWord,
}: {
  words: WordResult[];
  selectedWordId: string | null;
  onSelectWord: (id: string) => void;
  onEditWord: (word: WordResult, text: string) => void;
}) {
  const sorted = [...words].sort((a, b) => a.reading_order - b.reading_order);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slateink-900">Human review</h2>
          <p className="text-xs text-slateink-500">
            Click to cite · double-click to correct (immutable audit)
          </p>
        </div>
      </div>
      <div className="min-h-[280px] rounded-lg border border-slateink-200 bg-canvas-50 p-4 leading-8">
        {sorted.length === 0 && (
          <p className="text-sm text-slateink-500">No words yet.</p>
        )}
        {sorted.map((w) => {
          const color = decisionColor(w.decision_state);
          const selected = w.id === selectedWordId;
          const crossed = w.decision_state === "CROSSED_OUT";
          return (
            <span
              key={w.id}
              role="button"
              tabIndex={0}
              className="word-chip mb-1 mr-1"
              style={{
                backgroundColor: color + "18",
                color,
                textDecoration: crossed ? "line-through" : undefined,
                boxShadow: selected ? `0 0 0 2px ${color}` : undefined,
              }}
              onClick={() => onSelectWord(w.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter") onSelectWord(w.id);
              }}
              onDoubleClick={() => {
                const next = window.prompt(
                  "Edit text (immutable correction recorded)",
                  w.text
                );
                if (next != null && next !== w.text) onEditWord(w, next);
              }}
              title={`${w.decision_state} · conf ${w.confidence.toFixed(2)}`}
            >
              {w.text || "∅"}
            </span>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-3 text-xs text-slateink-500">
        <Legend color="#059669" label="Accepted" />
        <Legend color="#d97706" label="Review" />
        <Legend color="#dc2626" label="Illegible" />
        <Legend color="#64748b" label="Crossed-out" />
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}
