"use client";

import type { DocumentResult, WordResult } from "@/lib/types";

export function ParsePanel({
  result,
  words,
  selectedWordId,
  onSelectWord,
}: {
  result: DocumentResult;
  words: WordResult[];
  selectedWordId: string | null;
  onSelectWord: (id: string) => void;
}) {
  const sorted = [...words].sort((a, b) => a.reading_order - b.reading_order);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slateink-100 px-4 py-3">
        <h2 className="text-sm font-semibold text-slateink-900">Parse</h2>
        <p className="text-xs text-slateink-500">
          Layout-aware Markdown with decision annotations
        </p>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="space-y-2 font-mono text-[13px] leading-6 text-slateink-800">
          {sorted.map((w) => {
            const selected = w.id === selectedWordId;
            let rendered = w.text;
            if (w.decision_state === "CROSSED_OUT") {
              rendered = `~~${w.text}~~`;
            } else if (w.decision_state === "ILLEGIBLE") {
              rendered = `**[ILLEGIBLE]**`;
            } else if (w.decision_state === "REVIEW_REQUIRED") {
              rendered = `==${w.text}==`;
            }
            return (
              <button
                key={w.id}
                type="button"
                onClick={() => onSelectWord(w.id)}
                className={`block w-full rounded px-2 py-1 text-left transition ${
                  selected ? "bg-brand-50 ring-1 ring-brand-400" : "hover:bg-canvas-50"
                } ${w.decision_state === "CROSSED_OUT" ? "text-slateink-400 line-through" : ""}`}
              >
                {rendered}
              </button>
            );
          })}
        </div>
        {result.crossed_out_words.length > 0 && (
          <div className="mt-6 rounded-lg border border-slateink-200 bg-canvas-50 p-3 text-xs text-slateink-600">
            <p className="font-semibold text-slateink-800">Crossed-out (excluded)</p>
            <p className="mt-1">
              {result.crossed_out_words.map((w) => w.text).join(" · ")}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
