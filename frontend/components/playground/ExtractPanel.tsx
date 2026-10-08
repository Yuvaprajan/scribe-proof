"use client";

import { decisionColor } from "@/lib/api-client";
import type { DocumentResult, LineItem, WordResult } from "@/lib/types";

export function ExtractPanel({
  result,
  words,
  selectedWordId,
  onSelectWord,
  lowConfidenceOnly,
}: {
  result?: DocumentResult | null;
  words: WordResult[];
  selectedWordId: string | null;
  onSelectWord: (id: string) => void;
  lowConfidenceOnly: boolean;
}) {
  const category = result?.category || "unknown";
  const categoryConf = result?.category_confidence ?? 0;
  const summary = result?.summary || "";
  const layout = result?.layout_summary || [];
  const inventory = (result?.line_items || []).filter((it) =>
    lowConfidenceOnly ? (it.confidence ?? 1) < 0.72 : true
  );

  const sorted = [...words]
    .filter((w) => w.decision_state !== "CROSSED_OUT")
    .filter((w) =>
      lowConfidenceOnly
        ? w.decision_state !== "ACCEPTED" || w.uncertainty > 0.28
        : true
    )
    .sort((a, b) => a.reading_order - b.reading_order);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slateink-100 px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-sm font-semibold text-slateink-900">Extract</h2>
          <span className="rounded-full bg-brand-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-brand-700">
            {category.replace(/_/g, " ")}
            {categoryConf > 0 ? ` · ${Math.round(categoryConf * 100)}%` : ""}
          </span>
        </div>
        <p className="mt-1 text-xs text-slateink-500">
          {summary ||
            "Category · layout · inventory line items · grounded OCR fields"}
        </p>
      </div>

      <div className="flex-1 space-y-4 overflow-auto p-3">
        {layout.length > 0 && (
          <section>
            <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slateink-400">
              Layout
            </h3>
            <ul className="space-y-1.5">
              {layout.map((r, i) => (
                <li
                  key={`${r.region_type}-${i}`}
                  className="rounded-md border border-slateink-100 bg-slateink-50/60 px-2.5 py-1.5 text-xs text-slateink-700"
                >
                  <span className="font-mono text-[10px] uppercase text-slateink-400">
                    {r.region_type.replace(/_/g, " ")}
                  </span>
                  {r.description ? (
                    <span className="mt-0.5 block text-slateink-800">
                      {r.description}
                    </span>
                  ) : null}
                </li>
              ))}
            </ul>
          </section>
        )}

        <section>
          <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slateink-400">
            Line items
          </h3>
          <ul className="space-y-2">
            {inventory.length === 0 && sorted.length === 0 && (
              <li className="rounded-lg border border-dashed border-slateink-200 p-4 text-sm text-slateink-500">
                No fields match the current filter.
              </li>
            )}
            {inventory.map((it: LineItem, i: number) => {
              const color =
                it.decision_state
                  ? decisionColor(it.decision_state as WordResult["decision_state"])
                  : "#64748b";
              const wordMatch = words.find((w) => w.id === it.id);
              const selected = wordMatch?.id === selectedWordId;
              return (
                <li key={`${it.id}-${i}`}>
                  <button
                    type="button"
                    onClick={() => wordMatch && onSelectWord(wordMatch.id)}
                    className={`w-full rounded-lg border px-3 py-2.5 text-left transition ${
                      selected
                        ? "border-brand-500 bg-brand-50 shadow-soft"
                        : "border-slateink-200 bg-white hover:border-slateink-300"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-[10px] uppercase tracking-wide text-slateink-400">
                        {it.label || `item_${String(i + 1).padStart(2, "0")}`}
                      </span>
                      <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase text-slateink-500">
                        {it.kind}
                        {it.source ? ` · ${it.source}` : ""}
                      </span>
                    </div>
                    <p className="mt-1 text-sm font-medium text-slateink-900">
                      {it.value || "—"}
                    </p>
                    <div className="mt-2 flex items-center gap-2">
                      <div className="confidence-bar flex-1">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.max(4, (it.confidence || 0) * 100)}%`,
                            backgroundColor: color,
                          }}
                        />
                      </div>
                      <span className="font-mono text-[10px] text-slateink-500">
                        {Math.round((it.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </button>
                </li>
              );
            })}
            {inventory.length === 0 &&
              sorted.map((w, i) => {
                const color = decisionColor(w.decision_state);
                const selected = w.id === selectedWordId;
                return (
                  <li key={w.id}>
                    <button
                      type="button"
                      onClick={() => onSelectWord(w.id)}
                      className={`w-full rounded-lg border px-3 py-2.5 text-left transition ${
                        selected
                          ? "border-brand-500 bg-brand-50 shadow-soft"
                          : "border-slateink-200 bg-white hover:border-slateink-300"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-[10px] uppercase tracking-wide text-slateink-400">
                          field_{String(i + 1).padStart(2, "0")}
                        </span>
                        <span
                          className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase"
                          style={{ color, backgroundColor: color + "18" }}
                        >
                          {w.decision_state.replace("_", " ")}
                        </span>
                      </div>
                      <p className="mt-1 truncate text-sm font-medium text-slateink-900">
                        {w.text}
                      </p>
                      <div className="mt-2 flex items-center gap-2">
                        <div className="confidence-bar flex-1">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.max(4, w.confidence * 100)}%`,
                              backgroundColor: color,
                            }}
                          />
                        </div>
                        <span className="font-mono text-[10px] text-slateink-500">
                          {Math.round(w.confidence * 100)}%
                        </span>
                      </div>
                    </button>
                  </li>
                );
              })}
          </ul>
        </section>
      </div>
    </div>
  );
}
