"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  exportUrl,
  getDocumentResult,
  patchWord,
} from "@/lib/api-client";
import type { DocumentResult, WordResult } from "@/lib/types";
import { JobPoller } from "@/components/job-status/JobPoller";
import { PageCanvas } from "@/components/document-viewer/PageCanvas";
import { EvidencePanel } from "@/components/evidence-panel/EvidencePanel";
import { ExtractPanel } from "@/components/playground/ExtractPanel";
import { ParsePanel } from "@/components/playground/ParsePanel";
import { TranscriptEditor } from "@/components/transcription-editor/TranscriptEditor";

type Tool = "parse" | "extract" | "review";

export function ReviewWorkspace({
  documentId,
  initialStatus,
}: {
  documentId: string;
  initialStatus: string;
}) {
  const [ready, setReady] = useState(initialStatus === "completed");
  const [result, setResult] = useState<DocumentResult | null>(null);
  const [selectedWordId, setSelectedWordId] = useState<string | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>("extract");
  const [lowConfidenceOnly, setLowConfidenceOnly] = useState(false);

  const load = useCallback(async () => {
    try {
      const r = await getDocumentResult(documentId);
      setResult(r);
      setReady(true);
      setLoadError(null);
      if (!selectedWordId && r.words[0]) {
        setSelectedWordId(r.words[0].id);
      }
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Failed to load result");
    }
  }, [documentId, selectedWordId]);

  useEffect(() => {
    if (ready) load();
  }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedWord: WordResult | null = useMemo(() => {
    if (!result || !selectedWordId) return null;
    return result.words.find((w) => w.id === selectedWordId) ?? null;
  }, [result, selectedWordId]);

  const page = result?.pages[pageIndex];

  const stats = useMemo(() => {
    if (!result) return null;
    const words = result.words;
    return {
      total: words.length,
      accepted: words.filter((w) => w.decision_state === "ACCEPTED").length,
      review: words.filter((w) => w.decision_state === "REVIEW_REQUIRED").length,
      illegible: words.filter((w) => w.decision_state === "ILLEGIBLE").length,
      crossed: words.filter((w) => w.decision_state === "CROSSED_OUT").length,
    };
  }, [result]);

  async function handleEdit(word: WordResult, text: string) {
    await patchWord(word.id, {
      text,
      decision_state: "ACCEPTED",
      actor: "user",
    });
    await load();
    setSelectedWordId(word.id);
  }

  if (!ready) {
    return (
      <div className="shell-dotgrid min-h-[calc(100vh-3.5rem)] bg-canvas-50 px-4 py-16">
        <JobPoller documentId={documentId} onComplete={() => setReady(true)} />
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="mx-auto max-w-xl px-6 py-16">
        <div className="surface p-6 text-signal-illegible">{loadError}</div>
      </div>
    );
  }

  if (!result || !page) {
    return (
      <div className="px-6 py-16 text-center text-slateink-500">Loading playground…</div>
    );
  }

  const pageWords = result.words.filter((w) => w.page_id === page.id);

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] flex-col bg-canvas-50">
      {/* ADE-style top bar */}
      <div className="border-b border-slateink-200 bg-white">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-5">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="truncate text-sm font-semibold text-slateink-900">
                {result.filename}
              </h1>
              <span className="rounded-full bg-signal-accept/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-signal-accept">
                parsed
              </span>
              {result.category && (
                <span className="rounded-full bg-brand-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-brand-700">
                  {(result.category || "unknown").replace(/_/g, " ")}
                  {result.category_confidence
                    ? ` · ${Math.round(result.category_confidence * 100)}%`
                    : ""}
                </span>
              )}
            </div>
            <p className="font-mono text-[11px] text-slateink-400">{documentId}</p>
          </div>

          <div className="flex items-center gap-1 rounded-lg bg-slateink-100 p-1">
            {(
              [
                ["parse", "Parse"],
                ["extract", "Extract"],
                ["review", "Review"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setTool(id)}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
                  tool === id
                    ? "bg-white text-slateink-900 shadow-soft"
                    : "text-slateink-500 hover:text-slateink-800"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-slateink-200 bg-white px-2.5 py-1.5 text-xs text-slateink-600">
              <input
                type="checkbox"
                checked={lowConfidenceOnly}
                onChange={(e) => setLowConfidenceOnly(e.target.checked)}
                className="rounded border-slateink-300 text-brand-700 focus:ring-brand-500"
              />
              Low-confidence only
            </label>
            {(["json", "md", "txt"] as const).map((fmt) => (
              <a
                key={fmt}
                href={exportUrl(documentId, fmt)}
                target="_blank"
                rel="noreferrer"
                className="rounded-lg border border-slateink-200 bg-white px-2.5 py-1.5 text-xs font-medium uppercase tracking-wide text-slateink-600 hover:border-slateink-300"
              >
                {fmt}
              </a>
            ))}
          </div>
        </div>

        {stats && (
          <div className="mx-auto flex max-w-[1600px] flex-wrap gap-4 border-t border-slateink-100 px-4 py-2 text-[11px] text-slateink-500 sm:px-5">
            <span>
              <strong className="text-slateink-800">{stats.total}</strong> fields
            </span>
            <span>
              <strong className="text-slateink-800">
                {result.line_items?.length ?? 0}
              </strong>{" "}
              line items
            </span>
            <span className="text-signal-accept">{stats.accepted} accepted</span>
            <span className="text-signal-review">{stats.review} review</span>
            <span className="text-signal-illegible">{stats.illegible} illegible</span>
            <span className="text-signal-crossed">{stats.crossed} crossed-out</span>
            {result.pages.length > 1 && (
              <span className="ml-auto flex gap-1">
                {result.pages.map((p, i) => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => setPageIndex(i)}
                    className={`rounded px-2 py-0.5 ${
                      i === pageIndex
                        ? "bg-slateink-900 text-white"
                        : "bg-slateink-100 text-slateink-600"
                    }`}
                  >
                    p{i + 1}
                  </button>
                ))}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Three-pane playground */}
      <div className="mx-auto grid w-full max-w-[1600px] flex-1 gap-0 lg:grid-cols-12">
        <section className="border-b border-slateink-200 bg-white lg:col-span-5 lg:border-b-0 lg:border-r">
          <div className="border-b border-slateink-100 px-4 py-3">
            <h2 className="text-sm font-semibold text-slateink-900">Document</h2>
            <p className="text-xs text-slateink-500">
              Visual grounding · toggle preprocess variants · click a box
            </p>
          </div>
          <div className="p-4">
            <PageCanvas
              page={page}
              words={pageWords}
              selectedWordId={selectedWordId}
              highlightLowConfidence={lowConfidenceOnly}
              onSelectWord={setSelectedWordId}
              onSelectLine={(lineId) => {
                const w = pageWords.find((x) => x.line_id === lineId);
                if (w) setSelectedWordId(w.id);
              }}
            />
          </div>
        </section>

        <section className="flex min-h-[420px] flex-col border-b border-slateink-200 bg-white lg:col-span-4 lg:border-b-0 lg:border-r">
          {tool === "parse" && (
            <ParsePanel
              result={result}
              words={pageWords}
              selectedWordId={selectedWordId}
              onSelectWord={setSelectedWordId}
            />
          )}
          {tool === "extract" && (
            <ExtractPanel
              result={result}
              words={pageWords}
              selectedWordId={selectedWordId}
              onSelectWord={setSelectedWordId}
              lowConfidenceOnly={lowConfidenceOnly}
            />
          )}
          {tool === "review" && (
            <div className="p-4">
              <TranscriptEditor
                words={pageWords}
                selectedWordId={selectedWordId}
                onSelectWord={setSelectedWordId}
                onEditWord={handleEdit}
              />
            </div>
          )}
        </section>

        <section className="min-h-[360px] bg-white lg:col-span-3">
          <div className="border-b border-slateink-100 px-4 py-3">
            <h2 className="text-sm font-semibold text-slateink-900">Citations</h2>
            <p className="text-xs text-slateink-500">
              Crop · candidates · score breakdown
            </p>
          </div>
          <div className="p-4">
            <EvidencePanel result={result} word={selectedWord} />
          </div>
        </section>
      </div>
    </div>
  );
}
