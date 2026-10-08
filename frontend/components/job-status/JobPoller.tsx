"use client";

import { useEffect, useState } from "react";
import { getDocumentStatus } from "@/lib/api-client";
import type { DocumentStatus } from "@/lib/types";

const STAGES = [
  "queued",
  "loading",
  "preprocessing",
  "detection",
  "recognition",
  "completed",
];

export function JobPoller({
  documentId,
  onComplete,
}: {
  documentId: string;
  onComplete: () => void;
}) {
  const [status, setStatus] = useState<DocumentStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function tick() {
      try {
        const s = await getDocumentStatus(documentId);
        if (cancelled) return;
        setStatus(s);
        if (s.status === "completed") {
          onComplete();
          return;
        }
        if (s.status === "failed") {
          setError(s.error_message || "Processing failed");
          return;
        }
        timer = setTimeout(tick, 1000);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Status poll failed");
        }
      }
    }

    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [documentId, onComplete]);

  const progress = Math.round((status?.progress ?? 0) * 100);
  const stage = status?.stage ?? "queued";

  return (
    <div className="mx-auto max-w-lg">
      <div className="surface overflow-hidden">
        <div className="border-b border-slateink-100 px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-brand-700">
            Agentic job
          </p>
          <h2 className="mt-1 text-xl font-semibold text-slateink-900">
            Running extraction
          </h2>
          <p className="mt-1 text-sm text-slateink-500">
            {status?.message || "Starting pipeline…"}
          </p>
        </div>
        <div className="space-y-4 px-5 py-5">
          <div className="h-2 overflow-hidden rounded-full bg-slateink-100">
            <div
              className="h-full rounded-full bg-brand-600 transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <ol className="space-y-2">
            {STAGES.filter((s) => s !== "completed").map((s) => {
              const idx = STAGES.indexOf(s);
              const cur = STAGES.indexOf(stage);
              const done = cur > idx || stage === "completed";
              const active = stage === s;
              return (
                <li
                  key={s}
                  className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm ${
                    active
                      ? "bg-brand-50 text-brand-800"
                      : done
                        ? "text-slateink-700"
                        : "text-slateink-400"
                  }`}
                >
                  <span
                    className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold ${
                      done
                        ? "bg-signal-accept text-white"
                        : active
                          ? "bg-brand-600 text-white"
                          : "bg-slateink-100 text-slateink-400"
                    }`}
                  >
                    {done ? "✓" : idx + 1}
                  </span>
                  <span className="capitalize">{s}</span>
                  {active && (
                    <span className="ml-auto font-mono text-xs">{progress}%</span>
                  )}
                </li>
              );
            })}
          </ol>
          {error && (
            <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-signal-illegible">
              {error}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
