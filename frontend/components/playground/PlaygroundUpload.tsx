"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { uploadDocument } from "@/lib/api-client";

export function PlaygroundUpload() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a PNG, JPG, JPEG, or PDF file.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await uploadDocument(file);
      router.push(`/documents/${res.document_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setBusy(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="surface animate-fade-up overflow-hidden">
      <div className="flex items-center justify-between border-b border-slateink-100 px-4 py-3">
        <div className="flex gap-2">
          <span className="rounded-md bg-slateink-900 px-2.5 py-1 text-xs font-medium text-white">
            Parse
          </span>
          <span className="rounded-md bg-slateink-100 px-2.5 py-1 text-xs font-medium text-slateink-500">
            Extract
          </span>
          <span className="rounded-md bg-slateink-100 px-2.5 py-1 text-xs font-medium text-slateink-500">
            Review
          </span>
        </div>
        <span className="font-mono text-[11px] text-slateink-400">new job</span>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) setFile(f);
        }}
        className={`m-4 rounded-xl border-2 border-dashed px-6 py-16 text-center transition ${
          dragOver
            ? "border-brand-600 bg-brand-50"
            : "border-slateink-200 bg-canvas-50"
        }`}
      >
        <p className="text-lg font-semibold text-slateink-900">
          {file ? file.name : "Drop a handwritten document"}
        </p>
        <p className="mt-1 text-sm text-slateink-500">
          PNG · JPG · JPEG · PDF — stored by SHA-256, never overwritten
        </p>
        <label className="mt-6 inline-flex cursor-pointer rounded-lg border border-slateink-200 bg-white px-4 py-2 text-sm font-semibold text-slateink-800 shadow-soft hover:border-slateink-300">
          Browse files
          <input
            type="file"
            accept=".png,.jpg,.jpeg,.pdf,image/png,image/jpeg,application/pdf"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
      </div>

      {error && (
        <p className="mx-4 mb-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-signal-illegible">
          {error}
        </p>
      )}

      <div className="flex items-center justify-between border-t border-slateink-100 px-4 py-3">
        <p className="text-xs text-slateink-500">
          Local pipeline · optional verifier off by default
        </p>
        <button
          type="submit"
          disabled={busy || !file}
          className="rounded-lg bg-brand-700 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50 hover:bg-brand-800"
        >
          {busy ? "Uploading…" : "Run extraction"}
        </button>
      </div>
    </form>
  );
}
