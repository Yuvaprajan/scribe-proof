"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { uploadDocument } from "@/lib/api-client";

export function UploadForm() {
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
    <form onSubmit={onSubmit} className="panel animate-rise space-y-5 p-6">
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
        className={`rounded-2xl border-2 border-dashed px-6 py-14 text-center transition ${
          dragOver
            ? "border-accent bg-accent/10"
            : "border-ink-300/50 bg-white/50"
        }`}
      >
        <p className="font-display text-xl text-ink-900">
          {file ? file.name : "Drop handwriting scan here"}
        </p>
        <p className="mt-2 text-sm text-ink-600">
          or choose a file from disk
        </p>
        <label className="mt-5 inline-block cursor-pointer rounded-full bg-ink-900 px-5 py-2 text-sm text-parchment-50">
          Browse
          <input
            type="file"
            accept=".png,.jpg,.jpeg,.pdf,image/png,image/jpeg,application/pdf"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
      </div>

      {error && (
        <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-signal-illegible">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={busy || !file}
        className="w-full rounded-full bg-accent px-5 py-3 text-sm font-semibold text-ink-950 disabled:opacity-50"
      >
        {busy ? "Uploading…" : "Start digitizing"}
      </button>
    </form>
  );
}
