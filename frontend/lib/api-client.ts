import type {
  DocumentCreateResponse,
  DocumentMeta,
  DocumentResult,
  DocumentStatus,
  WordResult,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

export function getApiBase(): string {
  return API_BASE;
}

export function artifactUrl(artifactId: string): string {
  return `${API_BASE}/api/artifacts/${artifactId}`;
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function uploadDocument(file: File): Promise<DocumentCreateResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/documents`, {
    method: "POST",
    body: form,
  });
  return parseJson(res);
}

export async function getDocument(documentId: string): Promise<DocumentMeta> {
  const res = await fetch(`${API_BASE}/api/documents/${documentId}`, {
    cache: "no-store",
  });
  return parseJson(res);
}

export async function getDocumentStatus(
  documentId: string
): Promise<DocumentStatus> {
  const res = await fetch(`${API_BASE}/api/documents/${documentId}/status`, {
    cache: "no-store",
  });
  return parseJson(res);
}

export async function getDocumentResult(
  documentId: string
): Promise<DocumentResult> {
  const res = await fetch(`${API_BASE}/api/documents/${documentId}/result`, {
    cache: "no-store",
  });
  return parseJson(res);
}

export async function patchWord(
  wordId: string,
  body: { text?: string; decision_state?: string; actor?: string }
): Promise<{ word_id: string; text: string; decision_state: string; correction_id: string }> {
  const res = await fetch(`${API_BASE}/api/words/${wordId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseJson(res);
}

export function exportUrl(documentId: string, fmt: "txt" | "md" | "json"): string {
  return `${API_BASE}/api/documents/${documentId}/export/${fmt}`;
}

export function decisionColor(state: WordResult["decision_state"]): string {
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
