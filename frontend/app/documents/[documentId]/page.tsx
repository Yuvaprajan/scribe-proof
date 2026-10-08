import { getDocument } from "@/lib/api-client";
import { ReviewWorkspace } from "@/components/document-viewer/ReviewWorkspace";

export default async function DocumentPage({
  params,
}: {
  params: Promise<{ documentId: string }>;
}) {
  const { documentId } = await params;
  let status = "pending";
  let filename = documentId;
  try {
    const meta = await getDocument(documentId);
    status = meta.status;
    filename = meta.filename;
  } catch {
    // Client poller / workspace will surface errors
  }

  return (
    <div>
      <div className="sr-only">
        Document {filename} ({status})
      </div>
      <ReviewWorkspace documentId={documentId} initialStatus={status} />
    </div>
  );
}
