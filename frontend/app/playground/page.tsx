import { PlaygroundUpload } from "@/components/playground/PlaygroundUpload";

export default function PlaygroundPage() {
  return (
    <div className="shell-dotgrid min-h-[calc(100vh-3.5rem)] bg-canvas-50">
      <div className="mx-auto max-w-3xl px-4 py-14 sm:px-6">
        <div className="animate-fade-up mb-8 space-y-2">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand-700">
            Playground
          </p>
          <h1 className="font-display text-4xl text-slateink-950">
            Parse. Extract. Ground.
          </h1>
          <p className="max-w-2xl text-slateink-600">
            Drop a handwritten PNG, JPG, or PDF. ScribeProof runs the local
            evidence pipeline and opens a visual playground with citations for
            every field.
          </p>
        </div>
        <PlaygroundUpload />
      </div>
    </div>
  );
}
