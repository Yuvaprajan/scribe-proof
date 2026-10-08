export default function HomePage() {
  return (
    <div>
      {/* Hero — one composition, brand first */}
      <section className="relative overflow-hidden border-b border-slateink-200 bg-white">
        <div className="shell-dotgrid pointer-events-none absolute inset-0 opacity-70" />
        <div
          className="pointer-events-none absolute -right-24 top-0 h-[520px] w-[520px] rounded-full opacity-40"
          style={{
            background:
              "radial-gradient(circle, rgba(8,145,178,0.18) 0%, transparent 70%)",
          }}
        />
        <div className="relative mx-auto grid min-h-[calc(100vh-3.5rem)] max-w-[1400px] items-center gap-12 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:py-20">
          <div className="animate-fade-up max-w-xl space-y-6">
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-brand-700">
              Agentic document extraction
            </p>
            <h1 className="font-display text-5xl leading-[1.05] tracking-tight text-slateink-950 sm:text-6xl">
              ScribeProof
            </h1>
            <p className="text-lg leading-relaxed text-slateink-600">
              Vision-first handwriting intelligence with audit-ready
              traceability. Parse extreme bad handwriting into grounded text —
              never a silent guess.
            </p>
            <div className="flex flex-wrap gap-3 pt-1">
              <a
                href="/playground"
                className="rounded-lg bg-brand-700 px-5 py-2.5 text-sm font-semibold text-white shadow-soft transition hover:bg-brand-800"
              >
                Try the Playground
              </a>
              <a
                href="#product"
                className="rounded-lg border border-slateink-200 bg-white px-5 py-2.5 text-sm font-semibold text-slateink-800 transition hover:border-slateink-300"
              >
                See how it works
              </a>
            </div>
          </div>

          <div
            className="animate-fade-up relative"
            style={{ animationDelay: "100ms" }}
          >
            <div className="surface overflow-hidden">
              <div className="flex items-center justify-between border-b border-slateink-100 px-4 py-2.5">
                <div className="flex gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-slateink-200" />
                  <span className="h-2.5 w-2.5 rounded-full bg-slateink-200" />
                  <span className="h-2.5 w-2.5 rounded-full bg-slateink-200" />
                </div>
                <span className="font-mono text-[11px] text-slateink-400">
                  playground · parse + extract
                </span>
              </div>
              <div className="grid grid-cols-5 gap-0">
                <div className="col-span-3 border-r border-slateink-100 bg-canvas-50 p-4">
                  <div className="relative aspect-[4/3] overflow-hidden rounded-lg border border-slateink-200 bg-[linear-gradient(160deg,#f8fafc,#e2e8f0)]">
                    <div className="absolute left-[8%] top-[18%] h-7 w-[70%] rounded border-2 border-brand-600/80 bg-brand-500/10" />
                    <div className="absolute left-[10%] top-[38%] h-7 w-[55%] rounded border-2 border-signal-review/80 bg-amber-400/10" />
                    <div className="absolute left-[12%] top-[58%] h-7 w-[48%] rounded border-2 border-signal-illegible/70 bg-red-400/10" />
                    <div className="absolute left-[14%] top-[76%] h-6 w-[40%] rounded border-2 border-slateink-400/60 bg-slateink-400/10 line-through opacity-80" />
                  </div>
                </div>
                <div className="col-span-2 space-y-3 p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-wider text-slateink-400">
                    Extracted fields
                  </p>
                  <FieldPreview
                    label="date"
                    value="12/03/2026"
                    conf={0.91}
                    tone="accept"
                  />
                  <FieldPreview
                    label="dose"
                    value="500 mg"
                    conf={0.74}
                    tone="review"
                  />
                  <FieldPreview
                    label="line_3"
                    value="[ILLEGIBLE]"
                    conf={0.18}
                    tone="illegible"
                  />
                  <div className="rounded-lg border border-dashed border-slateink-200 p-2.5 text-[11px] text-slateink-500">
                    Click a field → jump to crop + OCR candidates
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="product" className="border-b border-slateink-200 bg-canvas-50 py-20">
        <div className="mx-auto max-w-[1400px] px-4 sm:px-6">
          <div className="mb-10 max-w-2xl">
            <h2 className="font-display text-3xl text-slateink-950 sm:text-4xl">
              Prototype in the Playground. Trust every citation.
            </h2>
            <p className="mt-3 text-slateink-600">
              An ADE-style workflow for handwriting: parse lines with visual
              grounding, extract structured fields, and review only what the
              evidence cannot settle.
            </p>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            <Capability
              title="Parse"
              body="Detect lines, preserve reading order, deskew and enhance pages into immutable artifacts."
            />
            <Capability
              title="Extract"
              body="Multi-hypothesis TrOCR with fusion scores. Structured fields carry confidence and uncertainty."
            />
            <Capability
              title="Ground"
              body="Every value links to a source crop, OCR candidates, and an explicit ACCEPTED / REVIEW / ILLEGIBLE / CROSSED_OUT decision."
            />
          </div>
        </div>
      </section>

      <section id="grounding" className="bg-white py-20">
        <div className="mx-auto grid max-w-[1400px] gap-10 px-4 sm:px-6 lg:grid-cols-2 lg:items-center">
          <div>
            <h2 className="font-display text-3xl text-slateink-950 sm:text-4xl">
              Visual grounding by default
            </h2>
            <p className="mt-3 text-slateink-600">
              Like Landing AI ADE, outputs are citation-first. Low-confidence
              spans are highlighted for human review. Crossed-out ink is retained
              but excluded from active text. Nothing is silently invented.
            </p>
            <ul className="mt-6 space-y-3 text-sm text-slateink-700">
              <li className="flex gap-2">
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-brand-600" />
                Bounding boxes on the source page
              </li>
              <li className="flex gap-2">
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-brand-600" />
                Up to 20 OCR hypotheses per line across image variants
              </li>
              <li className="flex gap-2">
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-brand-600" />
                Export Markdown, JSON, and plain text with uncertainty
              </li>
            </ul>
            <a
              href="/playground"
              className="mt-8 inline-flex rounded-lg bg-slateink-900 px-5 py-2.5 text-sm font-semibold text-white hover:bg-slateink-800"
            >
              Upload a handwritten page
            </a>
          </div>
          <div className="surface p-5">
            <div className="mb-3 flex items-center justify-between text-xs text-slateink-500">
              <span>Decision policy</span>
              <span className="font-mono">evidence engine</span>
            </div>
            <div className="space-y-2">
              <PolicyRow color="bg-signal-accept" label="ACCEPTED" desc="Strong consensus, low uncertainty" />
              <PolicyRow color="bg-signal-review" label="REVIEW_REQUIRED" desc="Disagreement or moderate risk" />
              <PolicyRow color="bg-signal-illegible" label="ILLEGIBLE" desc="Refuse to invent text" />
              <PolicyRow color="bg-signal-crossed" label="CROSSED_OUT" desc="Retained, not in active text" />
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-slateink-200 bg-canvas-50 py-8">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 px-4 text-sm text-slateink-500 sm:px-6">
          <span>ScribeProof · HNX26EPS04</span>
          <span>Provenance-preserving handwriting intelligence</span>
        </div>
      </footer>
    </div>
  );
}

function FieldPreview({
  label,
  value,
  conf,
  tone,
}: {
  label: string;
  value: string;
  conf: number;
  tone: "accept" | "review" | "illegible";
}) {
  const color =
    tone === "accept"
      ? "text-signal-accept"
      : tone === "review"
        ? "text-signal-review"
        : "text-signal-illegible";
  return (
    <div className="rounded-lg border border-slateink-100 bg-canvas-50 p-2.5">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-slateink-400">
        <span>{label}</span>
        <span className={color}>{Math.round(conf * 100)}%</span>
      </div>
      <p className="mt-1 truncate text-sm font-medium text-slateink-900">{value}</p>
      <div className="confidence-bar mt-2">
        <div
          className={`h-full rounded-full ${
            tone === "accept"
              ? "bg-signal-accept"
              : tone === "review"
                ? "bg-signal-review"
                : "bg-signal-illegible"
          }`}
          style={{ width: `${conf * 100}%` }}
        />
      </div>
    </div>
  );
}

function Capability({ title, body }: { title: string; body: string }) {
  return (
    <div className="surface p-5">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-brand-700">
        {title}
      </h3>
      <p className="mt-2 text-sm leading-relaxed text-slateink-600">{body}</p>
    </div>
  );
}

function PolicyRow({
  color,
  label,
  desc,
}: {
  color: string;
  label: string;
  desc: string;
}) {
  return (
    <div className="flex items-start gap-3 rounded-lg bg-canvas-50 px-3 py-2.5">
      <span className={`mt-1.5 h-2 w-2 rounded-full ${color}`} />
      <div>
        <p className="text-sm font-semibold text-slateink-900">{label}</p>
        <p className="text-xs text-slateink-500">{desc}</p>
      </div>
    </div>
  );
}
