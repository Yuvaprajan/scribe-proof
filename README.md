# ScribeProof

Provenance-preserving handwriting intelligence for **HackNex HNX26EPS04: Extreme Bad-Handwriting Digitizing Stack**.

Every output word keeps visual evidence, alternative OCR hypotheses, calibrated confidence, and an explicit decision: `ACCEPTED`, `REVIEW_REQUIRED`, `ILLEGIBLE`, or `CROSSED_OUT`.

## Product rules

1. Never silently invent text.
2. Never let an LLM freely rewrite OCR output.
3. Every final word links to its source crop and OCR evidence.
4. Insufficient evidence → `[ILLEGIBLE]`, not a confident guess.
5. Crossed-out text is retained but excluded from active final text.
6. UI shows original image, overlays, editable transcription, confidence, candidates, and evidence.

## Architecture

Modular monolith:

```text
backend/
  api/                 FastAPI routes + schemas
  domain/              entities + decision policy
  application/         document orchestration
  infrastructure/
    storage/           SHA-256 artifact store + SQLite repo
    models/            detection, TrOCR, reranker, verifier
    imaging/           preprocessing + quality scores
  workers/             BackgroundTasks + durable jobs table
  evaluation/          CER/WER + ablation helpers

frontend/
  app/                 Next.js 15 App Router
  components/          viewer, editor, evidence, job status
```

FastAPI is the source of truth. Next.js talks to it via `NEXT_PUBLIC_API_BASE_URL`.

## Quick start (Docker)

```bash
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:3000
- API health: http://localhost:8000/health
- OpenAPI: http://localhost:8000/docs

First TrOCR download can take several minutes. Until weights are available, the stack falls back to a **non-inventing mock recognizer** (`SCRIBEPROOF_ALLOW_MOCK_FALLBACK=true`) that refuses confident English guesses and surfaces `[ILLEGIBLE]` / review states from evidence rules.

### Force mock OCR (fast demo)

```bash
SCRIBEPROOF_OCR_MODE=mock docker compose up --build
```

### Enable optional VLM verifier

```bash
SCRIBEPROOF_VERIFIER_ENABLED=true
SCRIBEPROOF_VERIFIER_API_KEY=sk-...
```

Verifier may only pick an existing candidate or `ILLEGIBLE`. It never invents text. Core pipeline works with verifier disabled.

## Local development (without Docker)

### Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# optional: pip install -r requirements-optional.txt  # PaddleOCR
set SCRIBEPROOF_OCR_MODE=mock
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
set NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

## Demo flow (ADE-style Playground)

1. Open http://localhost:3000 → **Try the Playground** (`/playground`).
2. Upload PNG/JPG/JPEG/PDF (or generate `samples/messy_note.png`).
3. Watch agentic job stages: loading → preprocessing → detection → recognition.
4. On `/documents/[documentId]` use **Parse / Extract / Review** tools:
   - **Document** pane: visual grounding, preprocess variant toggles, click boxes.
   - **Parse**: layout-aware Markdown with decision annotations.
   - **Extract**: field cards with confidence bars (Landing AI ADE–style).
   - **Review**: editable transcription; double-click to correct (immutable audit).
   - **Citations**: crop, OCR candidates, score breakdown.
5. Toggle **Low-confidence only** to highlight review/illegible spans.
6. Export JSON / Markdown / TXT with uncertainty and crossed-out preserved.

Generate a sample page:

```bash
pip install opencv-python-headless numpy
python scripts/generate_sample.py
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| POST | `/api/documents` | Upload → document ID + background job |
| GET | `/api/documents/{id}` | Metadata |
| GET | `/api/documents/{id}/status` | Job progress |
| GET | `/api/documents/{id}/result` | Full evidence graph + words |
| PATCH | `/api/words/{word_id}` | User correction |
| GET | `/api/artifacts/{id}` | Content-addressed blob |
| GET | `/api/documents/{id}/export/{txt\|md\|json}` | Exports |

## Models (real OCR path)

Default runtime uses **both**:

1. **PaddleOCR** — text-line detection (+ recognition hypotheses as evidence)
2. **TrOCR handwritten** — primary handwriting recognizer with beam search

```bash
# Install (Windows: torch first, then paddle)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install paddlepaddle==3.0.0
pip install paddleocr==2.9.1 --no-deps
pip install -r backend/requirements.txt

# Download / warm weights (TrOCR is large; first run can take several minutes)
set SCRIBEPROOF_TROCR_MODEL=microsoft/trocr-large-handwritten
python scripts/download_models.py

# Run API with real OCR (no mock)
set SCRIBEPROOF_OCR_MODE=trocr
set SCRIBEPROOF_USE_PADDLE=true
set SCRIBEPROOF_ALLOW_MOCK_FALLBACK=false
```

Check readiness: `GET http://localhost:8000/api/models/status`

- **TrOCR config:** `num_beams=8`, `num_return_sequences=5`, `output_scores=True`
- **Default model:** `microsoft/trocr-large-handwritten` (override with `SCRIBEPROOF_TROCR_MODEL`; use `trocr-base-handwritten` on CPU for speed)
- **Line variants:** raw / CLAHE / adaptive-binarized / denoised → up to 20 TrOCR hypotheses + Paddle candidates
- **Decision engine:** deterministic fusion (no LLM rewrite)
- If TrOCR weights are still downloading, Paddle recognition continues so the playground stays functional

## Evaluation

```bash
# After API is up and sample exists:
python backend/evaluation/metrics.py samples/eval_case.json --api http://localhost:8000 --out eval_out.json
python backend/evaluation/run_ablation.py --case samples/eval_case.json --api http://localhost:8000 --out ablation_report.md
```

Reports CER, WER, accepted-only CER/WER, coverage, selective risk, and an A–E ablation table.

## Limitations (hackathon scope)

- SQLite by default (repository designed for PostgreSQL swap via `SCRIBEPROOF_DATABASE_URL`).
- PaddleOCR / poppler / GPU are optional; OpenCV + mock/TrOCR keep the vertical slice alive.
- TrOCR large model is heavy on CPU; expect slow first page.
- Printed-vs-handwriting / table cell parsing are heuristic, not a full layout LM.
- Ablation A–C require separate configured runs; D is measured live against the API.

## License

Hackathon demo code for HNX26EPS04.
