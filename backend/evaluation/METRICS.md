# Evaluation metrics (authoritative)

See `metrics_core.py` for the implementation.

## Pipeline

```text
raw image
  → dataset loader (manifest.json; split = dev | held_out | qualification)
  → independent baseline OR ScribeProof API
  → line list (hypothesis, decision_state, confidence/model_score)
  → align to ground_truth_lines by reading_order
  → CER / WER / exact match / FCE / FVE / risk-coverage
  → aggregate report + research claims + qualification gate
```

## Definitions

| Metric | Definition |
|--------|------------|
| CER | Levenshtein(chars) / max(\|ref\|, 1) |
| WER | Levenshtein(words) / max(\|ref words\|, 1) |
| Exact match | 1 if whitespace/case-normalized equal |
| FCE | incorrect AND `ACCEPTED` / total lines |
| FVE | incorrect AND visible ACCEPTED/REVIEW text / total lines |
| Review rate | REVIEW_REQUIRED / total |
| Abstention | (ILLEGIBLE + REVIEW) / total |
| Accepted coverage | ACCEPTED / total |
| Selective risk | CER on ACCEPTED lines only (0 if none; interpret with coverage=0) |
| Accepted accuracy | correct ACCEPTED / ACCEPTED |
| Risk-coverage | Rank by confidence; CER/FCE/acc at coverage 0.25/0.5/0.75/1.0 |

## Confidence semantics

| Field | Meaning |
|-------|---------|
| model_score | Uncalibrated visual/ranking score |
| uncertainty_score | Uncalibrated uncertainty |
| calibrated_confidence | Only if a calibrator is fitted (`null` today) |
| decision_status | ACCEPTED / REVIEW_REQUIRED / ILLEGIBLE / CROSSED_OUT |

Do **not** treat display confidence as P(correct) unless `confidence_is_calibrated` is true.

## Baseline kinds

| Kind | Meaning |
|------|---------|
| independent | Raw image → own detector/recognizer; no ScribeProof engine |
| derived | Same detections as ScribeProof API run (ablation only) |
| system_under_test | Full ScribeProof |

Never label a derived baseline as independent.
