# Baseline comparison

Mode: `api_derived_baselines` | cases: 5

| System | CER | WER | ExactMatch | FCE | Review | Coverage |
|--------|-----|-----|------------|-----|--------|----------|
| paddle_only | 0.362 | 0.633 | 0.000 | 0.670 | 0.000 | 0.895 |
| trocr_lines | 0.896 | 0.984 | 0.000 | 0.050 | 0.100 | 0.050 |
| scribeproof | 0.363 | 0.659 | 0.000 | 0.521 | 0.150 | 0.746 |

## Per-case CER / FCE

| Case | Synthetic | Paddle CER | TrOCR CER | ScribeProof CER | Paddle FCE | ScribeProof FCE |
|------|-----------|------------|-----------|-----------------|------------|-----------------|
| messy_rx_sample | True | 0.059 | 0.970 | 0.059 | 0.000 | 0.000 |
| messy_note | True | 0.466 | 0.903 | 0.466 | 1.000 | 0.833 |
| placeholder_messy_note | True | 0.408 | 0.835 | 0.388 | 1.000 | 0.800 |
| printed_report_sample | True | 0.075 | 0.950 | 0.075 | 0.875 | 0.875 |
| script_errors_Example31 | False | 0.800 | 0.824 | 0.828 | 0.476 | 0.095 |

## Non-synthetic subset (n=1)

| System | CER | FCE | Review | Coverage |
|--------|-----|-----|--------|----------|
| paddle_only | 0.800 | 0.476 | 0.000 | 0.476 |
| trocr_lines | 0.824 | 0.048 | 0.333 | 0.048 |
| scribeproof | 0.828 | 0.095 | 0.381 | 0.095 |

## Notes

- Baselines derived from the **same API run** (same detections/crops).
- `paddle_only`: best Paddle hyp per line, always ACCEPTED.
- `trocr_lines`: best TrOCR hyp per line; ACCEPTED if visual_score>=0.72 else REVIEW.
- `scribeproof`: full decision engine.
- Research signal: lower FCE vs paddle_only while keeping similar CER.
- Official hard-HW qualification still needs more non-synthetic GT pages (target 15-30).
