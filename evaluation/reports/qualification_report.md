# HNX26EPS04 Qualification Report

**Decision:** `NOT_READY_DATASET`

**Evidence status:** PRELIMINARY — NOT qualification evidence

Generated: 2026-10-08T20:22:41.097438+00:00

## 1. Dataset integrity

- Qualification eligible: False
- Eligible count: 0
- Failures: 1

## 2. Dataset composition

- Total: 14 (real=6, synthetic=8)
- Extreme: 3; missing GT: 5
- Required real for qualification: >=15 (target 30)
- Splits: `{"dev": {"exists": true, "n": 4, "real": 0, "synthetic": 4, "with_gt": 4}, "held_out": {"exists": true, "n": 10, "real": 6, "synthetic": 4, "with_gt": 5}, "qualification": {"exists": true, "n": 0, "real": 0, "synthetic": 0, "with_gt": 0}}`

## 3. Baseline methodology

```json
{
  "independent_paddle": "raw image \u2192 Paddle det+rec \u2192 ACCEPTED; no ScribeProof components",
  "independent_trocr": "raw image \u2192 OpenCV morphology lines \u2192 TrOCR \u2192 threshold decode; shared preprocessing: grayscale+Otsu for detection only",
  "derived": "same ScribeProof detections/crops \u2014 labeled derived; NOT an independent external baseline"
}
```

## 4. ScribeProof methodology

```json
{
  "pipeline": "Paddle detect + sanitize + TrOCR ensemble + DecisionEngine",
  "confidence_meaning": "model_score / final_score are uncalibrated ranking scores; calibrated_confidence is null unless a calibrator is fitted"
}
```

## 5–11. Aggregate metrics

| System | Kind | CER | WER | ExactMatch | FCE | FVE | Coverage | Acc |
|--------|------|-----|-----|------------|-----|-----|----------|-----|
| paddle_only | independent | 0.345 | 0.619 | 0.200 | 0.455 | 0.455 | 0.895 | 0.440 |
| trocr_lines | derived | 0.895 | 0.984 | 0.000 | 0.075 | 0.175 | 0.075 | 0.000 |
| scribeproof | system_under_test | 0.358 | 0.659 | 0.000 | 0.521 | 0.695 | 0.721 | 0.200 |

## 10. Risk-coverage (ScribeProof)

| Coverage | CER | WER | FCE | Accepted Accuracy |
| -------- | --- | --- | --- | ----------------- |
| 0.25 | 0.829 | 1.004 | 0.800 | 0.200 |
| 0.50 | 0.597 | 0.989 | 0.800 | 0.200 |
| 0.75 | 0.612 | 0.816 | 0.800 | 0.200 |
| 1.00 | 0.650 | 0.937 | 0.800 | 0.200 |

## 12–13. Real / extreme samples

Extreme cases analyzed: 1

```json
[
  {
    "id": "script_errors_Example31",
    "is_synthetic": false,
    "systems": {
      "scribeproof": {
        "cer": 0.828,
        "fce_rate": 0.09523809523809523,
        "coverage_accepted": 0.09523809523809523,
        "accepted_accuracy": 0.0,
        "abstention_rate": 0.9047619047619048,
        "baseline_kind": "system_under_test"
      },
      "paddle_only": {
        "cer": 0.812,
        "fce_rate": 0.47619047619047616,
        "coverage_accepted": 0.47619047619047616,
        "accepted_accuracy": 0.0,
        "abstention_rate": 0.5238095238095238,
        "baseline_kind": "independent"
      },
      "trocr_lines": {
        "cer": 0.824,
        "fce_rate": 0.047619047619047616,
        "coverage_accepted": 0.047619047619047616,
        "accepted_accuracy": 0.0,
        "abstention_rate": 0.9523809523809523,
        "baseline_kind": "derived"
      }
    }
  }
]
```

## 14. Ablation results

| Config | CER | FCE | Coverage |
|--------|-----|-----|----------|
| A | 0.345 | 0.455 | 0.895 |
| B | 0.895 | 0.075 | 0.075 |
| C | 0.895 | 0.175 | 0.175 |
| D | 0.358 | 0.521 | 0.721 |
| E | 0.358 | 0.521 | 0.721 |
| F | 0.358 | 0.521 | 0.721 |

## 15. Verifier contribution

See ablation E vs D and pipeline instrumentation fields `candidate_before_verifier` / `selection_applied`. If verifier env is off, E equals D.

## 16. Failure cases

Accepted-but-zero-accuracy cases: 4

## 17. Reproducibility

- `python -m evaluation.dataset_integrity --root ../evaluation/dataset`
- `python -m evaluation.run_baseline --dataset ../evaluation/dataset/held_out --independent`
- `python -m evaluation.run_ablation --dataset ../evaluation/dataset/held_out`
- `python -m evaluation.qualification_report`

## 18. Security / runtime

Local/offline preferred; do not send PHI to cloud OCR APIs

## 19. Qualification decision

**`NOT_READY_DATASET`**

Possible values: NOT_READY_DATASET | NOT_READY_EVALUATION | NOT_READY_ACCURACY | NOT_READY_UNCERTAINTY | QUALIFIED_FOR_INTERNAL_EVALUATION | QUALIFIED_FOR_CHALLENGE_EVALUATION

QUALIFIED_FOR_CHALLENGE_EVALUATION is refused until >=15 real hard qualification pages exist and accuracy/uncertainty gates pass.

## Research interpretation (auto-generated)

**Status:** PRELIMINARY — NOT qualification evidence

- Cases: 5 (real=1, synthetic=4)
- Qualification eligible: False

| Level | Claim |
|-------|-------|
| `NOT_PROVEN` | HNX26EPS04 qualification / challenge-ready performance is NOT PROVEN: n=5 cases, real=1, synthetic=4; qualification dataset gate not satisfied. |
| `SUPPORTED_BUT_PRELIMINARY` | ScribeProof CER (0.3584) is essentially equal to independent Paddle-only CER (0.3451) on this set (delta=0.0133). ScribeProof is not shown to outperform Paddle on CER. |
| `SUPPORTED_BUT_PRELIMINARY` | FCE reduction vs Paddle-only is not demonstrated (ScribeProof FCE=0.5207, Paddle FCE=0.4552). |
| `SUPPORTED_BUT_PRELIMINARY` | ScribeProof has lower accepted coverage than Paddle-only (expected under selective abstention). |
| `SUPPORTED_BUT_PRELIMINARY` | TrOCR-only achieves low FCE primarily through heavy abstention / low coverage (FCE=0.0745, coverage=0.0745). Do not treat low FCE alone as superiority. |
| `NOT_PROVEN` | Only 1 non-synthetic case(s) with scorable GT; evidence is insufficient for qualification. |
| `NOT_PROVEN` | Statistical significance of system differences is NOT PROVEN on n<15. |
