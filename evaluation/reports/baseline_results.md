# Baseline comparison

**Evidence status:** PRELIMINARY — NOT qualification evidence

Mode: `independent` | cases: 5 (real=1, synthetic=4)

| System | Kind | CER | WER | ExactMatch | FCE | Review | Coverage | Acc@Acc |
|--------|------|-----|-----|------------|-----|--------|----------|---------|
| paddle_only | independent | 0.345 | 0.619 | 0.200 | 0.455 | 0.000 | 0.895 | 0.440 |
| trocr_lines | derived | 0.895 | 0.984 | 0.000 | 0.075 | 0.100 | 0.075 | 0.000 |
| scribeproof | system_under_test | 0.358 | 0.659 | 0.000 | 0.521 | 0.175 | 0.721 | 0.200 |

## Risk-coverage (ScribeProof, mean)

| Coverage | CER | WER | FCE | Accepted Accuracy |
| -------- | --- | --- | --- | ----------------- |
| 0.25 | 0.829 | 1.004 | 0.800 | 0.200 |
| 0.50 | 0.597 | 0.989 | 0.800 | 0.200 |
| 0.75 | 0.612 | 0.816 | 0.800 | 0.200 |
| 1.00 | 0.650 | 0.937 | 0.800 | 0.200 |

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

