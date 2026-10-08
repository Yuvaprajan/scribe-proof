# ScribeProof Ablation Report

**Evidence status:** PRELIMINARY — NOT qualification evidence

| Config | Kind | CER | WER | FCE | Review | Coverage | Selective risk |
|--------|------|-----|-----|-----|--------|----------|----------------|
| A. independent_paddle_only | independent | 0.345 | 0.619 | 0.455 | 0.000 | 0.895 | 0.331 |
| B. independent_opencv_det_plus_trocr | derived | 0.895 | 0.984 | 0.075 | 0.100 | 0.075 | 0.626 |
| C. top1_accept_no_decision_engine | derived | 0.895 | 0.984 | 0.175 | 0.000 | 0.175 | 0.823 |
| D. full_scribeproof_decision_engine | system_under_test | 0.358 | 0.659 | 0.521 | 0.175 | 0.721 | 0.390 |
| E. D_plus_verifier | system_under_test | 0.358 | 0.659 | 0.521 | 0.175 | 0.721 | 0.390 |
| F. E_plus_conservative_thresholds | system_under_test | 0.358 | 0.659 | 0.521 | 0.175 | 0.721 | 0.390 |

## Note

In this run, B/C fell back to **derived** TrOCR hyps (shared ScribeProof crops) because independent TrOCR could not load alongside the API (torch DLL conflict). They must not be called independent baselines.

