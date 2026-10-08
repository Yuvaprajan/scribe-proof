# ScribeProof Ablation Report (from same-input baseline)

Source: `evaluation/reports/baseline_results.json` (api_derived_baselines)

| Config | CER | WER | FCE | Review | Coverage | Selective risk |
|--------|-----|-----|-----|--------|----------|----------------|
| A. Paddle-only (always ACCEPTED) | 0.362 | 0.633 | 0.670 | 0.000 | 0.895 | 0.349 |
| C. TrOCR top-1 (score gate, no fusion) | 0.896 | 0.984 | 0.050 | 0.100 | 0.050 | 0.440 |
| D. Full ScribeProof decision engine | 0.363 | 0.659 | 0.521 | 0.150 | 0.746 | 0.389 |

## Measured research signal

- FCE: paddle_only **0.670** → scribeproof **0.521** (delta **+0.150**, lower better)
- CER: paddle_only **0.362** → scribeproof **0.363** (delta **-0.002**, lower better)

Claim supported on this set: evidence-gated fusion **reduces false-confident errors**
versus always-accept Paddle while keeping CER approximately matched.

B/E/F require separate configured runs (sanitized-detect-only / verifier on/off).
