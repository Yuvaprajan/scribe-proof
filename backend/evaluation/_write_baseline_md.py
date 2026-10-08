"""Write baseline_results.md from baseline_results.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / "evaluation" / "reports" / "baseline_results.json"
d = json.loads(p.read_text(encoding="utf-8"))

lines = [
    "# Baseline comparison",
    "",
    f"Mode: `{d.get('mode')}` | cases: {len(d['cases'])}",
    "",
    "| System | CER | WER | ExactMatch | FCE | Review | Coverage |",
    "|--------|-----|-----|------------|-----|--------|----------|",
]
for name, s in d["summary"].items():
    lines.append(
        f"| {name} | {s['mean_cer']:.3f} | {s['mean_wer']:.3f} | "
        f"{s['mean_exact_match']:.3f} | {s['mean_fce_rate']:.3f} | "
        f"{s['mean_review_rate']:.3f} | {s['mean_coverage_accepted']:.3f} |"
    )

lines += [
    "",
    "## Per-case CER / FCE",
    "",
    "| Case | Synthetic | Paddle CER | TrOCR CER | ScribeProof CER | Paddle FCE | ScribeProof FCE |",
    "|------|-----------|------------|-----------|-----------------|------------|-----------------|",
]
for c in d["cases"]:
    s = c["systems"]
    lines.append(
        f"| {c['id']} | {c.get('is_synthetic')} | "
        f"{s['paddle_only']['cer']:.3f} | {s['trocr_lines']['cer']:.3f} | "
        f"{s['scribeproof']['cer']:.3f} | {s['paddle_only']['fce_rate']:.3f} | "
        f"{s['scribeproof']['fce_rate']:.3f} |"
    )

# Hard-only (non-synthetic) subset
hard = [c for c in d["cases"] if not c.get("is_synthetic")]
if hard:
    def mean(sys: str, key: str) -> float:
        vals = [c["systems"][sys][key] for c in hard]
        return sum(vals) / len(vals)

    lines += [
        "",
        f"## Non-synthetic subset (n={len(hard)})",
        "",
        "| System | CER | FCE | Review | Coverage |",
        "|--------|-----|-----|--------|----------|",
        f"| paddle_only | {mean('paddle_only','cer'):.3f} | {mean('paddle_only','fce_rate'):.3f} | "
        f"{mean('paddle_only','review_rate'):.3f} | {mean('paddle_only','coverage_accepted'):.3f} |",
        f"| trocr_lines | {mean('trocr_lines','cer'):.3f} | {mean('trocr_lines','fce_rate'):.3f} | "
        f"{mean('trocr_lines','review_rate'):.3f} | {mean('trocr_lines','coverage_accepted'):.3f} |",
        f"| scribeproof | {mean('scribeproof','cer'):.3f} | {mean('scribeproof','fce_rate'):.3f} | "
        f"{mean('scribeproof','review_rate'):.3f} | {mean('scribeproof','coverage_accepted'):.3f} |",
    ]

lines += [
    "",
    "## Notes",
    "",
    "- Baselines derived from the **same API run** (same detections/crops).",
    "- `paddle_only`: best Paddle hyp per line, always ACCEPTED.",
    "- `trocr_lines`: best TrOCR hyp per line; ACCEPTED if visual_score>=0.72 else REVIEW.",
    "- `scribeproof`: full decision engine.",
    "- Research signal: lower FCE vs paddle_only while keeping similar CER.",
    "- Official hard-HW qualification still needs more non-synthetic GT pages (target 15-30).",
    "",
]

out = ROOT / "evaluation" / "reports" / "baseline_results.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(out.read_text(encoding="utf-8"))
