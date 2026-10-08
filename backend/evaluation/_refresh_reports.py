"""Refresh ablation.md + qualification reports from existing JSON."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.qualification_report import build_report, report_to_markdown

ablation_path = Path("../evaluation/reports/ablation_report.json")
d = json.loads(ablation_path.read_text(encoding="utf-8"))
lines = [
    "# ScribeProof Ablation Report",
    "",
    "**Evidence status:** PRELIMINARY — NOT qualification evidence",
    "",
    "| Config | Kind | CER | WER | FCE | Review | Coverage | Selective risk |",
    "|--------|------|-----|-----|-----|--------|----------|----------------|",
]
for k in "ABCDEF":
    s = d["summary"][k]
    lines.append(
        f"| {k}. {s['label']} | {s['baseline_kind']} | {s['cer']:.3f} | "
        f"{s['wer']:.3f} | {s['fce_rate']:.3f} | {s['review_rate']:.3f} | "
        f"{s['coverage_accepted']:.3f} | {s['selective_risk']:.3f} |"
    )
lines.extend(
    [
        "",
        "## Note",
        "",
        "In this run, B/C fell back to **derived** TrOCR hyps (shared ScribeProof "
        "crops) because independent TrOCR could not load alongside the API "
        "(torch DLL conflict). They must not be called independent baselines.",
        "",
    ]
)
Path("../evaluation/reports/ablation_report.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

report = build_report(
    Path("../evaluation/dataset"),
    Path("../evaluation/reports/baseline_results.json"),
    ablation_path,
    Path("../evaluation/reports/dataset_integrity.json"),
    Path("../evaluation/reports/dataset_integrity.md"),
)
Path("../evaluation/reports/qualification_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
Path("../evaluation/reports/qualification_report.md").write_text(
    report_to_markdown(report), encoding="utf-8"
)
print("decision", report["qualification_decision"])
