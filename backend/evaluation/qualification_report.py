"""Generate qualification report with honest NOT_READY_* decisions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.dataset_integrity import (
    MIN_QUALIFICATION_REAL,
    TARGET_QUALIFICATION_REAL,
    refuse_qualified_status,
    validate_dataset_root,
    write_integrity_reports,
)
from evaluation.research_claims import claims_to_markdown, generate_claims


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def decide_qualification(
    integrity: Any,
    baseline: dict[str, Any],
    ablation: dict[str, Any],
) -> str:
    """Possible decisions — never QUALIFIED_FOR_CHALLENGE without gates."""
    base_decision = refuse_qualified_status(integrity)
    if base_decision == "NOT_READY_DATASET":
        return "NOT_READY_DATASET"

    summary = baseline.get("summary") or {}
    if not summary.get("paddle_only") or not summary.get("scribeproof"):
        return "NOT_READY_EVALUATION"

    # Uncertainty / calibration: refuse challenge if confidence claimed calibrated wrongly
    sp_rows = []
    for case in baseline.get("cases") or []:
        sp = (case.get("systems") or {}).get("scribeproof") or {}
        if sp and not sp.get("error"):
            sp_rows.append(sp)
    if not sp_rows:
        return "NOT_READY_EVALUATION"

    mean_acc = sum(float(r.get("accepted_accuracy") or 0) for r in sp_rows) / len(
        sp_rows
    )
    mean_fce = sum(float(r.get("fce_rate") or 0) for r in sp_rows) / len(sp_rows)
    if mean_acc < 0.5 and mean_fce > 0.2:
        return "NOT_READY_UNCERTAINTY"

    # Accuracy bar intentionally conservative — challenge needs real held-out win
    mean_cer = sum(float(r.get("cer") or 1) for r in sp_rows) / len(sp_rows)
    if mean_cer > 0.25:
        return "NOT_READY_ACCURACY"

    # Internal evaluation only unless challenge dataset + metrics met
    if integrity.qualification_eligible and integrity.qualification_eligible_count >= MIN_QUALIFICATION_REAL:
        # Still not challenge without stronger accuracy + calibration evidence
        return "QUALIFIED_FOR_INTERNAL_EVALUATION"

    return "NOT_READY_DATASET"


def build_report(
    dataset_root: Path,
    baseline_path: Path,
    ablation_path: Path,
    integrity_json: Path,
    integrity_md: Path,
) -> dict[str, Any]:
    integrity = validate_dataset_root(dataset_root.resolve())
    write_integrity_reports(integrity, integrity_json, integrity_md)

    baseline = _load_json(baseline_path)
    ablation = _load_json(ablation_path)
    decision = decide_qualification(integrity, baseline, ablation)

    summary = baseline.get("summary") or {}
    n_cases = len(baseline.get("cases") or [])
    n_real = sum(
        1 for c in (baseline.get("cases") or []) if not c.get("is_synthetic")
    )
    n_synth = sum(
        1 for c in (baseline.get("cases") or []) if c.get("is_synthetic")
    )
    claims = baseline.get("research_claims") or generate_claims(
        summary,
        n_cases=n_cases or integrity.total_cases,
        n_real=n_real or integrity.real_cases,
        n_synthetic=n_synth or integrity.synthetic_cases,
        qualification_eligible=integrity.qualification_eligible,
    )

    # Extreme case analysis
    extreme = []
    for case in baseline.get("cases") or []:
        tags = {t.lower() for t in (case.get("tags") or [])}
        if "extreme" in tags or "example31" in str(case.get("id", "")).lower():
            extreme.append(
                {
                    "id": case.get("id"),
                    "is_synthetic": case.get("is_synthetic"),
                    "systems": {
                        k: {
                            "cer": v.get("cer"),
                            "fce_rate": v.get("fce_rate"),
                            "coverage_accepted": v.get("coverage_accepted"),
                            "accepted_accuracy": v.get("accepted_accuracy"),
                            "abstention_rate": v.get("abstention_rate"),
                            "baseline_kind": v.get("baseline_kind"),
                        }
                        for k, v in (case.get("systems") or {}).items()
                        if isinstance(v, dict) and "cer" in v
                    },
                }
            )

    report = {
        "title": "HNX26EPS04 Qualification Report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "PRELIMINARY — NOT qualification evidence"
        if decision.startswith("NOT_READY")
        else decision,
        "qualification_decision": decision,
        "dataset_integrity": integrity.to_dict(),
        "dataset_composition": {
            "total": integrity.total_cases,
            "real": integrity.real_cases,
            "synthetic": integrity.synthetic_cases,
            "extreme": integrity.extreme_cases,
            "missing_gt": integrity.missing_gt,
            "min_required_real": MIN_QUALIFICATION_REAL,
            "target_real": TARGET_QUALIFICATION_REAL,
            "splits": integrity.splits,
        },
        "baseline_methodology": baseline.get("baseline_methodology")
        or {
            "note": "Run evaluation.run_baseline --independent first",
        },
        "scribeproof_methodology": {
            "pipeline": "Paddle detect + sanitize + TrOCR ensemble + DecisionEngine",
            "confidence_meaning": (
                "model_score / final_score are uncalibrated ranking scores; "
                "calibrated_confidence is null unless a calibrator is fitted"
            ),
        },
        "metrics_summary": summary,
        "risk_coverage": (summary.get("scribeproof") or {}).get("risk_coverage"),
        "ablation": ablation.get("summary") or {},
        "experiment_manifests": ablation.get("experiment_manifests") or {},
        "extreme_case_analysis": extreme,
        "research_claims": claims,
        "reproducibility": {
            "commands": [
                "python -m evaluation.dataset_integrity --root ../evaluation/dataset",
                "python -m evaluation.run_baseline --dataset ../evaluation/dataset/held_out --independent",
                "python -m evaluation.run_ablation --dataset ../evaluation/dataset/held_out",
                "python -m evaluation.qualification_report",
            ],
            "baseline_mode": baseline.get("mode"),
        },
        "security_runtime": {
            "note": "Local/offline preferred; do not send PHI to cloud OCR APIs",
        },
        "failure_cases": [
            c
            for c in (baseline.get("cases") or [])
            if ((c.get("systems") or {}).get("scribeproof") or {}).get("accepted_accuracy", 1)
            == 0
            and ((c.get("systems") or {}).get("scribeproof") or {}).get("accepted", 0)
            > 0
        ],
    }
    return report


def report_to_markdown(report: dict[str, Any]) -> str:
    dc = report["dataset_composition"]
    summary = report.get("metrics_summary") or {}
    lines = [
        "# HNX26EPS04 Qualification Report",
        "",
        f"**Decision:** `{report['qualification_decision']}`",
        "",
        f"**Evidence status:** {report['evidence_status']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## 1. Dataset integrity",
        "",
        f"- Qualification eligible: {report['dataset_integrity'].get('qualification_eligible')}",
        f"- Eligible count: {report['dataset_integrity'].get('qualification_eligible_count')}",
        f"- Failures: {len(report['dataset_integrity'].get('qualification_failures') or [])}",
        "",
        "## 2. Dataset composition",
        "",
        f"- Total: {dc['total']} (real={dc['real']}, synthetic={dc['synthetic']})",
        f"- Extreme: {dc['extreme']}; missing GT: {dc['missing_gt']}",
        f"- Required real for qualification: >={dc['min_required_real']} (target {dc['target_real']})",
        f"- Splits: `{json.dumps(dc['splits'])}`",
        "",
        "## 3. Baseline methodology",
        "",
        "```json",
        json.dumps(report.get("baseline_methodology"), indent=2)[:4000],
        "```",
        "",
        "## 4. ScribeProof methodology",
        "",
        "```json",
        json.dumps(report.get("scribeproof_methodology"), indent=2),
        "```",
        "",
        "## 5–11. Aggregate metrics",
        "",
        "| System | Kind | CER | WER | ExactMatch | FCE | FVE | Coverage | Acc |",
        "|--------|------|-----|-----|------------|-----|-----|----------|-----|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s.get('baseline_kind')} | "
            f"{s.get('mean_cer', float('nan')):.3f} | "
            f"{s.get('mean_wer', float('nan')):.3f} | "
            f"{s.get('mean_exact_match', float('nan')):.3f} | "
            f"{s.get('mean_fce_rate', float('nan')):.3f} | "
            f"{s.get('mean_fve_rate', float('nan')):.3f} | "
            f"{s.get('mean_coverage_accepted', float('nan')):.3f} | "
            f"{s.get('mean_accepted_accuracy', float('nan')):.3f} |"
        )

    rc = report.get("risk_coverage") or []
    lines.extend(
        [
            "",
            "## 10. Risk-coverage (ScribeProof)",
            "",
            "| Coverage | CER | WER | FCE | Accepted Accuracy |",
            "| -------- | --- | --- | --- | ----------------- |",
        ]
    )
    for row in rc:
        lines.append(
            f"| {row['coverage']:.2f} | {row['cer']:.3f} | {row['wer']:.3f} | "
            f"{row['fce']:.3f} | {row['accepted_accuracy']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 12–13. Real / extreme samples",
            "",
            f"Extreme cases analyzed: {len(report.get('extreme_case_analysis') or [])}",
            "",
            "```json",
            json.dumps(report.get("extreme_case_analysis"), indent=2)[:6000],
            "```",
            "",
            "## 14. Ablation results",
            "",
        ]
    )
    abl = report.get("ablation") or {}
    if abl:
        lines.append("| Config | CER | FCE | Coverage |")
        lines.append("|--------|-----|-----|----------|")
        for k, s in abl.items():
            if not isinstance(s, dict):
                continue
            lines.append(
                f"| {k} | {s.get('cer', float('nan')):.3f} | "
                f"{s.get('fce_rate', float('nan')):.3f} | "
                f"{s.get('coverage_accepted', float('nan')):.3f} |"
            )
    else:
        lines.append("_Ablation report not found — run run_ablation._")

    lines.extend(
        [
            "",
            "## 15. Verifier contribution",
            "",
            "See ablation E vs D and pipeline instrumentation fields "
            "`candidate_before_verifier` / `selection_applied`. "
            "If verifier env is off, E equals D.",
            "",
            "## 16. Failure cases",
            "",
            f"Accepted-but-zero-accuracy cases: {len(report.get('failure_cases') or [])}",
            "",
            "## 17. Reproducibility",
            "",
        ]
    )
    for cmd in (report.get("reproducibility") or {}).get("commands") or []:
        lines.append(f"- `{cmd}`")

    lines.extend(
        [
            "",
            "## 18. Security / runtime",
            "",
            str((report.get("security_runtime") or {}).get("note")),
            "",
            "## 19. Qualification decision",
            "",
            f"**`{report['qualification_decision']}`**",
            "",
            "Possible values: NOT_READY_DATASET | NOT_READY_EVALUATION | "
            "NOT_READY_ACCURACY | NOT_READY_UNCERTAINTY | "
            "QUALIFIED_FOR_INTERNAL_EVALUATION | QUALIFIED_FOR_CHALLENGE_EVALUATION",
            "",
            "QUALIFIED_FOR_CHALLENGE_EVALUATION is refused until >=15 real hard "
            "qualification pages exist and accuracy/uncertainty gates pass.",
            "",
        ]
    )
    lines.append(claims_to_markdown(report.get("research_claims") or {}))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root", type=Path, default=Path("evaluation/dataset")
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path("evaluation/reports/baseline_results.json"),
    )
    ap.add_argument(
        "--ablation",
        type=Path,
        default=Path("evaluation/reports/ablation_report.json"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("evaluation/reports/qualification_report.json"),
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=Path("evaluation/reports/qualification_report.md"),
    )
    ap.add_argument(
        "--integrity-json",
        type=Path,
        default=Path("evaluation/reports/dataset_integrity.json"),
    )
    ap.add_argument(
        "--integrity-md",
        type=Path,
        default=Path("evaluation/reports/dataset_integrity.md"),
    )
    args = ap.parse_args()

    report = build_report(
        args.dataset_root,
        args.baseline,
        args.ablation,
        args.integrity_json,
        args.integrity_md,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = report_to_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md.encode("ascii", "replace").decode("ascii"))
    print(f"\nDecision: {report['qualification_decision']}")
    print(f"Wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
