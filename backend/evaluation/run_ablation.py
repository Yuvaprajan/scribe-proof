"""Ablation A–F with machine-readable experiment manifests.

Configs differ ONLY by declared components (see experiment_configs.py):
  A = independent Paddle-only
  B = independent OpenCV lines + TrOCR
  C = B with top-1 force-accept (no decision engine)
  D = full ScribeProof decision engine (API; verifier off preferred)
  E = D + verifier
  F = E + conservative thresholds (same live run unless separately configured)

Usage:
  python -m evaluation.run_ablation --dataset ../evaluation/dataset/held_out \\
      --api http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.experiment_configs import all_experiment_manifests  # noqa: E402
from evaluation.independent_baselines import (  # noqa: E402
    derived_from_api_result,
    independent_paddle_only,
    independent_trocr_only,
)
from evaluation.research_claims import claims_to_markdown, generate_claims  # noqa: E402
from evaluation.run_baseline import (  # noqa: E402
    load_cases,
    resolve_image,
    score_system,
    scribeproof_via_api,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument(
        "--out", type=Path, default=Path("evaluation/reports/ablation_report.md")
    )
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--hard-only", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.dataset, require_gt=True, hard_only=args.hard_only)
    if not cases:
        raise SystemExit(f"No scorable cases in {args.dataset}")

    manifests = all_experiment_manifests()
    buckets: dict[str, list[dict]] = {k: [] for k in "ABCDEF"}

    for case in cases:
        image = resolve_image(args.dataset, case)
        gt_full = case.get("ground_truth") or ""
        gt_lines = case.get("ground_truth_lines") or gt_full.splitlines()

        words_a, meta_a = independent_paddle_only(image)
        buckets["A"].append(
            score_system(
                "A",
                words_a,
                gt_lines,
                gt_full,
                meta_a.get("elapsed_seconds"),
                baseline_kind="independent",
                methodology={**manifests["A"], **meta_a},
            )
        )

        words_b, meta_b = independent_trocr_only(image)
        api_result = None
        api_elapsed = None
        if not args.skip_api:
            try:
                words, api_result, api_elapsed = scribeproof_via_api(image, args.api)
            except Exception as exc:
                words = None
                api_err = str(exc)
            else:
                api_err = None
        else:
            words = None
            api_err = "skip_api"

        # B/C: prefer independent TrOCR; if unavailable, derived Trocr labeled derived
        b_kind = "independent"
        b_meta = {**manifests["B"], **meta_b}
        if meta_b.get("error") or not words_b:
            if api_result is not None:
                _, trocr_w, dmeta = derived_from_api_result(api_result)
                words_b = trocr_w
                b_kind = "derived"
                b_meta = {
                    **manifests["B"],
                    **dmeta,
                    "fallback_reason": "independent_trocr_unavailable",
                    "warning": (
                        "Config B used DERIVED TrOCR hyps (shared ScribeProof crops). "
                        "Not an independent baseline."
                    ),
                }
        buckets["B"].append(
            score_system(
                "B",
                words_b,
                gt_lines,
                gt_full,
                meta_b.get("elapsed_seconds") or api_elapsed,
                baseline_kind=b_kind,
                methodology=b_meta,
            )
        )

        words_c = []
        for w in words_b:
            ww = dict(w)
            if ww.get("hypothesis") and ww["hypothesis"] != "[ILLEGIBLE]":
                ww["decision_state"] = "ACCEPTED"
            words_c.append(ww)
        buckets["C"].append(
            score_system(
                "C",
                words_c,
                gt_lines,
                gt_full,
                meta_b.get("elapsed_seconds") or api_elapsed,
                baseline_kind=b_kind,
                methodology={
                    **manifests["C"],
                    "inherits_baseline_kind_from_B": b_kind,
                },
            )
        )

        if words is not None:
            scored = score_system(
                "D",
                words,
                gt_lines,
                gt_full,
                api_elapsed,
                baseline_kind="system_under_test",
                methodology=manifests["D"],
            )
            buckets["D"].append(scored)
            verifier_on = os.getenv("SCRIBEPROOF_VERIFIER_ENABLED", "false").lower() in {
                "1",
                "true",
                "yes",
            }
            e = {
                **scored,
                "methodology": {
                    **manifests["E"],
                    "verifier_enabled_env": verifier_on,
                },
            }
            f = {
                **scored,
                "methodology": {
                    **manifests["F"],
                    "verifier_enabled_env": verifier_on,
                    "note": (
                        "F shares the live D/E run unless a separate conservative "
                        "threshold deployment is measured; do not over-claim F isolation."
                    ),
                },
            }
            buckets["E"].append(e)
            buckets["F"].append(f)
        else:
            err = {
                "cer": float("nan"),
                "wer": float("nan"),
                "fce_rate": float("nan"),
                "error": api_err or "api_failed",
            }
            buckets["D"].append(err)
            buckets["E"].append(err)
            buckets["F"].append(err)

    def mean(rows, key):
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    labels = {k: manifests[k]["name"] for k in "ABCDEF"}

    lines = [
        "# ScribeProof Ablation Report",
        "",
        "**Evidence status:** PRELIMINARY — NOT qualification evidence",
        "",
        "| Config | Kind | CER | WER | FCE | Review | Coverage | Selective risk |",
        "|--------|------|-----|-----|-----|--------|----------|----------------|",
    ]
    summary = {}
    for key in "ABCDEF":
        rows = buckets[key]
        kinds = [r.get("baseline_kind") for r in rows if r.get("baseline_kind")]
        actual_kind = kinds[0] if kinds else manifests[key].get("baseline_kind")
        if kinds and len(set(kinds)) > 1:
            actual_kind = "mixed:" + ",".join(sorted(set(kinds)))
        summary[key] = {
            "label": labels[key],
            "baseline_kind": actual_kind,
            "declared_baseline_kind": manifests[key].get("baseline_kind"),
            "cer": mean(rows, "cer"),
            "wer": mean(rows, "wer"),
            "fce_rate": mean(rows, "fce_rate"),
            "review_rate": mean(rows, "review_rate"),
            "coverage_accepted": mean(rows, "coverage_accepted"),
            "selective_risk": mean(rows, "selective_risk"),
            "accepted_accuracy": mean(rows, "accepted_accuracy"),
            "n": len(rows),
            "manifest": manifests[key],
        }
        s = summary[key]
        lines.append(
            f"| {key}. {labels[key]} | {s['baseline_kind']} | {s['cer']:.3f} | "
            f"{s['wer']:.3f} | {s['fce_rate']:.3f} | {s['review_rate']:.3f} | "
            f"{s['coverage_accepted']:.3f} | {s['selective_risk']:.3f} |"
        )

    n_real = sum(1 for c in cases if not c.get("is_synthetic"))
    n_synth = sum(1 for c in cases if c.get("is_synthetic"))
    # Map A/B/D into claim generator keys
    claim_summary = {
        "paddle_only": {
            "mean_cer": summary["A"]["cer"],
            "mean_fce_rate": summary["A"]["fce_rate"],
            "mean_coverage_accepted": summary["A"]["coverage_accepted"],
        },
        "trocr_lines": {
            "mean_cer": summary["B"]["cer"],
            "mean_fce_rate": summary["B"]["fce_rate"],
            "mean_coverage_accepted": summary["B"]["coverage_accepted"],
            "mean_abstention_rate": 1.0 - summary["B"]["coverage_accepted"]
            if summary["B"]["coverage_accepted"] == summary["B"]["coverage_accepted"]
            else float("nan"),
        },
        "scribeproof": {
            "mean_cer": summary["D"]["cer"],
            "mean_fce_rate": summary["D"]["fce_rate"],
            "mean_coverage_accepted": summary["D"]["coverage_accepted"],
        },
    }
    claims = generate_claims(
        claim_summary,
        n_cases=len(cases),
        n_real=n_real,
        n_synthetic=n_synth,
        qualification_eligible=False,
    )

    lines.extend(
        [
            "",
            "## Experiment manifests",
            "",
            "Each configuration records detector, recognizer, preprocessing, "
            "candidate generation, reranking, verifier, decision engine, thresholds, "
            "post-processing, correction, and abstention behavior. "
            "A/B/C forbid DecisionEngine/Verifier leakage.",
            "",
            "## Interpretation",
            "",
            "- Lower CER/WER/FCE is better; do not optimize FCE alone.",
            "- Decision engine (D) should reduce FCE vs C (blind accept).",
            "- Coverage may drop when abstaining correctly.",
            "- E/F isolation requires verifier env + conservative threshold deployment; "
            "shared live runs are labeled accordingly.",
            "",
        ]
    )
    lines.append(claims_to_markdown(claims))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    args.out.write_text(text, encoding="utf-8")
    json_path = args.out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "evidence_status": "PRELIMINARY — NOT qualification evidence",
                "summary": summary,
                "buckets": buckets,
                "experiment_manifests": manifests,
                "research_claims": claims,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(text.encode("ascii", "replace").decode("ascii"))
    print(f"\nWrote {args.out} and {json_path}")


if __name__ == "__main__":
    main()
