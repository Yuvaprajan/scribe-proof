"""Baseline comparison with explicit independent vs derived labeling.

Usage:
  python -m evaluation.run_baseline --dataset ../evaluation/dataset/held_out \\
      --api http://localhost:8000 --independent

Independent baselines (default when --independent):
  raw image → Paddle / OpenCV+TrOCR → baseline decode

Derived baselines (--derived-only):
  same ScribeProof detections — labeled derived, NEVER independent.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from evaluation.fce_metrics import evaluate_word_list
from evaluation.independent_baselines import (
    derived_from_api_result,
    independent_paddle_only,
    independent_trocr_only,
)
from evaluation.research_claims import claims_to_markdown, generate_claims

logger = logging.getLogger("baseline")
logging.basicConfig(level=logging.INFO)


def load_cases(
    dataset_dir: Path,
    *,
    require_gt: bool = True,
    hard_only: bool = False,
    allow_synthetic: bool = True,
) -> list[dict[str, Any]]:
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        cases = data.get("cases") or data
    else:
        cases = []
        for meta in sorted(dataset_dir.glob("**/case.json")):
            cases.append(json.loads(meta.read_text(encoding="utf-8")))
    out = []
    for c in cases:
        if require_gt and not (c.get("ground_truth") or "").strip():
            continue
        if hard_only and c.get("is_synthetic"):
            continue
        if hard_only and c.get("needs_gt_review"):
            continue
        if not allow_synthetic and c.get("is_synthetic"):
            continue
        out.append(c)
    return out


def resolve_image(dataset_dir: Path, case: dict[str, Any]) -> Path:
    raw = case.get("image") or case.get("image_path")
    p = Path(raw)
    if p.is_file():
        return p
    cand = dataset_dir / raw
    if cand.is_file():
        return cand
    cand = dataset_dir / "images" / Path(raw).name
    if cand.is_file():
        return cand
    raise FileNotFoundError(raw)


def scribeproof_via_api(
    image_path: Path, api: str
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    t0 = time.time()
    with httpx.Client(timeout=900.0) as client:
        with image_path.open("rb") as f:
            r = client.post(
                f"{api}/api/documents",
                files={"file": (image_path.name, f, "application/octet-stream")},
            )
            r.raise_for_status()
            doc_id = r.json()["document_id"]
        while True:
            st = client.get(f"{api}/api/documents/{doc_id}/status").json()
            if st["status"] in {"completed", "failed"}:
                break
            time.sleep(1.5)
        if st["status"] != "completed":
            raise RuntimeError(st.get("error_message") or "failed")
        result = client.get(f"{api}/api/documents/{doc_id}/result").json()
    elapsed = time.time() - t0
    words = []
    for w in result.get("words") or []:
        sb = w.get("score_breakdown") or {}
        words.append(
            {
                "reference": "",
                "hypothesis": w.get("text") or "",
                "decision_state": w.get("decision_state") or "REVIEW_REQUIRED",
                "confidence": float(w.get("confidence") or 0.0),
                "reading_order": int(w.get("reading_order") or 0),
                "model_score": float(sb.get("model_score") or w.get("confidence") or 0.0),
                "uncertainty_score": float(
                    sb.get("uncertainty_score") or sb.get("uncertainty") or 0.0
                ),
                "calibrated_confidence": sb.get("calibrated_confidence"),
                "confidence_is_calibrated": bool(sb.get("confidence_is_calibrated")),
            }
        )
    return words, result, elapsed


# Back-compat aliases used by older ablation scripts
def paddle_only_transcribe(image_path: Path) -> list[dict[str, Any]]:
    words, _ = independent_paddle_only(image_path)
    return words


def trocr_line_baseline(image_path: Path) -> list[dict[str, Any]]:
    words, _ = independent_trocr_only(image_path)
    return words


def baselines_from_api_result(result: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    p, t, _ = derived_from_api_result(result)
    return p, t


def align_refs(gt_lines: list[str], words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(words, key=lambda w: w.get("reading_order", 0))
    out = []
    for i, w in enumerate(ordered):
        ref = gt_lines[i] if i < len(gt_lines) else ""
        ww = dict(w)
        ww["reference"] = ref
        out.append(ww)
    for j in range(len(ordered), len(gt_lines)):
        out.append(
            {
                "reference": gt_lines[j],
                "hypothesis": "",
                "decision_state": "ILLEGIBLE",
                "confidence": 0.0,
                "reading_order": j,
            }
        )
    return out


def score_system(
    name: str,
    words: list[dict[str, Any]],
    gt_lines: list[str],
    gt_full: str,
    elapsed: Optional[float] = None,
    *,
    baseline_kind: Optional[str] = None,
    methodology: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    aligned = align_refs(gt_lines, words)
    active = [
        w for w in aligned if str(w.get("decision_state")).upper() != "CROSSED_OUT"
    ]
    hyp_full = "\n".join(w["hypothesis"] for w in active)
    agg = evaluate_word_list(aligned, full_reference=gt_full, full_hypothesis=hyp_full)
    d = agg.to_dict()
    d.pop("lines", None)
    d["system"] = name
    d["elapsed_seconds"] = elapsed
    d["hypothesis_text"] = hyp_full
    if baseline_kind:
        d["baseline_kind"] = baseline_kind
    if methodology:
        d["methodology"] = methodology
    return d


def _mean_key(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _aggregate_risk_coverage(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Mean risk-coverage table across cases."""
    points = (0.25, 0.5, 0.75, 1.0)
    tables = [r.get("risk_coverage") or [] for r in rows if r.get("risk_coverage")]
    if not tables:
        return []
    out = []
    for i, p in enumerate(points):
        cells = [t[i] for t in tables if len(t) > i]
        if not cells:
            continue
        out.append(
            {
                "coverage": float(p),
                "cer": sum(c["cer"] for c in cells) / len(cells),
                "wer": sum(c["wer"] for c in cells) / len(cells),
                "fce": sum(c["fce"] for c in cells) / len(cells),
                "accepted_accuracy": sum(c["accepted_accuracy"] for c in cells)
                / len(cells),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument(
        "--out", type=Path, default=Path("evaluation/reports/baseline_results.json")
    )
    ap.add_argument("--skip-api", action="store_true")
    ap.add_argument("--skip-paddle", action="store_true")
    ap.add_argument("--skip-trocr", action="store_true")
    ap.add_argument(
        "--independent",
        action="store_true",
        default=True,
        help="Run true independent Paddle/TrOCR baselines (default)",
    )
    ap.add_argument(
        "--no-independent",
        action="store_true",
        help="Disable independent baselines",
    )
    ap.add_argument(
        "--derived-only",
        action="store_true",
        help="Also/only attach derived (shared-detector) baselines from API",
    )
    ap.add_argument(
        "--local-models",
        action="store_true",
        help="Deprecated alias for --independent",
    )
    ap.add_argument("--hard-only", action="store_true")
    ap.add_argument("--include-empty-gt", action="store_true")
    args = ap.parse_args()

    use_independent = (args.independent or args.local_models) and not args.no_independent
    if args.derived_only and args.no_independent:
        use_independent = False

    cases = load_cases(
        args.dataset,
        require_gt=not args.include_empty_gt,
        hard_only=args.hard_only,
    )
    if not cases:
        raise SystemExit(f"No scorable cases in {args.dataset}")

    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "mode": "independent" if use_independent else "derived_baselines",
        "evidence_status": "PRELIMINARY — NOT qualification evidence",
        "baseline_methodology": {
            "independent_paddle": (
                "raw image → Paddle det+rec → ACCEPTED; no ScribeProof components"
            ),
            "independent_trocr": (
                "raw image → OpenCV morphology lines → TrOCR → threshold decode; "
                "shared preprocessing: grayscale+Otsu for detection only"
            ),
            "derived": (
                "same ScribeProof detections/crops — labeled derived; "
                "NOT an independent external baseline"
            ),
        },
        "cases": [],
    }
    totals: dict[str, list[dict[str, Any]]] = {
        "paddle_only": [],
        "trocr_lines": [],
        "scribeproof": [],
        "paddle_derived": [],
        "trocr_derived": [],
    }

    for case in cases:
        image = resolve_image(args.dataset, case)
        gt_full = case.get("ground_truth") or ""
        gt_lines = case.get("ground_truth_lines") or gt_full.splitlines()
        tags = case.get("difficulty_tags") or case.get("tags") or []
        entry: dict[str, Any] = {
            "id": case.get("id") or image.stem,
            "image": str(image),
            "tags": tags,
            "is_synthetic": bool(case.get("is_synthetic")),
            "systems": {},
        }

        api_result = None
        api_elapsed = None
        if not args.skip_api:
            try:
                words, api_result, api_elapsed = scribeproof_via_api(image, args.api)
                entry["systems"]["scribeproof"] = score_system(
                    "scribeproof",
                    words,
                    gt_lines,
                    gt_full,
                    api_elapsed,
                    baseline_kind="system_under_test",
                    methodology={
                        "pipeline": "ScribeProof modular monolith",
                        "confidence_meaning": (
                            "uncalibrated evidence/ranking score — not P(correct)"
                        ),
                    },
                )
                totals["scribeproof"].append(entry["systems"]["scribeproof"])
            except Exception as e:
                entry["systems"]["scribeproof"] = {"error": str(e)}
                logger.exception("API failed for %s", entry["id"])

        independent_trocr_ok = False
        if use_independent:
            if not args.skip_paddle:
                words, meta = independent_paddle_only(image)
                entry["systems"]["paddle_only"] = score_system(
                    "paddle_only",
                    words,
                    gt_lines,
                    gt_full,
                    meta.get("elapsed_seconds"),
                    baseline_kind="independent",
                    methodology=meta,
                )
                totals["paddle_only"].append(entry["systems"]["paddle_only"])
            if not args.skip_trocr:
                words, meta = independent_trocr_only(image)
                if meta.get("error") or not words:
                    entry["systems"]["trocr_independent_failed"] = {
                        "error": meta.get("error") or "empty_output",
                        "baseline_kind": "independent",
                        "methodology": meta,
                        "note": (
                            "Independent TrOCR unavailable in this process; "
                            "see trocr_lines only if derived fallback was enabled."
                        ),
                    }
                else:
                    independent_trocr_ok = True
                    entry["systems"]["trocr_lines"] = score_system(
                        "trocr_lines",
                        words,
                        gt_lines,
                        gt_full,
                        meta.get("elapsed_seconds"),
                        baseline_kind="independent",
                        methodology=meta,
                    )
                    totals["trocr_lines"].append(entry["systems"]["trocr_lines"])

        # Derived baselines: explicit --derived-only, or fallback when independent TrOCR fails
        need_derived = (
            args.derived_only
            or not use_independent
            or (use_independent and not independent_trocr_ok and not args.skip_trocr)
        )
        if need_derived and api_result is not None:
            paddle_w, trocr_w, dmeta = derived_from_api_result(api_result)
            if not args.skip_paddle and (
                args.derived_only or not use_independent
            ):
                key_p = "paddle_derived" if use_independent else "paddle_only"
                entry["systems"][key_p] = score_system(
                    key_p,
                    paddle_w,
                    gt_lines,
                    gt_full,
                    api_elapsed,
                    baseline_kind="derived",
                    methodology=dmeta,
                )
                entry["systems"][key_p]["note"] = dmeta.get("warning")
                totals[key_p].append(entry["systems"][key_p])
            if not args.skip_trocr and not independent_trocr_ok:
                # Never call this independent — shared ScribeProof detections
                key_t = "trocr_lines"
                entry["systems"][key_t] = score_system(
                    key_t,
                    trocr_w,
                    gt_lines,
                    gt_full,
                    api_elapsed,
                    baseline_kind="derived",
                    methodology={
                        **dmeta,
                        "fallback_reason": "independent_trocr_unavailable",
                        "shared_preprocessing": [
                            "scribeproof_line_crops_from_same_api_run"
                        ],
                    },
                )
                entry["systems"][key_t]["note"] = (
                    "DERIVED baseline (shared detector/crops). NOT independent. "
                    + str(dmeta.get("warning") or "")
                )
                totals[key_t].append(entry["systems"][key_t])
            elif not args.skip_trocr and args.derived_only and independent_trocr_ok:
                entry["systems"]["trocr_derived"] = score_system(
                    "trocr_derived",
                    trocr_w,
                    gt_lines,
                    gt_full,
                    api_elapsed,
                    baseline_kind="derived",
                    methodology=dmeta,
                )
                entry["systems"]["trocr_derived"]["note"] = dmeta.get("warning")
                totals["trocr_derived"].append(entry["systems"]["trocr_derived"])

        report["cases"].append(entry)
        logger.info("Scored case %s", entry["id"])

    summary = {}
    for sys_name, rows in totals.items():
        if not rows:
            continue
        summary[sys_name] = {
            "n": len(rows),
            "baseline_kind": rows[0].get("baseline_kind"),
            "mean_cer": _mean_key(rows, "cer"),
            "mean_wer": _mean_key(rows, "wer"),
            "mean_exact_match": _mean_key(rows, "exact_match"),
            "mean_fce_rate": _mean_key(rows, "fce_rate"),
            "mean_fve_rate": _mean_key(rows, "fve_rate"),
            "mean_review_rate": _mean_key(rows, "review_rate"),
            "mean_abstention_rate": _mean_key(rows, "abstention_rate"),
            "mean_coverage_accepted": _mean_key(rows, "coverage_accepted"),
            "mean_selective_risk": _mean_key(rows, "selective_risk"),
            "mean_accepted_accuracy": _mean_key(rows, "accepted_accuracy"),
            "risk_coverage": _aggregate_risk_coverage(rows),
        }
    report["summary"] = summary

    n_real = sum(1 for c in report["cases"] if not c.get("is_synthetic"))
    n_synth = sum(1 for c in report["cases"] if c.get("is_synthetic"))
    claims = generate_claims(
        summary,
        n_cases=len(report["cases"]),
        n_real=n_real,
        n_synthetic=n_synth,
        qualification_eligible=False,
    )
    report["research_claims"] = claims

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Baseline comparison",
        "",
        f"**Evidence status:** {report['evidence_status']}",
        "",
        f"Mode: `{report.get('mode')}` | cases: {len(report['cases'])} "
        f"(real={n_real}, synthetic={n_synth})",
        "",
        "| System | Kind | CER | WER | ExactMatch | FCE | Review | Coverage | Acc@Acc |",
        "|--------|------|-----|-----|------------|-----|--------|----------|---------|",
    ]
    for sys_name, s in summary.items():
        md.append(
            f"| {sys_name} | {s.get('baseline_kind')} | {s['mean_cer']:.3f} | "
            f"{s['mean_wer']:.3f} | {s['mean_exact_match']:.3f} | "
            f"{s['mean_fce_rate']:.3f} | {s['mean_review_rate']:.3f} | "
            f"{s['mean_coverage_accepted']:.3f} | "
            f"{s['mean_accepted_accuracy']:.3f} |"
        )

    # Risk-coverage for ScribeProof
    rc = (summary.get("scribeproof") or {}).get("risk_coverage") or []
    if rc:
        md.extend(
            [
                "",
                "## Risk-coverage (ScribeProof, mean)",
                "",
                "| Coverage | CER | WER | FCE | Accepted Accuracy |",
                "| -------- | --- | --- | --- | ----------------- |",
            ]
        )
        for row in rc:
            md.append(
                f"| {row['coverage']:.2f} | {row['cer']:.3f} | {row['wer']:.3f} | "
                f"{row['fce']:.3f} | {row['accepted_accuracy']:.3f} |"
            )

    md.append("")
    md.append(claims_to_markdown(claims))

    md_path = args.out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode("ascii"))
    print(f"\nWrote {args.out} and {md_path}")


if __name__ == "__main__":
    main()
