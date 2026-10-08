"""CER, WER, selective risk, FCE, and evaluation helpers."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from rapidfuzz.distance import Levenshtein

# Re-export HNX26EPS04 metrics
from evaluation.fce_metrics import (  # noqa: E402
    AggregateEval,
    evaluate_word_list,
    exact_match,
    line_correct,
    risk_coverage_curve,
)


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return Levenshtein.distance(reference, hypothesis) / max(len(reference), 1)


def wer(reference: str, hypothesis: str) -> float:
    ref_w = reference.split()
    hyp_w = hypothesis.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return Levenshtein.distance(ref_w, hyp_w) / max(len(ref_w), 1)


@dataclass
class EvalReport:
    document_id: Optional[str]
    image_path: Optional[str]
    ground_truth: str
    hypothesis_full: str
    hypothesis_accepted: str
    cer: float
    wer: float
    accepted_cer: float
    accepted_wer: float
    coverage_accepted: float
    selective_risk: float
    illegible_precision: Optional[float]
    illegible_recall: Optional[float]
    crossed_out_precision: Optional[float]
    crossed_out_recall: Optional[float]
    processing_time_seconds: Optional[float]
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_texts(
    ground_truth: str,
    full_hypothesis: str,
    accepted_hypothesis: str,
    *,
    total_words: int,
    accepted_words: int,
    document_id: Optional[str] = None,
    image_path: Optional[str] = None,
    processing_time_seconds: Optional[float] = None,
    illegible_precision: Optional[float] = None,
    illegible_recall: Optional[float] = None,
    crossed_out_precision: Optional[float] = None,
    crossed_out_recall: Optional[float] = None,
) -> EvalReport:
    coverage = accepted_words / max(total_words, 1)
    # Selective risk ≈ error rate among accepted outputs
    selective = accepted_cer = cer(ground_truth, accepted_hypothesis)
    return EvalReport(
        document_id=document_id,
        image_path=image_path,
        ground_truth=ground_truth,
        hypothesis_full=full_hypothesis,
        hypothesis_accepted=accepted_hypothesis,
        cer=cer(ground_truth, full_hypothesis),
        wer=wer(ground_truth, full_hypothesis),
        accepted_cer=accepted_cer,
        accepted_wer=wer(ground_truth, accepted_hypothesis),
        coverage_accepted=coverage,
        selective_risk=selective,
        illegible_precision=illegible_precision,
        illegible_recall=illegible_recall,
        crossed_out_precision=crossed_out_precision,
        crossed_out_recall=crossed_out_recall,
        processing_time_seconds=processing_time_seconds,
    )


def run_case(case_path: Path, api_base: str = "http://localhost:8000") -> dict[str, Any]:
    """Upload image, wait for result, score against ground truth."""
    import httpx

    case = json.loads(case_path.read_text(encoding="utf-8"))
    image_path = Path(case["image_path"])
    ground_truth = case["ground_truth"]
    document_id = case.get("document_id")

    t0 = time.time()
    with httpx.Client(timeout=600.0) as client:
        if document_id:
            # Reuse existing
            pass
        else:
            with image_path.open("rb") as f:
                files = {"file": (image_path.name, f, "application/octet-stream")}
                resp = client.post(f"{api_base}/api/documents", files=files)
                resp.raise_for_status()
                document_id = resp.json()["document_id"]

        # Poll
        while True:
            st = client.get(f"{api_base}/api/documents/{document_id}/status").json()
            if st["status"] in {"completed", "failed"}:
                break
            time.sleep(1.5)
        if st["status"] == "failed":
            raise RuntimeError(st.get("error_message") or "processing failed")

        result = client.get(f"{api_base}/api/documents/{document_id}/result").json()

    elapsed = time.time() - t0
    words = result["words"]
    accepted = [w for w in words if w["decision_state"] == "ACCEPTED"]
    full_text = "\n".join(w["text"] for w in words if w["decision_state"] != "CROSSED_OUT")
    accepted_text = "\n".join(w["text"] for w in accepted)

    report = evaluate_texts(
        ground_truth,
        full_text,
        accepted_text,
        total_words=len(words),
        accepted_words=len(accepted),
        document_id=document_id,
        image_path=str(image_path),
        processing_time_seconds=elapsed,
    )
    return report.to_dict()


def ablation_report(results: dict[str, dict[str, Any]]) -> str:
    """Format ablation A–E comparison."""
    lines = [
        "# ScribeProof Ablation Report",
        "",
        "| Config | CER | WER | Accepted CER | Coverage | Selective Risk |",
        "|--------|-----|-----|--------------|----------|----------------|",
    ]
    labels = {
        "A": "TrOCR only",
        "B": "TrOCR + preprocessing",
        "C": "TrOCR + multi-variant ensemble",
        "D": "Full system (evidence decision engine)",
        "E": "Full system + verifier",
    }
    for key in ["A", "B", "C", "D", "E"]:
        if key not in results:
            lines.append(f"| {key}. {labels[key]} | — | — | — | — | — |")
            continue
        r = results[key]
        lines.append(
            f"| {key}. {labels[key]} | {r.get('cer', float('nan')):.3f} | "
            f"{r.get('wer', float('nan')):.3f} | {r.get('accepted_cer', float('nan')):.3f} | "
            f"{r.get('coverage_accepted', float('nan')):.3f} | "
            f"{r.get('selective_risk', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower CER/WER is better.",
            "- Selective risk should decrease as the decision engine refuses unsupported guesses.",
            "- Coverage may drop when the system correctly marks text as ILLEGIBLE.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ScribeProof evaluation")
    parser.add_argument("case", type=Path, help="JSON case file")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run_case(args.case, args.api)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
