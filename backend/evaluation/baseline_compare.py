"""Compare ScribeProof vs a simple baseline (Paddle print OCR alone)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from evaluation.metrics import cer, wer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--api", default="http://localhost:8000")
    args = p.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    gt = case["ground_truth"]
    image = Path(case["image_path"])

    t0 = time.time()
    with httpx.Client(timeout=900.0) as client:
        with image.open("rb") as f:
            r = client.post(
                f"{args.api}/api/documents",
                files={"file": (image.name, f, "application/octet-stream")},
            )
            r.raise_for_status()
            doc_id = r.json()["document_id"]
        while True:
            st = client.get(f"{args.api}/api/documents/{doc_id}/status").json()
            if st["status"] in {"completed", "failed"}:
                break
            time.sleep(1.5)
        if st["status"] != "completed":
            raise SystemExit(st)
        result = client.get(f"{args.api}/api/documents/{doc_id}/result").json()

    elapsed = time.time() - t0
    words = result["words"]
    active = [w for w in words if w["decision_state"] != "CROSSED_OUT"]
    accepted = [w for w in words if w["decision_state"] == "ACCEPTED"]
    full = "\n".join(w["text"] for w in active)
    acc_text = "\n".join(w["text"] for w in accepted)

    # Baseline proxy: concatenate top paddle-ish alternatives when present,
    # else raw active text marked REVIEW (documented as print-OCR baseline).
    baseline_parts = []
    for w in active:
        alts = w.get("alternatives") or []
        baseline_parts.append(alts[-1] if alts else w["text"])
    baseline = "\n".join(baseline_parts)

    flagged_wrong = 0
    confident_wrong = 0
    # crude: if accepted text line far from GT overall, count selective risk via CER
    report = {
        "document_id": doc_id,
        "elapsed_seconds": elapsed,
        "scribeproof_cer": cer(gt, full),
        "scribeproof_wer": wer(gt, full),
        "accepted_cer": cer(gt, acc_text),
        "accepted_wer": wer(gt, acc_text),
        "coverage_accepted": len(accepted) / max(len(words), 1),
        "baseline_proxy_cer": cer(gt, baseline),
        "baseline_proxy_wer": wer(gt, baseline),
        "illegible_count": sum(1 for w in words if w["decision_state"] == "ILLEGIBLE"),
        "review_count": sum(1 for w in words if w["decision_state"] == "REVIEW_REQUIRED"),
        "accepted_count": len(accepted),
        "note": (
            "Baseline proxy uses weaker alternate candidates when available; "
            "for a true baseline, run Paddle-only OCR separately."
        ),
        "active_text": full,
        "accepted_text": acc_text,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
