"""Produce ablation report skeleton and live D/E scores when API is up."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or backend/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import ablation_report, run_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=Path("ablation_report.md"))
    args = parser.parse_args()

    results: dict = {}
    # Live full-system run maps to D (and E if verifier configured)
    try:
        live = run_case(args.case, args.api)
        results["D"] = live
        # E uses same numbers unless verifier was enabled during processing
        results["E"] = {
            **live,
            "note": "Same run; enable SCRIBEPROOF_VERIFIER_ENABLED=true for true E",
        }
    except Exception as e:
        results["D"] = {"cer": float("nan"), "wer": float("nan"), "accepted_cer": float("nan"),
                        "coverage_accepted": float("nan"), "selective_risk": float("nan"),
                        "error": str(e)}

    # A/B/C filled as placeholders explaining offline measurement path
    for key, note in {
        "A": "Run with SCRIBEPROOF_OCR_MODE=trocr and single raw crop only",
        "B": "TrOCR on deskewed/CLAHE page without ensemble fusion",
        "C": "Multi-variant TrOCR without decision engine (always top-1)",
    }.items():
        results[key] = {
            "cer": float("nan"),
            "wer": float("nan"),
            "accepted_cer": float("nan"),
            "coverage_accepted": float("nan"),
            "selective_risk": float("nan"),
            "note": note,
        }

    md = ablation_report(results)
    md += "\n## Live full-system metrics (D)\n\n```json\n"
    md += json.dumps(results.get("D", {}), indent=2)
    md += "\n```\n"
    args.out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
