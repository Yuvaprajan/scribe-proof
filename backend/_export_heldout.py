"""Export processed documents' raw uploads into evaluation/dataset/held_out."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = Path(__file__).resolve().parent / "data" / "db" / "scribeproof.db"
OUT = ROOT / "evaluation" / "dataset" / "held_out"
IMG = OUT / "images"
IMG.mkdir(parents=True, exist_ok=True)


def guess_tags(filename: str, words: list[dict]) -> list[str]:
    tags = set()
    name = filename.lower()
    if "whatsapp" in name:
        tags.add("phone_capture")
    if "script" in name or "example" in name:
        tags.add("annotations")
        tags.add("form")
    states = {w.get("decision_state") for w in words}
    regions = {w.get("region_type") for w in words}
    if "CROSSED_OUT" in states:
        tags.add("crossed_out")
    if "margin_note" in regions:
        tags.add("margin_notes")
    if "printed_text" in regions and "main_handwriting" in regions:
        tags.add("mixed_print_hw")
    if "main_handwriting" in regions:
        tags.add("extreme")
        tags.add("cursive")
    if "table" in regions:
        tags.add("form")
    if not tags:
        tags.add("medical")
    return sorted(tags)


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"DB missing: {DB}")
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    docs = con.execute(
        "SELECT id, filename, status, page_count, source_artifact_id FROM documents "
        "WHERE status='completed' ORDER BY created_at"
    ).fetchall()

    cases = []
    seen_sha: set[str] = set()

    for d in docs:
        art = con.execute(
            "SELECT id, sha256, path, media_type FROM artifacts WHERE id=?",
            (d["source_artifact_id"],),
        ).fetchone()
        if not art:
            continue
        src = Path(art["path"])
        if not src.exists():
            # try relative under backend/data
            alt = Path(__file__).resolve().parent / "data" / "artifacts" / "blobs"
            # sha-based path: blobs/ab/abcdef...
            sha = art["sha256"]
            src = alt / sha[:2] / sha
        if not src.exists():
            print("missing blob", d["id"], art["path"])
            continue
        if art["sha256"] in seen_sha:
            continue
        seen_sha.add(art["sha256"])

        ext = Path(d["filename"]).suffix.lower() or ".png"
        if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
            ext = ".jpg" if "jpeg" in (art["media_type"] or "") else ".png"
        out_name = f"{d['id'][:8]}_{Path(d['filename']).stem[:40]}{ext}"
        out_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in out_name)
        dest = IMG / out_name
        shutil.copy2(src, dest)

        words = con.execute(
            "SELECT text, decision_state, region_type, reading_order, confidence "
            "FROM word_evidence WHERE document_id=? ORDER BY reading_order",
            (d["id"],),
        ).fetchall()
        word_dicts = [dict(w) for w in words]
        # GT bootstrap: use human-corrected / ACCEPTED where possible; else leave empty for manual
        # For qualification we need TRUE ground truth — mark as needs_gt_review
        gt_lines = [
            w["text"]
            for w in word_dicts
            if w["decision_state"] not in {"CROSSED_OUT", "ILLEGIBLE"}
            and w["text"]
            and w["text"] != "[ILLEGIBLE]"
        ]
        case = {
            "id": d["id"][:8],
            "document_id": d["id"],
            "image_path": f"images/{out_name}",
            "source_filename": d["filename"],
            "ground_truth": "\n".join(gt_lines),
            "ground_truth_lines": gt_lines,
            "difficulty_tags": guess_tags(d["filename"], word_dicts),
            "notes": (
                "GT bootstrapped from prior OCR active lines — "
                "MUST be human-reviewed before claiming qualification scores."
            ),
            "needs_gt_review": True,
            "word_count": len(word_dicts),
        }
        cases.append(case)
        print("exported", out_name, "words", len(word_dicts))

    # Also copy local sample images if present
    extras = [
        ROOT / "samples" / "messy_note.png",
        Path(__file__).resolve().parent / "data" / "samples" / "messy_rx_sample.png",
        Path(__file__).resolve().parent / "data" / "samples" / "printed_report_sample.png",
    ]
    for p in extras:
        if not p.exists():
            continue
        dest = IMG / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
        # avoid duplicate if already exported by sha
        if any(c.get("image_path") == f"images/{p.name}" for c in cases):
            continue
        if p.name == "messy_note.png":
            gt = (
                "Patient notes 12/03/2026\n"
                "Dose 500 mg twice daily\n"
                "Follow up with Dr. Rao\n"
                "BP 120/80 stable\nmargin: urgent"
            )
            cases.append(
                {
                    "id": "messy_note",
                    "image_path": f"images/{p.name}",
                    "ground_truth": gt,
                    "ground_truth_lines": gt.splitlines(),
                    "difficulty_tags": ["medical", "margin_notes"],
                    "notes": "Synthetic OpenCV-font scaffold — use only as smoke test, not sole held-out.",
                    "needs_gt_review": False,
                    "is_synthetic": True,
                }
            )
        elif p.name == "messy_rx_sample.png":
            cases.append(
                {
                    "id": "messy_rx_sample",
                    "image_path": f"images/{p.name}",
                    "ground_truth": "",
                    "ground_truth_lines": [],
                    "difficulty_tags": ["medical", "extreme", "form"],
                    "notes": "Needs human GT transcription.",
                    "needs_gt_review": True,
                }
            )
        elif p.name == "printed_report_sample.png":
            cases.append(
                {
                    "id": "printed_report_sample",
                    "image_path": f"images/{p.name}",
                    "ground_truth": "",
                    "ground_truth_lines": [],
                    "difficulty_tags": ["form", "mixed_print_hw"],
                    "notes": "Needs human GT transcription.",
                    "needs_gt_review": True,
                }
            )

    # Prefer non-synthetic first for held_out; keep synthetics tagged
    manifest = {
        "name": "scribeproof_held_out_v1",
        "description": (
            "Held-out package built from local processed documents + samples. "
            "Cases with needs_gt_review=true require human ground-truth before official scoring."
        ),
        "cases": cases,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote", OUT / "manifest.json", "n=", len(cases))


if __name__ == "__main__":
    main()
