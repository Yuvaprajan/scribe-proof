"""Deterministic dataset integrity + qualification gate for HNX26EPS04.

Qualification requires:
  - minimum 15 real (is_synthetic=false) hard pages with GT
  - target 30 real hard pages
  - no synthetic / placeholder / duplicate / empty-GT / leaked cases
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


REQUIRED_CASE_FIELDS = (
    "id",
    "image",  # or image_path
    "ground_truth",
    "source",
    "is_synthetic",
    "difficulty",  # or difficulty_tags
    "script",
    "tags",  # or difficulty_tags
)

MIN_QUALIFICATION_REAL = 15
TARGET_QUALIFICATION_REAL = 30

PLACEHOLDER_NAME_RE = re.compile(
    r"(placeholder|synthetic|opencv.?font|lorem|dummy)", re.I
)


@dataclass
class CaseRecord:
    split: str
    id: str
    image_path: Path
    image_sha256: str
    ground_truth: str
    gt_sha256: str
    is_synthetic: bool
    needs_gt_review: bool
    difficulty_tags: list[str]
    source: str
    script: str
    notes: str = ""
    issues: list[str] = field(default_factory=list)


@dataclass
class IntegrityReport:
    total_cases: int = 0
    real_cases: int = 0
    synthetic_cases: int = 0
    extreme_cases: int = 0
    missing_gt: int = 0
    placeholder_cases: int = 0
    duplicate_image_groups: list[list[str]] = field(default_factory=list)
    duplicate_gt_groups: list[list[str]] = field(default_factory=list)
    cross_split_leaks: list[dict[str, Any]] = field(default_factory=list)
    qualification_eligible_count: int = 0
    qualification_eligible: bool = False
    qualification_failures: list[str] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)
    splits: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_gt(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def resolve_image(dataset_dir: Path, case: dict[str, Any]) -> Optional[Path]:
    raw = case.get("image") or case.get("image_path") or ""
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p
    for cand in (
        dataset_dir / raw,
        dataset_dir / "images" / Path(raw).name,
        dataset_dir.parent / raw,
    ):
        if cand.is_file():
            return cand
    return None


def load_split_cases(split_dir: Path) -> list[dict[str, Any]]:
    manifest = split_dir / "manifest.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    cases = data.get("cases") or data
    if not isinstance(cases, list):
        return []
    return cases


def _looks_like_ocr_generated_gt(gt: str, notes: str) -> bool:
    blob = f"{gt}\n{notes}".lower()
    if "generated from ocr" in blob or "ocr output as gt" in blob:
        return True
    if "auto-transcribed" in blob and "human" not in blob:
        return True
    return False


def inspect_case(split: str, dataset_dir: Path, case: dict[str, Any]) -> CaseRecord:
    cid = str(case.get("id") or "unknown")
    issues: list[str] = []
    img = resolve_image(dataset_dir, case)
    gt = str(case.get("ground_truth") or "")
    is_synth = bool(case.get("is_synthetic"))
    needs_review = bool(case.get("needs_gt_review"))
    tags = list(case.get("tags") or case.get("difficulty_tags") or [])
    if case.get("difficulty") and case["difficulty"] not in tags:
        tags = tags + [str(case["difficulty"])]
    source = str(case.get("source") or case.get("notes") or "")
    script = str(case.get("script") or "unknown")
    notes = str(case.get("notes") or "")

    # Required fields for qualification-shaped records
    if split == "qualification":
        if not case.get("source"):
            issues.append("missing_source")
        if case.get("script") in (None, "", "unknown") and "script" not in case:
            issues.append("missing_script")
        if not (case.get("difficulty") or case.get("difficulty_tags") or case.get("tags")):
            issues.append("missing_difficulty_or_tags")
        if "is_synthetic" not in case:
            issues.append("missing_is_synthetic")

    if img is None:
        issues.append("missing_image")
        sha = ""
    else:
        sha = _sha256_file(img)
        name = img.name
        if PLACEHOLDER_NAME_RE.search(name) or PLACEHOLDER_NAME_RE.search(cid):
            issues.append("placeholder_name")
            if not is_synth:
                issues.append("placeholder_not_marked_synthetic")

    if not gt.strip():
        issues.append("missing_or_empty_ground_truth")
    if needs_review:
        issues.append("needs_gt_review")
    if is_synth and split == "qualification":
        issues.append("synthetic_in_qualification")
    if _looks_like_ocr_generated_gt(gt, notes):
        issues.append("gt_appears_ocr_generated")

    return CaseRecord(
        split=split,
        id=cid,
        image_path=img or Path(""),
        image_sha256=sha,
        ground_truth=gt,
        gt_sha256=_sha256_bytes(_normalize_gt(gt).encode("utf-8")) if gt.strip() else "",
        is_synthetic=is_synth,
        needs_gt_review=needs_review,
        difficulty_tags=tags,
        source=source,
        script=script,
        notes=notes,
        issues=issues,
    )


def validate_dataset_root(
    root: Path,
    *,
    splits: tuple[str, ...] = ("dev", "held_out", "qualification"),
) -> IntegrityReport:
    report = IntegrityReport()
    records: list[CaseRecord] = []

    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            report.splits[split] = {"exists": False, "n": 0}
            continue
        cases = load_split_cases(split_dir)
        split_recs = [inspect_case(split, split_dir, c) for c in cases]
        records.extend(split_recs)
        report.splits[split] = {
            "exists": True,
            "n": len(split_recs),
            "real": sum(1 for r in split_recs if not r.is_synthetic),
            "synthetic": sum(1 for r in split_recs if r.is_synthetic),
            "with_gt": sum(1 for r in split_recs if r.ground_truth.strip()),
        }

    report.total_cases = len(records)
    report.real_cases = sum(1 for r in records if not r.is_synthetic)
    report.synthetic_cases = sum(1 for r in records if r.is_synthetic)
    report.extreme_cases = sum(
        1 for r in records if "extreme" in {t.lower() for t in r.difficulty_tags}
    )
    report.missing_gt = sum(1 for r in records if not r.ground_truth.strip())
    report.placeholder_cases = sum(
        1 for r in records if "placeholder_name" in r.issues
    )

    # Duplicate images by SHA
    by_hash: dict[str, list[CaseRecord]] = defaultdict(list)
    for r in records:
        if r.image_sha256:
            by_hash[r.image_sha256].append(r)
    for h, group in by_hash.items():
        if len(group) > 1:
            report.duplicate_image_groups.append(
                [f"{g.split}:{g.id}" for g in group]
            )

    # Duplicate GT (non-empty)
    by_gt: dict[str, list[CaseRecord]] = defaultdict(list)
    for r in records:
        if r.gt_sha256:
            by_gt[r.gt_sha256].append(r)
    for h, group in by_gt.items():
        ids = {g.id for g in group}
        if len(group) > 1 and len(ids) > 1:
            report.duplicate_gt_groups.append([f"{g.split}:{g.id}" for g in group])

    # Cross-split image leaks (same SHA in multiple splits)
    for h, group in by_hash.items():
        splits_present = {g.split for g in group}
        if len(splits_present) > 1:
            report.cross_split_leaks.append(
                {
                    "sha256": h,
                    "cases": [f"{g.split}:{g.id}" for g in group],
                    "splits": sorted(splits_present),
                }
            )

    # Qualification eligibility
    qual = [r for r in records if r.split == "qualification"]
    failures: list[str] = []
    eligible = []
    for r in qual:
        bad = list(r.issues)
        if r.is_synthetic:
            bad.append("synthetic")
        if not r.ground_truth.strip():
            bad.append("no_gt")
        if r.needs_gt_review:
            bad.append("needs_review")
        if not bad:
            eligible.append(r)
        else:
            failures.append(f"{r.id}: {', '.join(sorted(set(bad)))}")

    # Also fail if duplicates / leaks touch qualification
    for group in report.duplicate_image_groups:
        if any(x.startswith("qualification:") for x in group):
            failures.append(f"duplicate_image_in_qualification: {group}")
    for leak in report.cross_split_leaks:
        if "qualification" in leak.get("splits", []):
            failures.append(f"cross_split_leak: {leak['cases']}")

    report.qualification_eligible_count = len(eligible)
    if len(eligible) < MIN_QUALIFICATION_REAL:
        failures.append(
            f"fewer_than_{MIN_QUALIFICATION_REAL}_real_qualification_pages "
            f"(have {len(eligible)}; target {TARGET_QUALIFICATION_REAL})"
        )
    report.qualification_failures = failures
    report.qualification_eligible = len(failures) == 0 and len(eligible) >= MIN_QUALIFICATION_REAL

    report.cases = [
        {
            "split": r.split,
            "id": r.id,
            "image": str(r.image_path),
            "sha256": r.image_sha256,
            "is_synthetic": r.is_synthetic,
            "has_gt": bool(r.ground_truth.strip()),
            "tags": r.difficulty_tags,
            "issues": r.issues,
        }
        for r in records
    ]
    return report


def refuse_qualified_status(report: IntegrityReport) -> str:
    """Return the qualification decision string (never exaggerate)."""
    if not report.qualification_eligible:
        if report.qualification_eligible_count < MIN_QUALIFICATION_REAL:
            return "NOT_READY_DATASET"
        if any("synthetic" in f for f in report.qualification_failures):
            return "NOT_READY_DATASET"
        if any("missing" in f or "no_gt" in f for f in report.qualification_failures):
            return "NOT_READY_DATASET"
        return "NOT_READY_DATASET"
    return "QUALIFIED_FOR_INTERNAL_EVALUATION"


def write_integrity_reports(
    report: IntegrityReport, out_json: Path, out_md: Path
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    lines = [
        "# Dataset integrity report",
        "",
        f"- Total cases: {report.total_cases}",
        f"- Real cases: {report.real_cases}",
        f"- Synthetic cases: {report.synthetic_cases}",
        f"- Extreme-tagged cases: {report.extreme_cases}",
        f"- Missing GT: {report.missing_gt}",
        f"- Placeholder-named: {report.placeholder_cases}",
        f"- Duplicate image groups: {len(report.duplicate_image_groups)}",
        f"- Duplicate GT groups: {len(report.duplicate_gt_groups)}",
        f"- Cross-split leaks: {len(report.cross_split_leaks)}",
        f"- Qualification eligible count: {report.qualification_eligible_count}",
        f"- Qualification eligible: **{report.qualification_eligible}**",
        f"- Decision: `{refuse_qualified_status(report)}`",
        "",
        "## Splits",
        "",
    ]
    for name, info in report.splits.items():
        lines.append(f"- `{name}`: {json.dumps(info)}")
    if report.qualification_failures:
        lines.extend(["", "## Qualification failures", ""])
        for f in report.qualification_failures:
            lines.append(f"- {f}")
    if report.duplicate_image_groups:
        lines.extend(["", "## Duplicate images", ""])
        for g in report.duplicate_image_groups:
            lines.append(f"- {', '.join(g)}")
    if report.cross_split_leaks:
        lines.extend(["", "## Cross-split leaks", ""])
        for leak in report.cross_split_leaks:
            lines.append(f"- {leak}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate evaluation dataset integrity")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("evaluation/dataset"),
        help="Dataset root containing dev/held_out/qualification",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("evaluation/reports/dataset_integrity.json"),
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=Path("evaluation/reports/dataset_integrity.md"),
    )
    args = ap.parse_args()
    report = validate_dataset_root(args.root.resolve())
    write_integrity_reports(report, args.out_json, args.out_md)
    decision = refuse_qualified_status(report)
    print(f"qualification_eligible={report.qualification_eligible} decision={decision}")
    print(f"Wrote {args.out_json} and {args.out_md}")
    if not report.qualification_eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
