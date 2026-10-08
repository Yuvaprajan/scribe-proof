"""Dataset qualification gate, duplicates, placeholders, synthetic rejection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.dataset_integrity import (
    MIN_QUALIFICATION_REAL,
    refuse_qualified_status,
    validate_dataset_root,
)
from evaluation.experiment_configs import all_experiment_manifests, get_experiment_manifest
from evaluation.research_claims import generate_claims


ROOT = Path(__file__).resolve().parents[2] / "evaluation" / "dataset"


def test_validate_repo_dataset_not_qualification_ready():
    report = validate_dataset_root(ROOT)
    assert report.total_cases >= 5
    assert report.synthetic_cases >= 1
    assert report.qualification_eligible is False
    assert refuse_qualified_status(report) == "NOT_READY_DATASET"
    assert report.qualification_eligible_count < MIN_QUALIFICATION_REAL


def test_qualification_split_empty():
    report = validate_dataset_root(ROOT)
    assert report.splits.get("qualification", {}).get("n", 0) == 0


def test_dev_and_held_out_separated():
    report = validate_dataset_root(ROOT)
    assert report.splits.get("dev", {}).get("exists")
    assert report.splits.get("held_out", {}).get("exists")
    # Cross-split image copies (dev mirrors synthetics) must be flagged
    assert len(report.cross_split_leaks) >= 1 or len(report.duplicate_image_groups) >= 1


def test_placeholder_detected(tmp_path: Path):
    (tmp_path / "qualification" / "images").mkdir(parents=True)
    img = tmp_path / "qualification" / "images" / "placeholder_x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    manifest = {
        "cases": [
            {
                "id": "placeholder_x",
                "image": "images/placeholder_x.png",
                "ground_truth": "hello",
                "source": "test",
                "is_synthetic": False,
                "difficulty": "easy",
                "script": "latin",
                "tags": ["easy"],
            }
        ]
    }
    (tmp_path / "qualification" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (tmp_path / "dev").mkdir()
    (tmp_path / "held_out").mkdir()
    (tmp_path / "dev" / "manifest.json").write_text('{"cases":[]}', encoding="utf-8")
    (tmp_path / "held_out" / "manifest.json").write_text('{"cases":[]}', encoding="utf-8")
    report = validate_dataset_root(tmp_path)
    assert report.placeholder_cases >= 1
    assert report.qualification_eligible is False


def test_synthetic_in_qualification_rejected(tmp_path: Path):
    (tmp_path / "qualification" / "images").mkdir(parents=True)
    img = tmp_path / "qualification" / "images" / "real_looking.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x01" * 64)
    manifest = {
        "cases": [
            {
                "id": "syn_q",
                "image": "images/real_looking.png",
                "ground_truth": "hello world",
                "source": "synth",
                "is_synthetic": True,
                "difficulty": "hard",
                "script": "latin",
                "tags": ["extreme"],
            }
        ]
    }
    (tmp_path / "qualification" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for split in ("dev", "held_out"):
        (tmp_path / split).mkdir(exist_ok=True)
        (tmp_path / split / "manifest.json").write_text('{"cases":[]}', encoding="utf-8")
    report = validate_dataset_root(tmp_path)
    assert report.qualification_eligible is False
    assert any("synthetic" in f for f in report.qualification_failures)


def test_duplicate_images_detected(tmp_path: Path):
    for split in ("dev", "held_out", "qualification"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "manifest.json").write_text('{"cases":[]}', encoding="utf-8")
    blob = b"\x89PNG\r\n\x1a\n" + b"dup" * 20
    a = tmp_path / "dev" / "images" / "a.png"
    b = tmp_path / "held_out" / "images" / "b.png"
    a.write_bytes(blob)
    b.write_bytes(blob)
    (tmp_path / "dev" / "manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "a",
                        "image": "images/a.png",
                        "ground_truth": "x",
                        "source": "t",
                        "is_synthetic": True,
                        "difficulty": "easy",
                        "script": "latin",
                        "tags": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "held_out" / "manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "b",
                        "image": "images/b.png",
                        "ground_truth": "y",
                        "source": "t",
                        "is_synthetic": False,
                        "difficulty": "easy",
                        "script": "latin",
                        "tags": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = validate_dataset_root(tmp_path)
    assert len(report.duplicate_image_groups) >= 1
    assert len(report.cross_split_leaks) >= 1


def test_experiment_manifests_isolate_abc():
    manifests = all_experiment_manifests()
    for key in "ABC":
        forbidden = set(manifests[key].get("forbidden_scribeproof_components") or [])
        assert "DecisionEngine" in forbidden or key == "C"
        assert manifests[key]["decision_engine"] is False
        assert manifests[key]["verifier"] is False
    assert get_experiment_manifest("D")["decision_engine"] is True


def test_research_claims_do_not_claim_cer_win():
    summary = {
        "paddle_only": {
            "mean_cer": 0.36,
            "mean_fce_rate": 0.67,
            "mean_coverage_accepted": 0.89,
        },
        "trocr_lines": {
            "mean_cer": 0.90,
            "mean_fce_rate": 0.05,
            "mean_coverage_accepted": 0.05,
            "mean_abstention_rate": 0.95,
        },
        "scribeproof": {
            "mean_cer": 0.36,
            "mean_fce_rate": 0.52,
            "mean_coverage_accepted": 0.75,
        },
    }
    bundle = generate_claims(
        summary, n_cases=5, n_real=1, n_synthetic=4, qualification_eligible=False
    )
    text = " ".join(c["claim"] for c in bundle["claims"]).lower()
    assert "essentially equal" in text or "not better" in text
    assert "not proven" in text
    assert any(c["level"] == "NOT_PROVEN" for c in bundle["claims"])
    assert "outperform paddle on cer" in text or "not better" in text
    assert "scribeproof beats" not in text
