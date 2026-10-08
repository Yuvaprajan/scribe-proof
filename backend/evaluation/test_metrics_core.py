"""Unit tests for CER/WER/FCE/FVE/risk-coverage and abstention anti-cheat."""

from __future__ import annotations

import pytest

from evaluation.metrics_core import (
    cer,
    exact_match,
    evaluate_word_list,
    wer,
)


def test_cer_reference_values():
    assert cer("abc", "abc") == 0.0
    assert cer("abc", "abd") == pytest.approx(1 / 3)
    assert cer("", "") == 0.0
    assert cer("", "x") == 1.0
    assert cer("hello", "hallo") == pytest.approx(0.2)


def test_wer_reference_values():
    assert wer("a b c", "a b c") == 0.0
    assert wer("a b c", "a x c") == pytest.approx(1 / 3)
    assert wer("", "") == 0.0


def test_exact_match_normalized():
    assert exact_match("Hello  World", "hello world") == 1.0
    assert exact_match("a", "b") == 0.0


def test_fce_and_fve():
    pairs = [
        {
            "reference": "hello",
            "hypothesis": "hello",
            "decision_state": "ACCEPTED",
            "confidence": 0.9,
        },
        {
            "reference": "world",
            "hypothesis": "wrld",
            "decision_state": "ACCEPTED",
            "confidence": 0.95,
        },
        {
            "reference": "foo",
            "hypothesis": "bar",
            "decision_state": "REVIEW_REQUIRED",
            "confidence": 0.4,
        },
        {
            "reference": "baz",
            "hypothesis": "[ILLEGIBLE]",
            "decision_state": "ILLEGIBLE",
            "confidence": 0.1,
        },
    ]
    agg = evaluate_word_list(pairs)
    assert agg.fce_count == 1
    assert agg.incorrect_accepted == 1
    assert agg.correct_accepted == 1
    assert agg.fve_count >= 2
    assert agg.coverage_accepted == 0.5
    assert agg.accepted_accuracy == 0.5
    assert agg.confidence_is_calibrated is False


def test_illegible_everywhere_does_not_win_at_full_coverage():
    pairs = [
        {
            "reference": f"line{i}",
            "hypothesis": "[ILLEGIBLE]",
            "decision_state": "ILLEGIBLE",
            "confidence": 0.01,
        }
        for i in range(8)
    ]
    agg = evaluate_word_list(pairs)
    assert agg.fce_rate == 0.0
    assert agg.coverage_accepted == 0.0
    assert agg.abstention_rate == 1.0
    full = [r for r in agg.risk_coverage if r["coverage"] == 1.0][0]
    assert full["cer"] > 0.5
    assert full["accepted_accuracy"] == 0.0


def test_risk_coverage_monotonic_pool_size():
    pairs = [
        {
            "reference": "aa",
            "hypothesis": "aa",
            "decision_state": "ACCEPTED",
            "confidence": 0.9 - 0.05 * i,
        }
        for i in range(4)
    ] + [
        {
            "reference": "zz",
            "hypothesis": "yy",
            "decision_state": "ACCEPTED",
            "confidence": 0.1,
        }
    ]
    agg = evaluate_word_list(pairs)
    ns = [r["n"] for r in agg.risk_coverage]
    assert ns == sorted(ns)


def test_selective_risk_empty_accepted_documented():
    pairs = [
        {
            "reference": "x",
            "hypothesis": "[ILLEGIBLE]",
            "decision_state": "ILLEGIBLE",
            "confidence": 0.0,
        }
    ]
    agg = evaluate_word_list(pairs)
    assert agg.selective_risk == 0.0
    assert agg.coverage_accepted == 0.0
    assert "selective_risk_note" in agg.extras
