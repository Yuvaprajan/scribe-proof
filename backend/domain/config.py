"""Configurable decision thresholds and runtime settings."""

from __future__ import annotations

import os

from domain.policy import DecisionThresholds


def load_thresholds() -> DecisionThresholds:
    def f(name: str, default: float) -> float:
        return float(os.getenv(name, str(default)))

    return DecisionThresholds(
        accepted_min_score=f("SCRIBEPROOF_ACCEPTED_MIN_SCORE", 0.68),
        accepted_max_uncertainty=f("SCRIBEPROOF_ACCEPTED_MAX_UNCERTAINTY", 0.32),
        review_max_uncertainty=f("SCRIBEPROOF_REVIEW_MAX_UNCERTAINTY", 0.62),
        illegible_min_visual=f("SCRIBEPROOF_ILLEGIBLE_MIN_VISUAL", 0.22),
        consensus_min_support=int(os.getenv("SCRIBEPROOF_CONSENSUS_MIN_SUPPORT", "1")),
        high_risk_accepted_min_score=f("SCRIBEPROOF_HIGH_RISK_ACCEPTED_MIN_SCORE", 0.78),
        high_risk_max_uncertainty=f("SCRIBEPROOF_HIGH_RISK_MAX_UNCERTAINTY", 0.24),
        crossed_out_threshold=f("SCRIBEPROOF_CROSSED_OUT_THRESHOLD", 0.55),
        require_trocr_for_accepted=os.getenv(
            "SCRIBEPROOF_REQUIRE_TROCR_ACCEPTED", "true"
        ).lower()
        in {"1", "true", "yes"},
    )
