"""Honest research claim generator — never exaggerate significance."""

from __future__ import annotations

from typing import Any, Optional


ClaimLevel = str  # PROVEN | SUPPORTED_BUT_PRELIMINARY | NOT_PROVEN


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def generate_claims(
    summary: dict[str, Any],
    *,
    n_cases: int,
    n_real: int,
    n_synthetic: int,
    qualification_eligible: bool,
    paddle_key: str = "paddle_only",
    trocr_key: str = "trocr_lines",
    system_key: str = "scribeproof",
) -> dict[str, Any]:
    paddle = summary.get(paddle_key) or {}
    trocr = summary.get(trocr_key) or {}
    system = summary.get(system_key) or {}

    claims: list[dict[str, str]] = []

    def add(level: ClaimLevel, text: str) -> None:
        claims.append({"level": level, "claim": text})

    if n_cases < 15 or n_real < 15 or not qualification_eligible:
        add(
            "NOT_PROVEN",
            "HNX26EPS04 qualification / challenge-ready performance is NOT PROVEN: "
            f"n={n_cases} cases, real={n_real}, synthetic={n_synthetic}; "
            "qualification dataset gate not satisfied.",
        )

    p_cer = _f(paddle.get("mean_cer", paddle.get("cer")))
    s_cer = _f(system.get("mean_cer", system.get("cer")))
    p_fce = _f(paddle.get("mean_fce_rate", paddle.get("fce_rate")))
    s_fce = _f(system.get("mean_fce_rate", system.get("fce_rate")))
    p_cov = _f(paddle.get("mean_coverage_accepted", paddle.get("coverage_accepted")))
    s_cov = _f(
        system.get("mean_coverage_accepted", system.get("coverage_accepted"))
    )
    t_fce = _f(trocr.get("mean_fce_rate", trocr.get("fce_rate")))
    t_cov = _f(trocr.get("mean_coverage_accepted", trocr.get("coverage_accepted")))
    t_abs = _f(trocr.get("mean_abstention_rate", trocr.get("abstention_rate")))

    if n_cases >= 1 and not (p_cer != p_cer or s_cer != s_cer):  # not NaN
        cer_delta = abs(s_cer - p_cer)
        if cer_delta <= 0.05:
            add(
                "SUPPORTED_BUT_PRELIMINARY" if n_cases < 15 else "PROVEN",
                f"ScribeProof CER ({s_cer:.4f}) is essentially equal to independent "
                f"Paddle-only CER ({p_cer:.4f}) on this set (delta={cer_delta:.4f}). "
                "ScribeProof is not shown to outperform Paddle on CER.",
            )
        elif s_cer < p_cer:
            add(
                "SUPPORTED_BUT_PRELIMINARY" if n_cases < 15 else "PROVEN",
                f"ScribeProof CER ({s_cer:.4f}) is lower than Paddle-only ({p_cer:.4f}) "
                "on this set; sample size may be insufficient for significance.",
            )
        else:
            add(
                "SUPPORTED_BUT_PRELIMINARY" if n_cases < 15 else "PROVEN",
                f"ScribeProof CER ({s_cer:.4f}) is not better than Paddle-only "
                f"({p_cer:.4f}) on this set.",
            )

    if n_cases >= 1 and not (p_fce != p_fce or s_fce != s_fce):
        if s_fce + 1e-9 < p_fce:
            add(
                "SUPPORTED_BUT_PRELIMINARY" if n_cases < 15 else "PROVEN",
                f"ScribeProof reduces FCE relative to Paddle-only "
                f"({s_fce:.4f} vs {p_fce:.4f}) while accepted coverage is "
                f"{s_cov:.4f} vs {p_cov:.4f}.",
            )
        else:
            add(
                "SUPPORTED_BUT_PRELIMINARY",
                f"FCE reduction vs Paddle-only is not demonstrated "
                f"(ScribeProof FCE={s_fce:.4f}, Paddle FCE={p_fce:.4f}).",
            )

    if n_cases >= 1 and not (s_cov != s_cov or p_cov != p_cov):
        if s_cov + 1e-9 < p_cov:
            add(
                "SUPPORTED_BUT_PRELIMINARY",
                "ScribeProof has lower accepted coverage than Paddle-only "
                "(expected under selective abstention).",
            )

    if n_cases >= 1 and not (t_fce != t_fce):
        if t_fce < 0.15 and (t_cov < 0.25 or t_abs > 0.7):
            add(
                "SUPPORTED_BUT_PRELIMINARY",
                "TrOCR-only achieves low FCE primarily through heavy abstention / "
                f"low coverage (FCE={t_fce:.4f}, coverage={t_cov:.4f}). "
                "Do not treat low FCE alone as superiority.",
            )

    if n_real <= 1:
        add(
            "NOT_PROVEN",
            f"Only {n_real} non-synthetic case(s) with scorable GT; "
            "evidence is insufficient for qualification.",
        )

    add(
        "NOT_PROVEN",
        "Statistical significance of system differences is NOT PROVEN on n<15.",
    )

    return {
        "claim_policy": {
            "PROVEN": "Reproducible on qualification-eligible held-out set",
            "SUPPORTED_BUT_PRELIMINARY": "Observed on current small/dev set; not qualification evidence",
            "NOT_PROVEN": "Insufficient evidence or contradicted",
        },
        "dataset": {
            "n_cases": n_cases,
            "n_real": n_real,
            "n_synthetic": n_synthetic,
            "qualification_eligible": qualification_eligible,
            "status": "PRELIMINARY — NOT qualification evidence",
        },
        "claims": claims,
    }


def claims_to_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "## Research interpretation (auto-generated)",
        "",
        f"**Status:** {bundle['dataset']['status']}",
        "",
        f"- Cases: {bundle['dataset']['n_cases']} "
        f"(real={bundle['dataset']['n_real']}, "
        f"synthetic={bundle['dataset']['n_synthetic']})",
        f"- Qualification eligible: {bundle['dataset']['qualification_eligible']}",
        "",
        "| Level | Claim |",
        "|-------|-------|",
    ]
    for c in bundle.get("claims") or []:
        text = c["claim"].replace("|", "\\|")
        lines.append(f"| `{c['level']}` | {text} |")
    lines.append("")
    return "\n".join(lines)
