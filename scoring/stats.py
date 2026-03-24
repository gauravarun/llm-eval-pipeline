# scoring/stats.py
# The interview-worthy module — statistical rigour over raw accuracy numbers.

import math
from collections import defaultdict


# ── Confidence intervals ───────────────────────────────────────────────────────

def wilson_confidence_interval(n_pass: int, n_total: int,
                                confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score interval for a pass rate.
    More accurate than normal approximation, especially at extreme proportions.
    Returns (lower, upper) bounds as proportions.

    Why Wilson over normal approximation:
    Normal CI can produce bounds outside [0,1] for small n or extreme rates.
    Wilson stays valid across all sample sizes — important for per-category scores
    where n may be as small as 2-4 tasks.
    """
    if n_total == 0:
        return (0.0, 0.0)

    z = _z_score(confidence)
    p_hat = n_pass / n_total
    n = n_total

    centre = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / (1 + z**2 / n)

    return (round(max(0.0, centre - margin), 3),
            round(min(1.0, centre + margin), 3))


def _z_score(confidence: float) -> float:
    """Z-score for common confidence levels."""
    table = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    return table.get(confidence, 1.960)


# ── Variance & spread ──────────────────────────────────────────────────────────

def score_variance(scores: list[float]) -> dict:
    """
    Returns mean, std deviation, min, max, and coefficient of variation.
    CV = std/mean — normalised spread. High CV = inconsistent category performance.
    """
    if not scores:
        return {}
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(variance)
    cv = round(std / mean, 3) if mean > 0 else 0.0

    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "cv": cv,
        "n": n
    }


# ── Significance testing ───────────────────────────────────────────────────────

def chi_square_test(pass_a: int, total_a: int,
                    pass_b: int, total_b: int) -> dict:
    """
    2×2 chi-square test comparing pass rates of two checkpoints/models.
    Returns chi2 statistic, p-value (approximated), and significance verdict.

    Used to answer: 'Is checkpoint B meaningfully better than checkpoint A,
    or could the difference be random noise?'

    Note: approximation holds well for n >= 5 per cell.
    For very small samples, treat results as indicative only.
    """
    fail_a = total_a - pass_a
    fail_b = total_b - pass_b

    total = total_a + total_b
    if total == 0:
        return {"chi2": 0, "p_value": 1.0, "significant": False, "note": "no data"}

    # Expected frequencies
    expected_pass_a = (pass_a + pass_b) * total_a / total
    expected_pass_b = (pass_a + pass_b) * total_b / total
    expected_fail_a = (fail_a + fail_b) * total_a / total
    expected_fail_b = (fail_a + fail_b) * total_b / total

    def safe_chi(obs, exp):
        return ((obs - exp) ** 2 / exp) if exp > 0 else 0

    chi2 = (safe_chi(pass_a, expected_pass_a) +
            safe_chi(pass_b, expected_pass_b) +
            safe_chi(fail_a, expected_fail_a) +
            safe_chi(fail_b, expected_fail_b))

    # P-value approximation for df=1 (chi-square CDF)
    p_value = _chi2_p_value(chi2)

    return {
        "chi2": round(chi2, 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "note": "p < 0.05 → difference is statistically significant" if p_value < 0.05
                else "p >= 0.05 → difference may be noise"
    }


def _chi2_p_value(chi2: float, df: int = 1) -> float:
    """
    Approximate p-value for chi-square distribution (df=1).
    Uses Wilson-Hilferty cube-root transformation — accurate to ~2 decimal places.
    """
    if chi2 <= 0:
        return 1.0
    # Cube-root approximation
    mu = 1 - 2 / (9 * df)
    sigma = math.sqrt(2 / (9 * df))
    z = (math.pow(chi2 / df, 1 / 3) - mu) / sigma
    return round(1 - _normal_cdf(z), 4)


def _normal_cdf(z: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    """
    Full statistical summary of an eval run.
    Per-category and overall: mean score, pass rate, CI, variance, CV.
    """
    category_results = defaultdict(list)
    for r in results:
        category_results[r["category"]].append(r)

    category_summary = {}
    for cat, items in category_results.items():
        scores = [r["score"] for r in items]
        n_pass = sum(1 for r in items if r["pass"])
        n_total = len(items)
        ci = wilson_confidence_interval(n_pass, n_total)
        var = score_variance(scores)

        category_summary[cat] = {
            "num_tasks": n_total,
            "avg_score": var["mean"],
            "pass_rate": round(n_pass / n_total, 3),
            "ci_95": ci,
            "std": var["std"],
            "cv": var["cv"],
            "min_score": var["min"],
            "max_score": var["max"],
            "failed_tasks": [r["id"] for r in items if not r["pass"]]
        }

    all_scores = [r["score"] for r in results]
    n_pass_total = sum(1 for r in results if r["pass"])
    overall_ci = wilson_confidence_interval(n_pass_total, len(results))
    overall_var = score_variance(all_scores)

    # Difficulty breakdown
    diff_breakdown = defaultdict(lambda: {"total": 0, "pass": 0})
    for r in results:
        d = r.get("difficulty", "unknown")
        diff_breakdown[d]["total"] += 1
        if r["pass"]:
            diff_breakdown[d]["pass"] += 1

    return {
        "mean_score": overall_var["mean"],
        "std": overall_var["std"],
        "pass_rate": round(n_pass_total / len(results), 3),
        "ci_95": overall_ci,
        "total_tasks": len(results),
        "difficulty_breakdown": {
            d: {
                "pass_rate": round(v["pass"] / v["total"], 3) if v["total"] else 0,
                "total": v["total"]
            }
            for d, v in diff_breakdown.items()
        },
        "category_breakdown": category_summary
    }