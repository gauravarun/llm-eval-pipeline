# scoring/stats_check.py
# Standalone test — runs stats functions on synthetic data, no model needed.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scoring.stats import (wilson_confidence_interval, score_variance,
                            chi_square_test, aggregate)

if __name__ == "__main__":
    print("\n  Stats Engine Check")
    print(f"  {'─'*50}")

    # 1. Wilson CI
    print("\n  Wilson confidence intervals (95%):")
    cases = [(8, 10), (1, 4), (15, 20), (3, 3)]
    for n_pass, n_total in cases:
        lo, hi = wilson_confidence_interval(n_pass, n_total)
        rate = round(n_pass / n_total, 2)
        print(f"    {n_pass}/{n_total}  pass rate={rate:.0%}  CI=[{lo:.3f}, {hi:.3f}]")

    # 2. Variance
    print("\n  Score variance:")
    sets = {
        "consistent  ": [0.9, 0.95, 0.88, 0.92, 0.91],
        "inconsistent": [0.2, 0.9, 0.5, 1.0, 0.1],
    }
    for label, scores in sets.items():
        v = score_variance(scores)
        print(f"    {label}  mean={v['mean']}  std={v['std']}  cv={v['cv']}")

    # 3. Chi-square significance test
    print("\n  Significance tests (checkpoint A vs B):")
    comparisons = [
        ("large improvement", 14, 20, 20, 20),
        ("small improvement", 16, 20, 18, 20),
        ("no difference   ", 10, 20, 10, 20),
    ]
    for label, pa, ta, pb, tb in comparisons:
        result = chi_square_test(pa, ta, pb, tb)
        print(f"    {label}  chi2={result['chi2']}  p={result['p_value']}  "
              f"significant={result['significant']}")

    # 4. Full aggregation on synthetic results
    print("\n  Full aggregation (synthetic data):")
    synthetic = [
        {"id": f"t{i}", "category": cat, "difficulty": diff,
         "score": score, "pass": score >= 0.5}
        for i, (cat, diff, score) in enumerate([
            ("reasoning", "easy", 1.0), ("reasoning", "hard", 0.33),
            ("factual",   "easy", 1.0), ("factual",   "medium", 0.5),
            ("math",      "medium", 0.67), ("math",   "hard", 0.0),
        ])
    ]
    summary = aggregate(synthetic)
    print(f"    Overall mean  : {summary['mean_score']}")
    print(f"    Pass rate     : {summary['pass_rate']} "
          f"CI={summary['ci_95']}")
    print(f"    Difficulty    : {summary['difficulty_breakdown']}")
    for cat, s in summary["category_breakdown"].items():
        print(f"    {cat:12s}  avg={s['avg_score']}  "
              f"CI={s['ci_95']}  cv={s['cv']}")
    print()