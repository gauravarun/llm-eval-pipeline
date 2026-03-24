# reporting/failure_report.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.interface import load_all_checkpoint_results
from reporting.aggregator import build_failure_matrix, get_regressions
from collections import defaultdict


def print_failure_report():
    all_results = load_all_checkpoint_results()
    if not all_results:
        print("No checkpoint results found. Run python run.py first.")
        return

    matrix = build_failure_matrix(all_results)
    regressions = get_regressions(matrix)

    print("\n  FAILURE MODE REPORT")
    print(f"  {'═'*65}")

    # Tasks that failed on ALL checkpoints
    consistent_failures = [
        (tid, data) for tid, data in matrix.items()
        if all(not v["pass"] for v in data["checkpoints"].values())
    ]
    print(f"\n  Consistent failures (failed on every checkpoint): "
          f"{len(consistent_failures)}")
    for tid, data in consistent_failures:
        print(f"  · {tid:<25} [{data['category']}]  "
              f"difficulty={data['difficulty']}")
        print(f"    {data['notes']}")
        for label, r in data["checkpoints"].items():
            if r["missing"]:
                print(f"    missing in {label}: {r['missing']}")

    # Regressions
    print(f"\n  Regressions (passed baseline, failed later): {len(regressions)}")
    for reg in regressions:
        print(f"  · {reg['task_id']:<25} [{reg['category']}]  "
              f"passed={reg['passed_on']}  failed={reg['failed_on']}")

    # Category gap analysis
    print(f"\n  Category gap analysis (avg score across checkpoints):")
    cat_scores = defaultdict(list)
    for cr in all_results:
        for cat, stats in cr["summary"]["category_breakdown"].items():
            cat_scores[cat].append(stats["avg_score"])

    sorted_cats = sorted(cat_scores.items(),
                         key=lambda x: sum(x[1]) / len(x[1]))
    for cat, scores in sorted_cats:
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {cat:<25} {bar}  {avg:.3f}")

    print(f"\n  {'═'*65}\n")