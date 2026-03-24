# reporting/aggregator.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.interface import load_all_checkpoint_results


def build_comparison_table(checkpoint_results: list[dict]) -> list[dict]:
    """
    Flatten all checkpoint summaries into a list of rows for the dashboard.
    Each row = one checkpoint × one category.
    """
    rows = []
    for cr in checkpoint_results:
        s = cr["summary"]
        for cat, stats in s["category_breakdown"].items():
            rows.append({
                "checkpoint_id": s["checkpoint_id"],
                "label": s["label"],
                "category": cat,
                "avg_score": stats["avg_score"],
                "pass_rate": stats["pass_rate"],
                "ci_lower": stats["ci_95"][0],
                "ci_upper": stats["ci_95"][1],
                "cv": stats["cv"],
                "failed_tasks": stats["failed_tasks"],
            })
    return rows


def build_failure_matrix(checkpoint_results: list[dict]) -> dict:
    """
    For each task, record which checkpoints passed and which failed.
    Returns dict: task_id → {label: pass/fail, ...}
    """
    matrix = {}
    for cr in checkpoint_results:
        label = cr["summary"]["label"]
        for r in cr["results"]:
            tid = r["id"]
            if tid not in matrix:
                matrix[tid] = {
                    "category": r["category"],
                    "difficulty": r["difficulty"],
                    "notes": r.get("notes", ""),
                    "checkpoints": {}
                }
            matrix[tid]["checkpoints"][label] = {
                "pass": r["pass"],
                "score": r["score"],
                "missing": r.get("missing_keywords", [])
            }
    return matrix


def get_regressions(matrix: dict) -> list[dict]:
    """
    Find tasks that passed on ckpt_001 (temp=0) but failed on a later checkpoint.
    These are regressions — capability that degrades under variance.
    """
    regressions = []
    for tid, data in matrix.items():
        ckpts = data["checkpoints"]
        labels = list(ckpts.keys())
        if len(labels) < 2:
            continue
        baseline_pass = ckpts[labels[0]]["pass"]
        for label in labels[1:]:
            if baseline_pass and not ckpts[label]["pass"]:
                regressions.append({
                    "task_id": tid,
                    "category": data["category"],
                    "difficulty": data["difficulty"],
                    "passed_on": labels[0],
                    "failed_on": label,
                    "notes": data["notes"]
                })
    return regressions