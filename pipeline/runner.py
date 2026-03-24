# pipeline/runner.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
from registry.dataset_store import get_tasks, register_dataset, TASKS
from models.interface import (query_model, get_all_checkpoints,
                               save_checkpoint_result, load_checkpoint_result)
from scoring.scorer import score_batch
from scoring.stats import aggregate, chi_square_test
from colorama import Fore, Style, init

init(autoreset=True)


def run_checkpoint(checkpoint: dict, tasks: list) -> tuple[list, dict]:
    """Run a full evaluation for one checkpoint. Returns (results, summary)."""
    model = checkpoint["model"]
    temperature = checkpoint["temperature"]
    ckpt_id = checkpoint["checkpoint_id"]
    label = checkpoint["label"]

    print(f"\n  {Fore.CYAN}Running {label}{Style.RESET_ALL}")
    print(f"  {'─' * 50}")

    responses = []
    for i, task in enumerate(tasks):
        print(f"  [{i+1:02d}/{len(tasks)}] {task['id']}...", end=" ", flush=True)
        result = query_model(task["prompt"], model=model, temperature=temperature)
        responses.append(result)
        if result["error"]:
            print(f"{Fore.RED}ERROR{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}ok{Style.RESET_ALL}  ({result['latency_s']}s)")

    results = score_batch(responses, tasks)
    summary = aggregate(results)
    summary["checkpoint_id"] = ckpt_id
    summary["label"] = label
    summary["model"] = model
    summary["temperature"] = temperature
    summary["run_at"] = datetime.now().isoformat(timespec="seconds")
    summary["dataset_version"] = register_dataset(TASKS)["version_id"]

    save_checkpoint_result(ckpt_id, results, summary)
    return results, summary


def print_checkpoint_summary(summary: dict):
    """Print a single checkpoint result summary."""
    ci = summary["ci_95"]
    print(f"\n  {summary['label']}")
    print(f"  Mean score : {summary['mean_score']}  "
          f"(95% CI [{ci[0]}, {ci[1]}])")
    print(f"  Pass rate  : {summary['pass_rate'] * 100:.1f}%  "
          f"std={summary['std']}")
    print(f"  Tasks      : {summary['total_tasks']}  "
          f"| run at {summary['run_at']}")

    print(f"\n  {'Category':<22} {'Avg':>6} {'Pass':>6} {'CI 95%':>16} {'CV':>6}")
    print(f"  {'─'*60}")
    for cat, s in summary["category_breakdown"].items():
        ci_str = f"[{s['ci_95'][0]:.2f},{s['ci_95'][1]:.2f}]"
        print(f"  {cat:<22} {s['avg_score']:>6.3f} "
              f"{s['pass_rate']*100:>5.0f}%  {ci_str:>16}  {s['cv']:>5.3f}")


def print_comparison(summaries: list[dict]):
    """Compare all checkpoint summaries side by side."""
    print(f"\n\n  {'═'*65}")
    print(f"  CHECKPOINT COMPARISON")
    print(f"  {'═'*65}")
    print(f"  {'Checkpoint':<30} {'Score':>7} {'Pass%':>7} {'CI lower':>9} {'CI upper':>9}")
    print(f"  {'─'*65}")
    for s in summaries:
        ci = s["ci_95"]
        print(f"  {s['label']:<30} {s['mean_score']:>7.3f} "
              f"{s['pass_rate']*100:>6.1f}%  {ci[0]:>9.3f}  {ci[1]:>9.3f}")

    # Significance test: ckpt_001 vs ckpt_003 (baseline vs highest variance)
    if len(summaries) >= 3:
        a, b = summaries[0], summaries[2]
        n_a = a["total_tasks"]
        n_b = b["total_tasks"]
        pass_a = round(a["pass_rate"] * n_a)
        pass_b = round(b["pass_rate"] * n_b)
        test = chi_square_test(pass_a, n_a, pass_b, n_b)
        print(f"\n  Significance test ({a['label']} vs {b['label']}):")
        print(f"  chi2={test['chi2']}  p={test['p_value']}  "
              f"significant={test['significant']}")
        print(f"  → {test['note']}")

    print(f"\n  Difficulty breakdown ({summaries[0]['label']} baseline):")
    for diff, stats in summaries[0]["difficulty_breakdown"].items():
        print(f"  {diff:<10}  pass rate={stats['pass_rate']*100:.0f}%  "
              f"n={stats['total']}")
    print(f"  {'═'*65}\n")


def run_pipeline(checkpoints: str = "all", split: str = "all"):
    """
    Main entry point.
    checkpoints: 'all' | single checkpoint_id e.g. 'ckpt_001'
    split: 'all' | 'easy' | 'medium' | 'hard'
    """
    tasks = get_tasks(split)
    all_checkpoints = get_all_checkpoints()

    if checkpoints != "all":
        all_checkpoints = [c for c in all_checkpoints
                           if c["checkpoint_id"] == checkpoints]

    print(f"\n  {'═'*65}")
    print(f"  LLM EVALUATION PIPELINE")
    print(f"  {'═'*65}")
    print(f"  Tasks       : {len(tasks)} (split={split})")
    print(f"  Checkpoints : {len(all_checkpoints)}")
    print(f"  Dataset     : {register_dataset(TASKS)['version_id']}")
    print(f"  {'═'*65}")

    summaries = []
    for ckpt in all_checkpoints:
        # Use cached result if already run — avoid re-running expensive evals
        cached = load_checkpoint_result(ckpt["checkpoint_id"])
        if cached:
            print(f"\n  {Fore.YELLOW}Using cached result for "
                  f"{ckpt['label']}{Style.RESET_ALL}")
            summaries.append(cached["summary"])
            print_checkpoint_summary(cached["summary"])
        else:
            results, summary = run_checkpoint(ckpt, tasks)
            summaries.append(summary)
            print_checkpoint_summary(summary)

    if len(summaries) > 1:
        print_comparison(summaries)