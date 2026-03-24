# scoring/scorer.py
import re

def score_response(response: str, task: dict) -> dict:
    """
    Multi-method scorer. Selects method based on task metadata.
    Returns score (0.0–1.0), method used, matched/missing signals, and pass/fail.
    """
    response_lower = response.lower().strip()

    # ── Method 1: Word count constraint ───────────────────────────────────────
    if "expected_word_count" in task:
        actual = len(response_lower.split())
        target = task["expected_word_count"]
        exact = actual == target
        # Partial credit: within 1 word
        close = abs(actual - target) <= 1
        score = 1.0 if exact else (0.5 if close else 0.0)
        return {
            "score": score,
            "method": "word_count",
            "matched_keywords": [f"word_count={actual}"] if exact else [],
            "missing_keywords": [] if exact else [f"expected {target} words, got {actual}"],
            "pass": score >= 0.5,
            "detail": f"Target: {target} words | Got: {actual} words"
        }

    # ── Method 2: Keyword presence ────────────────────────────────────────────
    keywords = task.get("expected_keywords", [])
    if not keywords:
        # No keywords defined and no word count — unscored
        return {
            "score": 0.0,
            "method": "unscored",
            "matched_keywords": [],
            "missing_keywords": [],
            "pass": False,
            "detail": "No scoring criteria defined for this task"
        }

    matched = [kw for kw in keywords if kw.lower() in response_lower]
    missing = [kw for kw in keywords if kw.lower() not in response_lower]
    score = round(len(matched) / len(keywords), 3)

    return {
        "score": score,
        "method": "keyword",
        "matched_keywords": matched,
        "missing_keywords": missing,
        "pass": score >= 0.5,
        "detail": f"{len(matched)}/{len(keywords)} keywords matched"
    }


def score_batch(responses: list[dict], tasks: list[dict]) -> list[dict]:
    """Score a list of (task, response) pairs. Returns enriched result dicts."""
    results = []
    for task, resp in zip(tasks, responses):
        scored = score_response(resp["response"], task)
        results.append({
            "id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "prompt": task["prompt"],
            "response": resp["response"],
            "model": resp.get("model", "unknown"),
            "latency_s": resp.get("latency_s", 0),
            "tokens_evaluated": resp.get("tokens_evaluated"),
            "error": resp.get("error"),
            **scored,
            "notes": task.get("notes", ""),
            "failure_modes": task.get("failure_modes", [])
        })
    return results