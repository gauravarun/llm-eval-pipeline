# models/interface.py
import requests
import json
import time
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
CHECKPOINTS_DIR = Path("checkpoints")


def query_model(prompt: str, model: str, temperature: float = 0.0, seed: int = 42) -> dict:
    """
    Query any Ollama model. Returns a result dict with response + metadata.
    temperature=0.0 and fixed seed = deterministic — essential for checkpoint comparison.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed
        }
    }
    start = time.perf_counter()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        latency = round(time.perf_counter() - start, 3)
        return {
            "response": data.get("response", "").strip(),
            "model": model,
            "latency_s": latency,
            "tokens_evaluated": data.get("eval_count", None),
            "error": None
        }
    except requests.exceptions.ConnectionError:
        return {"response": "", "model": model, "latency_s": 0,
                "tokens_evaluated": None,
                "error": "Ollama not running — start with: ollama serve"}
    except Exception as e:
        return {"response": "", "model": model, "latency_s": 0,
                "tokens_evaluated": None, "error": str(e)}


def list_available_models() -> list[str]:
    """Return models currently pulled in Ollama."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ── Checkpoint simulation ──────────────────────────────────────────────────────
# In real pre-training, checkpoints are saved model weights at training steps.
# Here we simulate multiple "checkpoints" using different Ollama models or
# temperature settings — so you can show checkpoint comparison in the dashboard.

CHECKPOINT_REGISTRY = [
    {
        "checkpoint_id": "ckpt_001",
        "label": "gemma3 / temp=0.0",
        "model": "gemma3",
        "temperature": 0.0,
        "description": "Baseline — deterministic, temperature 0"
    },
    {
        "checkpoint_id": "ckpt_002",
        "label": "gemma3 / temp=0.3",
        "model": "gemma3",
        "temperature": 0.3,
        "description": "Slight variance — simulates mid-training checkpoint"
    },
    {
        "checkpoint_id": "ckpt_003",
        "label": "gemma3 / temp=0.7",
        "model": "gemma3",
        "temperature": 0.7,
        "description": "Higher variance — simulates early training checkpoint"
    },
]


def get_checkpoint(checkpoint_id: str) -> dict | None:
    for ckpt in CHECKPOINT_REGISTRY:
        if ckpt["checkpoint_id"] == checkpoint_id:
            return ckpt
    return None


def get_all_checkpoints() -> list[dict]:
    return CHECKPOINT_REGISTRY


def save_checkpoint_result(checkpoint_id: str, results: list[dict], summary: dict):
    """Persist a checkpoint run to disk for later comparison."""
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    path = CHECKPOINTS_DIR / f"{checkpoint_id}_results.json"
    payload = {
        "checkpoint_id": checkpoint_id,
        "summary": summary,
        "results": results
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def load_checkpoint_result(checkpoint_id: str) -> dict | None:
    path = CHECKPOINTS_DIR / f"{checkpoint_id}_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_all_checkpoint_results() -> list[dict]:
    """Load every saved checkpoint run — used by the dashboard for comparison."""
    results = []
    for path in sorted(CHECKPOINTS_DIR.glob("ckpt_*_results.json")):
        results.append(json.loads(path.read_text()))
    return results