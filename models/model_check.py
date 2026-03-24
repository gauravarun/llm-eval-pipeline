# models/model_check.py
# Run this to verify model interface is working and show available checkpoints.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.interface import query_model, list_available_models, get_all_checkpoints

if __name__ == "__main__":
    print("\n  Model Interface Check")
    print(f"  {'─'*50}")

    # 1. List available models
    models = list_available_models()
    if models:
        print(f"\n  Available models in Ollama:")
        for m in models:
            print(f"    · {m}")
    else:
        print("\n  No models found — is Ollama running?")

    # 2. Fire a test prompt
    print(f"\n  Running test prompt on gemma3...")
    result = query_model(
        prompt="Reply with exactly three words: evaluation pipeline ready.",
        model="gemma3"
    )

    if result["error"]:
        print(f"  ERROR: {result['error']}")
    else:
        print(f"  Response  : {result['response']}")
        print(f"  Latency   : {result['latency_s']}s")
        print(f"  Tokens    : {result['tokens_evaluated']}")

    # 3. Show checkpoint registry
    print(f"\n  Checkpoint registry:")
    for ckpt in get_all_checkpoints():
        print(f"    {ckpt['checkpoint_id']}  |  {ckpt['label']:30s}  |  {ckpt['description']}")

    print()