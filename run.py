# run.py
# Main entry point — runs the full pipeline across all checkpoints.
# Usage:
#   python run.py                          → all checkpoints, all tasks
#   python run.py --checkpoint ckpt_001    → single checkpoint
#   python run.py --split hard             → hard tasks only
#   python run.py --fresh                  → ignore cache, re-run everything

import sys
import os

# Remove cached checkpoint results if --fresh flag passed
if "--fresh" in sys.argv:
    from pathlib import Path
    for f in Path("checkpoints").glob("ckpt_*_results.json"):
        f.unlink()
    print("  Cache cleared.")

checkpoint_arg = "all"
split_arg = "all"

if "--checkpoint" in sys.argv:
    idx = sys.argv.index("--checkpoint")
    checkpoint_arg = sys.argv[idx + 1]

if "--split" in sys.argv:
    idx = sys.argv.index("--split")
    split_arg = sys.argv[idx + 1]

from pipeline.runner import run_pipeline
run_pipeline(checkpoints=checkpoint_arg, split=split_arg)