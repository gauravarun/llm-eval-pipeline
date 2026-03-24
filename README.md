# LLM Evaluation Pipeline

An end-to-end evaluation pipeline for language models — from benchmark curation to statistical analysis to an interactive HTML dashboard. Built to produce actionable training signals, not just accuracy numbers.

---

## What this is

Most eval scripts run a benchmark and report a score. This pipeline asks a harder question: **is the difference between two checkpoints real, or is it noise?**

Every result comes with a 95% confidence interval. Every checkpoint comparison is run through a significance test. Every failed task is categorised by failure mode and tracked across runs.

The design mirrors what a pre-training evaluation team actually needs: fast, reproducible, statistically rigorous signals that tell you whether a training run is moving in the right direction.

---

## Architecture

```
Dataset Registry          →   versioned benchmarks, deterministic splits
        ↓
Model Interface           →   Ollama adapter, checkpoint loader
        ↓
Scorer + Stats Engine     →   keyword scoring, Wilson CI, chi-square tests
        ↓
Result Aggregator         →   per-category, per-checkpoint, failure taxonomy
        ↓
Dashboard + Reports       →   HTML dashboard, failure matrix, regression detection
```

---

## Modules

| Module | What it does |
|---|---|
| `registry/dataset_store.py` | Versioned benchmark storage — MD5 checksum on task list, auto-increments version on change |
| `models/interface.py` | Model adapter (Ollama) + checkpoint registry. Deterministic inference: `temperature=0, seed=42` |
| `scoring/scorer.py` | Multi-method scoring: keyword presence, exact match, word-count constraints |
| `scoring/stats.py` | Wilson confidence intervals, score variance + CV, chi-square significance tests |
| `pipeline/runner.py` | Orchestrates full eval runs, caches results per checkpoint, prints comparison tables |
| `reporting/aggregator.py` | Builds comparison tables and failure matrix across checkpoints |
| `reporting/failure_report.py` | Gap analysis, regression detection, category bar charts |
| `reporting/dashboard.py` | Generates self-contained HTML dashboard with Chart.js visualisations |

---

## Statistical methods

### Wilson score confidence interval
Used for all pass rates instead of the normal approximation. The normal CI produces bounds outside [0, 1] for small samples — which matters when per-category n is 2–4 tasks. Wilson stays valid across all sample sizes.

```
8/10 pass rate = 80%  →  95% CI [0.490, 0.943]
1/4  pass rate = 25%  →  95% CI [0.046, 0.699]   ← wide, as it should be
3/3  pass rate = 100% →  95% CI [0.438, 1.000]   ← uncertainty despite perfect score
```

### Coefficient of variation (CV)
Normalised spread: `CV = std / mean`. High CV flags inconsistent category performance even when the mean looks acceptable. A category with mean=0.7 and CV=1.0 is a worse signal than mean=0.6 and CV=0.1.

### Chi-square significance test
2×2 contingency test comparing pass rates between two checkpoints. Answers: *is this improvement real, or could it be random variance?*

```
Large improvement (14/20 vs 20/20): chi2=7.059  p=0.0078  significant=True
Small improvement (16/20 vs 18/20): chi2=0.784  p=0.3797  significant=False
```

---

## Results — gemma3, 3 checkpoints

Checkpoints are simulated using temperature variation: `temp=0.0` (baseline), `temp=0.3` (mid-training), `temp=0.7` (early training).

| Checkpoint | Mean score | Pass rate | 95% CI |
|---|---|---|---|
| gemma3 / temp=0.0 | 0.709 | 76.5% | [0.527, 0.904] |
| gemma3 / temp=0.3 | 0.709 | 70.6% | [0.469, 0.867] |
| gemma3 / temp=0.7 | 0.709 | 70.6% | [0.469, 0.867] |

**Significance test (baseline vs temp=0.7):** chi2=0.151, p=0.699 — difference is noise, not signal. The model's capability is stable across temperature settings; variance is in the CI width, not the mean.

### Category breakdown (baseline)

| Category | Avg score | Pass rate | CV |
|---|---|---|---|
| code | 0.944 | 100% | 0.083 |
| factual | 1.000 | 100% | 0.000 |
| reasoning | 0.889 | 100% | 0.177 |
| math | 0.667 | 67% | 0.408 |
| instruction_following | 0.407 | 33% | 1.053 |
| hallucination | 0.167 | 0% | 1.000 |

### Key findings

**Hallucination resistance is the critical failure (0% pass rate, CV=1.0).** The model confidently fabricated a Nobel Prize winner for a future year (2031) and generated plausible-sounding summaries for a non-existent research paper. For deployment in knowledge-sensitive enterprise contexts, this is a disqualifying gap.

**Instruction following degrades sharply under dual constraints (33% pass rate, CV=1.053).** Single-format instructions (numbered list) pass reliably. Word-count constraints and letter-restriction tasks fail consistently — the model cannot hold both content and form constraints simultaneously.

**math_002 is a regression under temperature variance.** Passes at temp=0.0 and temp=0.3, fails at temp=0.7. The task requires multi-step unit comparison — capability that is sensitive to sampling noise. A reliable checkpoint should pass this deterministically.

**Difficulty calibration holds.** Easy=100%, Medium=67%, Hard=67% — the benchmark correctly stratifies task difficulty.

---

## Task failure matrix

Every task × every checkpoint, colour-coded PASS/FAIL. Consistent failures (all checkpoints) identify systematic gaps. Mixed results identify unstable capabilities worth targeting in training.

Generated automatically to `checkpoints/dashboard.html`.

---

## Running the pipeline

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `gemma3` pulled

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-eval-pipeline
cd llm-eval-pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install requests pandas tabulate colorama scipy numpy jinja2
```

### Commands

```bash
# Full pipeline — all checkpoints, all tasks (~10 min)
python run.py

# Single checkpoint
python run.py --checkpoint ckpt_001

# Hard tasks only
python run.py --split hard

# Re-run everything (ignore cache)
python run.py --fresh

# Generate HTML dashboard from saved results
python reporting/dashboard.py

# Print failure report + gap analysis
python reporting/failure_report.py

# Inspect dataset versions
cd registry && python version_check.py

# Test model interface
python models/model_check.py

# Test stats engine (no model needed)
python scoring/stats_check.py
```

---

## Extending the benchmark

Tasks in `registry/dataset_store.py` follow this schema:

```python
{
    "id": "category_001",
    "category": "category_name",
    "difficulty": "easy | medium | hard",
    "prompt": "...",
    "expected_keywords": ["keyword1", "keyword2"],   # OR
    "expected_word_count": 10,                        # word-count scoring
    "failure_modes": ["mode1", "mode2"],
    "notes": "What this task measures and why."
}
```

Adding a task changes the dataset checksum → auto-registers a new dataset version. The pipeline stores which dataset version each checkpoint run was evaluated against.

---

## Design decisions

**Keyword scoring over LLM-as-judge for checkpoint evals.**
Deterministic, zero-cost, produces the same score on every run. LLM-as-judge introduces variance that contaminates checkpoint comparisons. Reserved for open-ended generation tasks where keyword presence is insufficient.

**Wilson CI over normal approximation.**
Per-category sample sizes are small (2–6 tasks). The normal approximation breaks at the boundaries — Wilson stays valid.

**Chi-square for checkpoint comparison.**
The right question is not "which checkpoint scored higher?" but "is this difference statistically distinguishable from noise?" A 5% pass rate improvement with p=0.38 is not a training signal — it is random variance.

**Caching by checkpoint ID.**
Expensive eval runs are cached to `checkpoints/ckpt_XXX_results.json`. Rerunning the pipeline only evaluates missing checkpoints. Use `--fresh` to override.

---

## Planned extensions

- [ ] LLM-as-judge scoring for open-ended generation tasks
- [ ] Multi-model comparison (gemma3 vs mistral vs llama3)
- [ ] Weighted scoring by difficulty tier
- [ ] Scheduled checkpoint evaluation triggered by model version changes
- [ ] Integration with German Language Capability Evaluation Suite

---

## Related

[german-eval-suite](https://github.com/YOUR_USERNAME/german-eval-suite) — companion project: structured benchmark for German language capabilities in enterprise LLM deployment.

---

## Author

**Gaurav Arun** — MSc Artificial Intelligence, Berlin School of Business and Innovation  
[github.com/gauravarun](https://github.com/gauravarun) · [linkedin.com/in/gaurav-arun-528b6733a](https://linkedin.com/in/gaurav-arun-528b6733a)
