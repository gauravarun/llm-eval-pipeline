# registry/dataset_store.py
import json
import hashlib
from datetime import datetime
from pathlib import Path

REGISTRY_PATH = Path("checkpoints/dataset_registry.json")

# ── Built-in benchmark tasks ───────────────────────────────────────────────────

TASKS = [
    # ── Reasoning ─────────────────────────────────────────────────────────────
    {
        "id": "reasoning_001",
        "category": "reasoning",
        "difficulty": "easy",
        "prompt": (
            "If all Bloops are Razzies and all Razzies are Lazzies, "
            "are all Bloops definitely Lazzies? Answer yes or no and explain why."
        ),
        "expected_keywords": ["yes", "transitive", "all"],
        "failure_modes": ["wrong_answer", "no_justification"],
        "notes": "Classic syllogism. Tests deductive reasoning."
    },
    {
        "id": "reasoning_002",
        "category": "reasoning",
        "difficulty": "medium",
        "prompt": (
            "A bat and ball cost $1.10 in total. "
            "The bat costs $1.00 more than the ball. "
            "How much does the ball cost? Show your working."
        ),
        "expected_keywords": ["0.05", "five cents", "5 cents"],
        "failure_modes": ["intuitive_error", "says_10_cents"],
        "notes": "CRT question — intuitive answer ($0.10) is wrong. Tests System 2 reasoning."
    },
    {
        "id": "reasoning_003",
        "category": "reasoning",
        "difficulty": "hard",
        "prompt": (
            "Five people each have a different job: doctor, lawyer, teacher, engineer, chef. "
            "Alice is not the doctor or lawyer. Bob is not the teacher. "
            "Carol is the engineer. Dave is not the chef. Eve is the lawyer. "
            "What is Alice's job?"
        ),
        "expected_keywords": ["teacher", "alice"],
        "failure_modes": ["wrong_assignment", "incomplete_reasoning"],
        "notes": "Constraint satisfaction puzzle. Tests logical elimination."
    },

    # ── Factual recall ─────────────────────────────────────────────────────────
    {
        "id": "factual_001",
        "category": "factual",
        "difficulty": "easy",
        "prompt": "What is the capital of Australia? Answer in one word.",
        "expected_keywords": ["canberra"],
        "failure_modes": ["says_sydney", "says_melbourne"],
        "notes": "Common misconception trap — most people say Sydney."
    },
    {
        "id": "factual_002",
        "category": "factual",
        "difficulty": "medium",
        "prompt": (
            "Which programming language was created by Guido van Rossum? "
            "In what year was it first released publicly? Answer in one sentence."
        ),
        "expected_keywords": ["python", "1991"],
        "failure_modes": ["wrong_year", "wrong_language"],
        "notes": "Tests precise factual recall — year is commonly wrong."
    },
    {
        "id": "factual_003",
        "category": "factual",
        "difficulty": "hard",
        "prompt": (
            "What is the name of the phenomenon where light bends around "
            "a massive object due to gravity? Who first predicted it and in which theory?"
        ),
        "expected_keywords": ["gravitational lensing", "einstein", "general relativity"],
        "failure_modes": ["incomplete_answer", "wrong_attribution"],
        "notes": "Multi-part factual question testing depth of recall."
    },

    # ── Instruction following ──────────────────────────────────────────────────
    {
        "id": "instruction_001",
        "category": "instruction_following",
        "difficulty": "easy",
        "prompt": (
            "List exactly three fruits. "
            "Format your response as a numbered list. "
            "Do not include any other text."
        ),
        "expected_keywords": ["1.", "2.", "3."],
        "failure_modes": ["wrong_count", "wrong_format", "extra_text"],
        "notes": "Tests precise format adherence."
    },
    {
        "id": "instruction_002",
        "category": "instruction_following",
        "difficulty": "medium",
        "prompt": (
            "Write a sentence that contains exactly 10 words. "
            "Count carefully before answering."
        ),
        "expected_keywords": [],
        "expected_word_count": 10,
        "failure_modes": ["wrong_word_count"],
        "notes": "Counting constraint — models frequently miscalculate."
    },
    {
        "id": "instruction_003",
        "category": "instruction_following",
        "difficulty": "hard",
        "prompt": (
            "Answer the following question using only words that start with the letter S: "
            "What season comes after summer?"
        ),
        "expected_keywords": ["september", "season", "starts", "succeeds", "subsequent", "shift", "spring", "second", "so"],
        "failure_modes": ["uses_non_s_words", "wrong_answer"],
        "notes": "Dual constraint — correct content AND letter restriction. Hard to hold both."
    },

    # ── Math ───────────────────────────────────────────────────────────────────
    {
        "id": "math_001",
        "category": "math",
        "difficulty": "easy",
        "prompt": "What is 17 × 24? Show your working step by step.",
        "expected_keywords": ["408"],
        "failure_modes": ["arithmetic_error", "no_working"],
        "notes": "Basic multiplication — tests whether model shows working."
    },
    {
        "id": "math_002",
        "category": "math",
        "difficulty": "medium",
        "prompt": (
            "A train travels 120km in 1.5 hours. "
            "Another train travels 200km in 2.5 hours. "
            "Which train is faster and by how much?"
        ),
        "expected_keywords": ["80", "first", "faster"],
        "failure_modes": ["wrong_speed", "wrong_comparison"],
        "notes": "Two-step rate problem. Tests unit consistency and comparison."
    },
    {
        "id": "math_003",
        "category": "math",
        "difficulty": "hard",
        "prompt": (
            "What is the sum of the interior angles of a polygon with 9 sides? "
            "Show the formula you used."
        ),
        "expected_keywords": ["1260", "n-2", "180"],
        "failure_modes": ["wrong_formula", "arithmetic_error", "no_formula_shown"],
        "notes": "Geometry formula application — tests recall + execution."
    },

    # ── Code ───────────────────────────────────────────────────────────────────
    {
        "id": "code_001",
        "category": "code",
        "difficulty": "easy",
        "prompt": (
            "Write a Python function that takes a list of numbers and returns "
            "the sum of all even numbers in the list. Include one example call."
        ),
        "expected_keywords": ["def", "for", "if", "% 2", "return", "sum"],
        "failure_modes": ["syntax_error", "wrong_logic", "no_example"],
        "notes": "Basic Python — filter + accumulate pattern."
    },
    {
        "id": "code_002",
        "category": "code",
        "difficulty": "medium",
        "prompt": (
            "Write a Python function to check if a string is a palindrome. "
            "It should ignore spaces and capitalisation. Show two test cases."
        ),
        "expected_keywords": ["def", "lower", "replace", "return", "==", "reverse"],
        "failure_modes": ["ignores_spaces", "case_sensitive", "no_tests"],
        "notes": "String manipulation + normalisation. Common edge case: spaces."
    },
    {
        "id": "code_003",
        "category": "code",
        "difficulty": "hard",
        "prompt": (
            "Implement a Python function that finds the first non-repeating "
            "character in a string. Return None if all characters repeat. "
            "Explain your time complexity."
        ),
        "expected_keywords": ["def", "dict", "for", "return", "o(n)", "none"],
        "failure_modes": ["wrong_logic", "no_complexity", "o_n_squared"],
        "notes": "Hash map pattern. Tests algorithm thinking + complexity awareness."
    },

    # ── Hallucination resistance ───────────────────────────────────────────────
    {
        "id": "hallucination_001",
        "category": "hallucination",
        "difficulty": "medium",
        "prompt": (
            "Who won the Nobel Prize in Physics in 2031? "
            "If you don't know, say so clearly."
        ),
        "expected_keywords": ["don't know", "cannot", "not aware", "future", "no information", "2031"],
        "failure_modes": ["fabricates_winner", "confident_wrong_answer"],
        "notes": "Future date trap — correct answer is to refuse to speculate."
    },
    {
        "id": "hallucination_002",
        "category": "hallucination",
        "difficulty": "hard",
        "prompt": (
            "Summarise the key findings of the research paper "
            "'Quantum Gradient Descent via Entangled Backpropagation' by Dr. Lena Voss, 2023. "
            "If you are not sure this paper exists, say so."
        ),
        "expected_keywords": ["not sure", "cannot verify", "don't have", "may not exist", "unable to confirm"],
        "failure_modes": ["fabricates_summary", "invents_findings", "confident_hallucination"],
        "notes": "Fake paper trap — tests whether model fabricates plausible-sounding research."
    },
]


# ── Dataset versioning ──────────────────────────────────────────────────────────

def _compute_checksum(tasks: list) -> str:
    """Stable hash of the task list — changes if any task is edited."""
    content = json.dumps(tasks, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:10]


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"versions": []}


def _save_registry(registry: dict):
    REGISTRY_PATH.parent.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def register_dataset(tasks: list, name: str = "default") -> dict:
    """
    Register a dataset version. Returns the version record.
    If the checksum matches the latest version, returns existing record (no duplicate).
    """
    checksum = _compute_checksum(tasks)
    registry = _load_registry()

    if registry["versions"]:
        latest = registry["versions"][-1]
        if latest["checksum"] == checksum:
            return latest  # no change

    version = {
        "version_id": f"v{len(registry['versions']) + 1}",
        "name": name,
        "checksum": checksum,
        "num_tasks": len(tasks),
        "categories": list({t["category"] for t in tasks}),
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }
    registry["versions"].append(version)
    _save_registry(registry)
    return version


def get_tasks(split: str = "all") -> list:
    """
    Return tasks for evaluation.
    split='all'  → all tasks
    split='easy' / 'medium' / 'hard' → filter by difficulty
    """
    if split == "all":
        return TASKS
    return [t for t in TASKS if t.get("difficulty") == split]


def list_versions() -> list:
    return _load_registry().get("versions", [])