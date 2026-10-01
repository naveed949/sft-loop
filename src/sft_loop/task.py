"""Synthetic uppercase task and exact-match grader.

``task_score`` is the fraction of held-out words whose prediction equals the
uppercase target. That number is a task metric. It is not a safety judgment.
"""

from __future__ import annotations

METRIC_NAME = "exact_match"
TASK_ID = "uppercase-word-v1"

TRAIN_WORDS = (
    "apple",
    "river",
    "stone",
    "cloud",
    "maple",
    "tiger",
    "piano",
    "bread",
    "ocean",
    "cedar",
    "flame",
    "pearl",
    "glass",
    "north",
    "quiet",
    "lemon",
)

EVAL_WORDS = (
    "amber",
    "bridge",
    "cinder",
    "delta",
    "falcon",
    "garden",
    "harbor",
    "island",
)


def prompt_for(word: str) -> str:
    return f"Uppercase the word: {word}"


def target_for(word: str) -> str:
    return word.upper()


def train_examples() -> list[dict[str, str]]:
    return [
        {"prompt": prompt_for(word), "completion": target_for(word)}
        for word in TRAIN_WORDS
    ]


def eval_examples() -> list[dict[str, str]]:
    return [
        {"prompt": prompt_for(word), "completion": target_for(word)}
        for word in EVAL_WORDS
    ]


def prediction_token(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.split()[0]


def grade(predictions: list[str], completions: list[str]) -> float:
    """Exact-match rate in ``[0, 1]``. Empty input is refused."""

    if not completions or len(predictions) != len(completions):
        raise ValueError("predictions and completions must be the same non-empty length")
    hits = sum(
        prediction_token(prediction) == completion
        for prediction, completion in zip(predictions, completions)
    )
    return hits / len(completions)
