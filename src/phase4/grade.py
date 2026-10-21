"""Deterministic canonical-figure grader and prompt template."""

from __future__ import annotations

import re

TASK_ID = "tenk-extract-v1"
METRIC_NAME = "exact_match"
PROMPT_PREAMBLE = (
    "Read the 10-K excerpt. Reply with only the canonical figure, no extra words."
)
PROMPT_CHUNK = "Excerpt:"
PROMPT_QUESTION = "Question:"
PROMPT_ANSWER = "Answer:"

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[.,;:]+$")
_COMMA = re.compile(r",")
_DOLLAR = re.compile(r"^\$")
_PARENS = re.compile(r"^\((.*)\)$")
BARE_FIGURE = re.compile(r"^-?\d+(?:\.\d+)?%?$")


def canonicalize(text: str) -> str:
    if text is None:
        return ""
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    line = _WS.sub("", line)
    line = _PUNCT.sub("", line)
    parens = _PARENS.match(line)
    if parens:
        line = "-" + parens.group(1)
    line = _DOLLAR.sub("", line)
    line = _COMMA.sub("", line)
    return line


def render_prompt(chunk: str, question: str) -> str:
    return (
        f"{PROMPT_PREAMBLE}\n\n"
        f"{PROMPT_CHUNK}\n{chunk.strip()}\n\n"
        f"{PROMPT_QUESTION}\n{question.strip()}\n\n"
        f"{PROMPT_ANSWER}\n"
    )


def prompt_template_static() -> str:
    return (
        f"{PROMPT_PREAMBLE}\n\n"
        f"{PROMPT_CHUNK}\n\n"
        f"{PROMPT_QUESTION}\n\n"
        f"{PROMPT_ANSWER}\n"
    )


def grade_one(prediction: str, target: str) -> int:
    left = canonicalize(prediction)
    right = canonicalize(target)
    if not left or not right:
        return 0
    if not BARE_FIGURE.fullmatch(left) or not BARE_FIGURE.fullmatch(right):
        return 0
    return int(left == right)


def grade(predictions: list[str], completions: list[str]) -> float:
    if not completions or len(predictions) != len(completions):
        raise ValueError("predictions and completions must be the same non-empty length")
    hits = sum(grade_one(prediction, completion) for prediction, completion in zip(predictions, completions))
    return hits / len(completions)
