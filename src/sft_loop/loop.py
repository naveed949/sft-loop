"""Same-seed before / PEFT SFT / after loop.

When the stack cannot train, the result is one unsupported report. Eval and
train callables are not invoked. A scored report is built only after both
splits return and the train proof verifies.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sft_loop import stack as stack_mod
from sft_loop.report import (
    build_scored_report,
    default_proof_dir,
    detail_for,
    emit_json,
    require_model_id,
    require_seed,
    unsupported_report,
    validate_report,
)
from sft_loop.train import execute_train, weights_dir

PHASE3_RUN_ID = "phase3-unsupported-no-gpu"
PHASE3_SEED = 0

EvalFn = Callable[..., dict[str, Any]]
TrainFn = Callable[..., dict[str, Any]]


def _real_eval(*, seed: int, split: str, adapter_dir: Path | None) -> dict[str, Any]:
    from sft_loop.eval import execute_eval

    return execute_eval(seed=seed, split=split, adapter_dir=adapter_dir)


def _real_train(*, seed: int, proof_dir: Path) -> dict[str, Any]:
    return execute_train(seed=seed, proof_dir=proof_dir, adapter_dir=weights_dir())


def run_loop(
    *,
    seed: int,
    model_id: str,
    run_id: str,
    eval_fn: EvalFn | None = None,
    train_fn: TrainFn | None = None,
    proof_dir: Path | None = None,
) -> dict[str, Any]:
    require_model_id(model_id)
    require_seed(seed)
    directory = proof_dir if proof_dir is not None else default_proof_dir()
    blocked = stack_mod.stack_block_reason()
    if blocked is not None:
        prefix = f"Requested seed {seed} for the before eval and the after eval. "
        report = unsupported_report(
            run_id=run_id,
            reason=blocked,
            detail=detail_for(blocked, prefix),
            seed=seed,
        )
        validate_report(report)
        return report

    evaluate = eval_fn if eval_fn is not None else _real_eval
    train = train_fn if train_fn is not None else _real_train
    try:
        before = evaluate(seed=seed, split="before", adapter_dir=None)
        proof = train(seed=seed, proof_dir=directory)
        adapter = weights_dir() if train_fn is None else directory
        after = evaluate(seed=seed, split="after", adapter_dir=adapter)
        report = build_scored_report(
            run_id=run_id,
            seed=seed,
            before=before,
            after=after,
            train_proof=proof,
            proof_dir=directory,
        )
    except Exception as exc:
        sys.stderr.write(
            f"measured loop failed closed ({type(exc).__name__}). "
            "No before/after metrics were produced.\n"
        )
        prefix = f"Requested seed {seed} for the before eval and the after eval. "
        report = unsupported_report(
            run_id=run_id,
            reason="train_not_run",
            detail=detail_for("train_not_run", prefix),
            seed=seed,
        )
        validate_report(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same-seed before/after SFT loop, or print an unsupported "
            "report when the pinned stack cannot train."
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", default=PHASE3_RUN_ID)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = run_loop(seed=args.seed, model_id=args.model_id, run_id=args.run_id)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    emit_json(report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
