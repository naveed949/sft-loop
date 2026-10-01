"""Same-seed eval for the synthetic uppercase task.

When the pinned stack cannot load, the CLI prints an unsupported run report
and does not call the grader. A task score is returned only from generations
of the pinned model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from sft_loop import stack as stack_mod
from sft_loop.report import (
    MODEL_ID,
    detail_for,
    emit_json,
    require_model_id,
    require_seed,
    unsupported_report,
    validate_report,
)
from sft_loop.task import METRIC_NAME, eval_examples, grade, prediction_token

MAX_SEQ_LENGTH = 128
MAX_NEW_TOKENS = 8
SPLITS = ("before", "after")


def execute_eval(
    *,
    seed: int,
    split: str,
    adapter_dir: Path | None,
) -> dict[str, Any]:
    """Generate on the held-out set and return one eval slice.

    ``split`` selects whether the base model or a saved adapter is loaded.
    Both calls must be given the same ``seed`` by the loop. Greedy decoding
    is used so the seed is the sampling control, not a hidden resample.
    """

    require_seed(seed)
    if split not in SPLITS:
        raise ValueError("split must be before or after")
    if split == "after" and adapter_dir is None:
        raise RuntimeError("train_not_run")
    blocked = stack_mod.stack_block_reason()
    if blocked is not None:
        raise RuntimeError(blocked)

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = eval_examples()
    predictions: list[str] = []
    completions: list[str] = []
    device = next(model.parameters()).device
    for example in examples:
        messages = [{"role": "user", "content": example["prompt"]}]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids.to(device)
        else:
            input_ids = tokenized.to(device)
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output[0][input_ids.shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predictions.append(prediction_token(text))
        completions.append(example["completion"])
    score = grade(predictions, completions)
    return {
        "seed": seed,
        "metric_name": METRIC_NAME,
        "task_score": score,
        "n": len(completions),
    }


def build_report(
    *,
    seed: int,
    model_id: str,
    split: str,
    run_id: str,
) -> dict[str, Any]:
    require_model_id(model_id)
    require_seed(seed)
    if split not in SPLITS:
        raise ValueError("split must be before or after")
    blocked = stack_mod.stack_block_reason()
    reason = blocked if blocked is not None else "train_not_run"
    prefix = f"Requested eval split {split} at seed {seed}. "
    report = unsupported_report(
        run_id=run_id,
        reason=reason,
        detail=detail_for(reason, prefix),
        seed=seed,
    )
    validate_report(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Eval the pinned model on the synthetic task. "
            "Prints an unsupported run report when the stack cannot load. "
            "Does not invent a task score."
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--run-id", default="phase3-eval")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        require_model_id(args.model_id)
        require_seed(args.seed)
        if args.split not in SPLITS:
            raise ValueError("split must be before or after")
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    blocked = stack_mod.stack_block_reason()
    if blocked is None and not (args.split == "after" and args.adapter is None):
        try:
            slice_ = execute_eval(
                seed=args.seed,
                split=args.split,
                adapter_dir=args.adapter,
            )
        except Exception:
            report = build_report(
                seed=args.seed,
                model_id=args.model_id,
                split=args.split,
                run_id=args.run_id,
            )
            emit_json(report, args.out)
            return 0
        emit_json(slice_, args.out)
        return 0
    try:
        report = build_report(
            seed=args.seed,
            model_id=args.model_id,
            split=args.split,
            run_id=args.run_id,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    emit_json(report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
