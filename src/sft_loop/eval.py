"""Same-seed eval for the synthetic uppercase task.

The MPS path loads the pinned model with transformers and, after training,
PEFT. It does not import Unsloth. The optional CUDA path still uses Unsloth.
When the requested device cannot load, the CLI prints an unsupported run
report and does not call the grader. A task score is returned only from
generations of the pinned model.
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


def _generate_scores(
    model: Any,
    tokenizer: Any,
    *,
    seed: int,
    device: str | None,
) -> dict[str, Any]:
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = eval_examples()
    predictions: list[str] = []
    completions: list[str] = []
    placed = device if device is not None else next(model.parameters()).device
    for example in examples:
        messages = [{"role": "user", "content": example["prompt"]}]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids.to(placed)
        else:
            input_ids = tokenized.to(placed)
        with torch.inference_mode():
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


def _execute_eval_trl_peft(
    *,
    seed: int,
    adapter_dir: Path | None,
    device: str,
) -> dict[str, Any]:
    """Greedy eval with transformers. After the train step, load a PEFT adapter.

    Does not import the optional CUDA package.
    """

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    if device == stack_mod.DEVICE_MPS and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device)
    model.eval()
    try:
        return _generate_scores(model, tokenizer, seed=seed, device=device)
    finally:
        if device == stack_mod.DEVICE_MPS and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _execute_eval_unsloth(
    *,
    seed: int,
    adapter_dir: Path | None,
) -> dict[str, Any]:
    """Optional CUDA-only eval. Not used by the MPS scored path."""

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
    return _generate_scores(model, tokenizer, seed=seed, device=None)


def execute_eval(
    *,
    seed: int,
    split: str,
    adapter_dir: Path | None,
    device: str | None = None,
) -> dict[str, Any]:
    """Generate on the held-out set and return one eval slice.

    ``split`` selects whether the base model or a saved adapter is loaded.
    Both calls must be given the same ``seed`` by the loop. Greedy decoding
    is used so the seed is the sampling control, not a hidden resample.
    ``device="mps"`` uses transformers and PEFT. ``device="cuda"`` uses the
    optional Unsloth loader.
    """

    require_seed(seed)
    if split not in SPLITS:
        raise ValueError("split must be before or after")
    if split == "after" and adapter_dir is None:
        raise RuntimeError("train_not_run")
    blocked = (
        stack_mod.stack_block_reason()
        if device is None
        else stack_mod.stack_block_reason(device=device)
    )
    if blocked is not None:
        raise RuntimeError(blocked)
    target = stack_mod.resolve_train_target(device)
    if target is None:
        raise RuntimeError("gpu_or_train_deps_unavailable")
    resolved_device, backend = target
    if backend == stack_mod.BACKEND_TRL_PEFT:
        return _execute_eval_trl_peft(
            seed=seed,
            adapter_dir=adapter_dir,
            device=resolved_device,
        )
    if backend == stack_mod.BACKEND_UNSLOTH:
        return _execute_eval_unsloth(seed=seed, adapter_dir=adapter_dir)
    raise RuntimeError("gpu_or_train_deps_unavailable")


def build_report(
    *,
    seed: int,
    model_id: str,
    split: str,
    run_id: str,
    device: str | None = None,
) -> dict[str, Any]:
    require_model_id(model_id)
    require_seed(seed)
    if split not in SPLITS:
        raise ValueError("split must be before or after")
    blocked = (
        stack_mod.stack_block_reason()
        if device is None
        else stack_mod.stack_block_reason(device=device)
    )
    reason = blocked if blocked is not None else "train_not_run"
    if device is None:
        prefix = f"Requested eval split {split} at seed {seed}. "
    else:
        prefix = f"Requested device {device} eval split {split} at seed {seed}. "
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
            "MPS uses transformers and PEFT. CUDA Unsloth is optional. "
            "Prints an unsupported run report when the stack cannot load. "
            "Does not invent a task score."
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--device", choices=stack_mod.DEVICES)
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
    device = args.device
    blocked = (
        stack_mod.stack_block_reason()
        if device is None
        else stack_mod.stack_block_reason(device=device)
    )
    if blocked is None and not (args.split == "after" and args.adapter is None):
        try:
            slice_ = execute_eval(
                seed=args.seed,
                split=args.split,
                adapter_dir=args.adapter,
                device=device,
            )
        except Exception:
            report = build_report(
                seed=args.seed,
                model_id=args.model_id,
                split=args.split,
                run_id=args.run_id,
                device=device,
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
            device=device,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    emit_json(report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
