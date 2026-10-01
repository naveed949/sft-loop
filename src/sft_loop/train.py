"""Unsloth + TRL PEFT SFT on the pinned <=3B instruct model.

The public report builder fail-closes when CUDA, Unsloth, or TRL is missing.
``execute_train`` is the only function that calls ``trainer.train``. It refuses
to start unless ``stack_block_reason`` is clear, and it refuses to write proof
unless the trainer reports ``global_step >= 1`` and a loss in ``log_history``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from sft_loop import stack as stack_mod
from sft_loop.report import (
    ADAPTER_CONFIG_NAME,
    MODEL_ID,
    TRAIN_LOG_NAME,
    detail_for,
    emit_json,
    format_loss,
    render_train_log,
    require_model_id,
    require_seed,
    sha256_file,
    unsupported_report,
    validate_report,
)
from sft_loop.task import train_examples

LORA_R = 8
LORA_ALPHA = 16
MAX_STEPS = 2
MAX_SEQ_LENGTH = 128
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def weights_dir() -> Path:
    from sft_loop.report import repo_root

    return repo_root() / "adapters" / "phase3"


def _history_loss(log_history: list[dict[str, Any]]) -> bool:
    for item in log_history:
        loss = item.get("loss") if isinstance(item, dict) else None
        if isinstance(loss, (int, float)) and not isinstance(loss, bool):
            return True
    return False


def execute_train(*, seed: int, proof_dir: Path, adapter_dir: Path) -> dict[str, Any]:
    """Run a short LoRA SFT and return a train_proof object.

    Raises ``RuntimeError`` with an unsupported reason when the stack is
    blocked or the trainer does not finish a step. No task score is computed.
    """

    require_seed(seed)
    blocked = stack_mod.stack_block_reason()
    if blocked is not None:
        raise RuntimeError(blocked)

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    proof_dir.mkdir(parents=True, exist_ok=True)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    rows = []
    for example in train_examples():
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
        rows.append(
            {
                "text": tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                )
            }
        )
    dataset = Dataset.from_list(rows)
    args = SFTConfig(
        output_dir=str(adapter_dir),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()
    global_step = int(getattr(result, "global_step", 0) or 0)
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    if global_step < 1 or not _history_loss(log_history):
        raise RuntimeError("train_not_run")
    train_loss = float(getattr(result, "training_loss"))
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    config_src = adapter_dir / ADAPTER_CONFIG_NAME
    if not config_src.is_file():
        raise RuntimeError("train_not_run")
    config_dst = proof_dir / ADAPTER_CONFIG_NAME
    shutil.copyfile(config_src, config_dst)
    log_text = render_train_log(
        seed=seed,
        global_step=global_step,
        train_loss=train_loss,
        max_steps=MAX_STEPS,
        lora_r=LORA_R,
        log_history=log_history,
    )
    log_path = proof_dir / TRAIN_LOG_NAME
    log_path.write_text(log_text, encoding="utf-8")
    return {
        "trainer": "trl.SFTTrainer",
        "global_step": global_step,
        "train_loss": float(format_loss(train_loss)),
        "lora_r": LORA_R,
        "max_steps": MAX_STEPS,
        "adapter_config_sha256": sha256_file(config_dst),
        "train_log_sha256": sha256_file(log_path),
        "adapter_config_path": ADAPTER_CONFIG_NAME,
        "train_log_path": TRAIN_LOG_NAME,
    }


def build_report(*, seed: int, model_id: str, run_id: str) -> dict[str, Any]:
    """Report for the train CLI. This CLI does not emit task scores.

    When the stack is blocked, the report is ``unsupported``. When the stack
    is ready, a finished step still does not invent before/after numbers; the
    measured pair is produced by ``sft_loop.loop``.
    """

    require_model_id(model_id)
    require_seed(seed)
    blocked = stack_mod.stack_block_reason()
    reason = blocked if blocked is not None else "train_not_run"
    prefix = f"Requested train at seed {seed}. "
    if blocked is None:
        prefix = (
            f"Requested train at seed {seed}. "
            "This command does not emit before/after task scores. "
        )
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
            "Attempt pinned Unsloth + TRL PEFT SFT. "
            "Prints an unsupported run report when the stack cannot train. "
            "Does not print task scores."
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", default="phase3-train")
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        require_model_id(args.model_id)
        require_seed(args.seed)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    blocked = stack_mod.stack_block_reason()
    if blocked is None:
        try:
            proof = execute_train(
                seed=args.seed,
                proof_dir=_proof_dir(),
                adapter_dir=weights_dir(),
            )
        except Exception:
            report = build_report(seed=args.seed, model_id=args.model_id, run_id=args.run_id)
            emit_json(report, args.out)
            return 0
        emit_json({"train_proof": proof, "seed": args.seed, "model_id": MODEL_ID}, args.out)
        return 0
    try:
        report = build_report(seed=args.seed, model_id=args.model_id, run_id=args.run_id)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    emit_json(report, args.out)
    return 0


def _proof_dir() -> Path:
    from sft_loop.report import default_proof_dir

    return default_proof_dir()


if __name__ == "__main__":
    raise SystemExit(main())
