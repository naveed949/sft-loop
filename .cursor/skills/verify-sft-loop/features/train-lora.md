# LoRA train

Train is TRL `SFTTrainer` plus PEFT LoRA on the pinned instruct model. The scored product path is MPS. This CPU VM cannot label that path. A 2-step CPU smoke exists only as plumbing.

## Sub-features

- `train-cli-blocked` is `python -m sft_loop.train` on a CPU VM.
- `train-mps-flag` is `--device mps` when MPS is missing.
- `train-cpu-smoke` is 2 optimizer steps, LoRA rank 8, seed 0, pinned model, `device=cpu`.
- `train-proof-files` is `adapter_config.json` plus `train_log.txt` with `global_step >= 1` and a finite loss.

## How to get to it (user POV)

- Run `python -m sft_loop.train --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0`.
- Add `--device mps` on a Mac that has the train extra.
- Add `--device cuda` only with the optional Unsloth extra.
- Run `scripts/verify smoke` on this CPU VM for plumbing.

## Driving it with verify-sft-loop

Preconditions:

- Doctor is green.
- `VERIFY_SCRATCH_DIR` is empty or disposable.
- Evidence dir exists.

- **Official CLI.** Run `scripts/verify unsupported`. `reports/train-default.json` is `unsupported`, reason `gpu_or_train_deps_unavailable`, no `task_score`.
- **MPS flag.** `reports/loop-mps.json` is the same reason and does not set `device`.
- **CPU smoke.** Run `.cursor/skills/verify-sft-loop/scripts/verify smoke --run-id "$VERIFY_RUN_ID"`. If it finishes, `smoke/train_log.txt` contains `trainer=trl.SFTTrainer`, `seed=0`, `global_step=2`, `lora_r=8`, and `model_id=Qwen/Qwen2.5-1.5B-Instruct`. `smoke/adapter_config.json` has `peft_type` `LORA` and `r` 8.
- **Plumbing label.** Read `smoke/plumbing.json`. `kind` is `plumbing_smoke_not_quality`. Hardware lists `cuda` false. Metrics in `smoke/run_report.json` are not quality results.
- **Fail closed.** If smoke cannot load or step, `smoke/run_report.json` is `unsupported` with `train_not_run`. No invented loss.
- **Proof.** `smoke/report-validation.json` has `"ok": true` when smoke scored, or the unsupported report validates.

## Gotchas

- Product CLI `--device` accepts only `mps` and `cuda`. `--device cpu` raises `ValueError`.
- Smoke calls `_execute_train_trl_peft(..., device="cpu")`. That is verification scaffolding. It is not a new scored device in the schema.
- Do not copy smoke proof onto `runs/phase3/proof/`. Cap's MPS hashes live there.
- `max_steps` is 2. A low train loss is not evidence the model learned the task.
- Unsloth, bitsandbytes, and flash-attn are unsupported here. Record them in `surfaces.json` with `gpu_or_train_deps_unavailable`.
