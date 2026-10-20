# LoRA train

Train is TRL `SFTTrainer` plus PEFT LoRA on the pinned instruct model. The scored product path is MPS. This CPU VM cannot label that path. A 2-step CPU smoke exists only as plumbing.

## Sub-features

- `train-cli-blocked` is `python -m sft_loop.train` on a CPU VM.
- `train-mps-flag` is `--device mps` when MPS is missing.
- `train-cpu-smoke` is 2 optimizer steps, LoRA rank 8, seed 0, tiny pinned plumbing model, `device=cpu`.
- `train-proof-files` is `adapter_config.json` plus `train_log.txt` with finite loss, `global_step == max_steps`, and a nonzero saved `lora_B`.

## How to get to it (user POV)

- Run `python -m sft_loop.train --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0`.
- Add `--device mps` on a Mac that has the train extra.
- Add `--device cuda` only with the optional Unsloth extra.
- Run `scripts/verify smoke` on this CPU VM for plumbing.
- Add `--pinned-model` only when smoke should load `Qwen/Qwen2.5-1.5B-Instruct` instead of the tiny plumbing model.

## Driving it with verify-sft-loop

Preconditions:

- Doctor is green.
- Scratch derived from `--run-id` is empty or disposable.
- Evidence dir exists.

- **Official CLI.** Run `scripts/verify unsupported`. `reports/train-default.json` is `unsupported`, reason `gpu_or_train_deps_unavailable`, no `task_score`.
- **MPS flag.** `reports/loop-mps.json` is the same reason and does not set `device`.
- **CPU smoke.** Run `.cursor/skills/verify-sft-loop/scripts/verify smoke --run-id "$VERIFY_RUN_ID"`. If it finishes, `smoke/plumbing.json` has `status` `plumbing`, `device` `cpu`, `backend` `trl-peft`, `stack.unsloth` `unused`, and `weight_change_claimed` false. `train.global_step` equals `train.max_steps`. `lora_b_max_abs` is > 0. `smoke/train_log.txt` contains `trainer=trl.SFTTrainer`, `seed=0`, `global_step=2`, `lora_r=8`, `lora_b_max_abs`, and `device=cpu`. `smoke/adapter_config.json` has `peft_type` `LORA` and `r` 8.
- **Plumbing label.** Read `smoke/plumbing.json`. `kind` is `plumbing_smoke_not_quality`. Hardware lists `cuda` false. No file under `smoke/` validates as a scored report.
- **Fail closed.** If loss is non-finite, `global_step != max_steps`, or every saved `lora_B` is still zero, smoke exits nonzero and `plumbing.json` is `status` `plumbing_failed`. No invented loss.
- **Proof.** Re-run `assert_smoke_not_scored` on the smoke directory. `validate_report` must not accept any file there as `status: scored`.

## Gotchas

- Product CLI `--device` argparse choices are `{mps,cuda}`. `--device cpu` exits 2.
- Smoke is verification scaffolding. It is not a new scored device in the schema.
- Do not copy smoke proof onto `runs/phase3/proof/`. Cap's MPS hashes live there.
- `max_steps` is 2. A low train loss is not evidence the model learned the task. A 0.0/0.0 tiny-model `exact_match` is also not that evidence. The LoRA `lora_B` max-abs is, because `delta W = BA` and B starts at 0.
- Default smoke uses `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` at a pinned revision so the gate does not download ~3 GB. `--pinned-model` is the Qwen 1.5B path.
- Unsloth, bitsandbytes, and flash-attn are unsupported here. `surfaces.json` rows come from live import probes; a probe error is still `unsupported` with the error text.
