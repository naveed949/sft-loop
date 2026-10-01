# sft-loop

Public MIT loop for same-seed before/after PEFT supervised fine-tuning on one pinned open instruct model at or under 3B parameters.

The scored train path is **TRL + PEFT LoRA SFT** on Cap **M1 Pro 32GB (MPS)**. It does not use Unsloth. Unsloth stays an optional **CUDA-only** extra and is not required for `weight_change_claimed`.

The checked-in Phase 3 result in this checkout is fail-closed `unsupported`. This environment has no MPS device and does not have the TRL + PEFT train extra installed, so no train step ran and no before/after task scores exist. Scored `runs/` artifacts are not invented here. Cap runs the train on the Mac after this path lands and checks in the scored report plus proof.

## Pinned stack

| Component | Pin | Role |
| --- | --- | --- |
| TRL | `trl==0.24.0` | Scored-path trainer (`trl.SFTTrainer`) |
| PEFT | `peft>=0.8.0` | LoRA on the MPS scored path (TRL 0.24 extra floor) |
| Transformers | `transformers>=4.56.1` | Model load on the MPS scored path |
| Model | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Pinned instruct model |
| Method | PEFT LoRA supervised fine-tuning (`sft-lora`) | Rank 8, 2 optimizer steps |
| Device | `mps` on Cap M1 Pro 32GB | Report field when that path scores |
| Unsloth | `unsloth==2026.9.12` | Optional CUDA-only. Not used on MPS. Not required for `weight_change_claimed`. |

`Qwen/Qwen2.5-1.5B-Instruct` is a public Hugging Face instruction-tuned model with 1.54B parameters under Apache-2.0. 1.54B is inside the ≤3B bound.

`trl==0.24.0` requires `transformers>=4.56.1`, `datasets>=3.0.0`, and `accelerate>=1.4.0`. Those floors are the `train` extra in `pyproject.toml`, together with `torch>=2.2` and `peft>=0.8.0`. The MPS path loads the model in float32 with eager attention. It does not use 4-bit quantization.

`trl==0.24.0` is the highest release inside the range Unsloth 2026.9.12 declares: `trl>=0.18.2,!=0.19.0,<=0.24.0`. That pin is the optional CUDA package. It is not installed for the MPS scored path.

## Honesty

- Soft-PASS is unused and is refused. It is not a status and not a gate outcome.
- This repository is not an AdaptiveSandbox qualification. Reports have no `adaptiveSandboxQualified` field.
- score ≠ safety. A task score is a task metric. The report field `metric_role` is fixed to `task_score_not_safety`.
- This is not frontier RL.
- This is not full pretraining.
- No invented uplift: the checked-in Phase 3 report has no `before`, no `after`, and no `task_score`. `weight_change_claimed` is false. No scored before/after JSON is checked in by this change.
- `weight_change_claimed` can be true only on a `scored` report whose `train_proof` matches a checked-in LoRA `runs/phase3/proof/adapter_config.json` and a TRL `SFTTrainer` log `runs/phase3/proof/train_log.txt` with `global_step >= 1`. Unsloth is not required for that claim. On MPS the report labels `device` `mps`, `backend` `trl-peft`, and `stack.unsloth` is `unused`.

## Task

Synthetic held-out task `uppercase-word-v1`. The grader is exact match on the first token of the generation (`exact_match`). Train words and eval words are disjoint. The metric is a task score, not a safety judgment.

## `runs/` path

Canonical schema: [`schema/run_report.schema.json`](schema/run_report.schema.json) (JSON Schema draft 2020-12).

Every object sets `additionalProperties` to `false`. Status is `unsupported`, `unscored`, or `scored`.

| Path | `status` | Contents |
| --- | --- | --- |
| [`runs/fixtures/unsupported.json`](runs/fixtures/unsupported.json) | `unsupported` | Scaffold fixture for the blocked shape. No `before` or `after`. |
| [`runs/fixtures/unscored.json`](runs/fixtures/unscored.json) | `unscored` | Placeholder with no scores and no weight-change claim. |
| [`runs/phase3/unsupported.json`](runs/phase3/unsupported.json) | `unsupported` | Live Phase 3 loop result for seed 0 on a machine with no MPS and no CUDA Unsloth stack. Same seed was requested for both eval splits. Neither split ran. |
| `runs/phase3/scored.json` | `scored` | Not in this checkout. Cap writes it on the M1 after a real train. Gitignore allows that filename. |
| `runs/phase3/proof/adapter_config.json` | proof | Not in this checkout. Copied from the PEFT adapter after `global_step >= 1`. |
| `runs/phase3/proof/train_log.txt` | proof | Not in this checkout. TRL log with `device=mps` and `backend=trl-peft` when the MPS path ran. |

`.gitignore` ignores scratch files under `runs/` and un-ignores the fixture directory, `runs/phase3/unsupported.json`, `runs/phase3/scored.json`, and the two proof filenames. Adapter weights (`*.safetensors` and the `adapters/` directory) stay ignored.

## Fail-closed unsupported path

When the requested device or the train dependencies are unavailable, each command below prints one JSON report whose `status` is `unsupported`. The report does not set `device`, so it does not claim MPS or CUDA. Soft-PASS is unused and is refused.

```bash
python -m sft_loop
python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0
python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --device mps
python -m sft_loop.train --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0
python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split before
python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split after
```

`sft_loop.train` and `sft_loop.eval` do not print a task score on this path. `weight_change_claimed` is false. `before` and `after` are absent. `python -m sft_loop.loop` without `--device` is the command that wrote `runs/phase3/unsupported.json`.

| Condition | `unsupported.reason` |
| --- | --- |
| MPS is unavailable, or torch / transformers / PEFT / TRL / datasets / accelerate is not usable, and the optional CUDA Unsloth stack is also unusable | `gpu_or_train_deps_unavailable` |
| The requested device is `mps` and MPS or the TRL + PEFT libraries are not usable. Missing Unsloth does not by itself block this path | `gpu_or_train_deps_unavailable` |
| The requested device is `cuda` and CUDA or Unsloth or TRL is not usable | `gpu_or_train_deps_unavailable` |
| The pinned stack is usable and this invocation did not finish a train step | `train_not_run` |

Both reasons are fail-closed. `unsupported` is not a pass.

To copy the probe JSON onto a scratch path (gitignored, outside the tracked carve-outs):

```bash
python -m sft_loop --out runs/scratch/unsupported.json
```

## Reproduce on this checkout (no MPS)

Commands that ran for the checked-in unsupported artifact:

```bash
python -m pip install -e ".[dev]"
pytest
python -m sft_loop
python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --out runs/phase3/unsupported.json
python -m sft_loop.train --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0
python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split before
python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split after
```

## Cap M1 Pro 32GB (MPS) scored path

Cap runs this on the Mac. It is TRL + PEFT LoRA SFT without Unsloth. Do not install the `unsloth` extra for this path. The loop evals the held-out words before training and again after training with the same `--seed`. A finished step writes proof under `runs/phase3/proof/` (`adapter_config.json` and `train_log.txt` with `global_step >= 1`). The scored report is stdout and, with `--out`, `runs/phase3/scored.json`. Adapter weights land in `adapters/phase3/` and stay gitignored.

```bash
python -m pip install -e ".[dev,train]"
python -m sft_loop.loop \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --seed 0 \
  --device mps \
  --run-id phase3-mps-seed0 \
  --out runs/phase3/scored.json
```

Check the report before committing it:

- `status` is `scored`
- `device` is `mps`
- `backend` is `trl-peft`
- `stack.unsloth` is `unused`
- `weight_change_claimed` is true only together with the two proof files
- `before.seed` and `after.seed` equal `seed`
- `metric_role` is `task_score_not_safety`

If the command cannot finish a real step, it still prints `unsupported`, omits `device`, and does not fill in before/after numbers.

Files Cap checks in after that run:

- `runs/phase3/scored.json`
- `runs/phase3/proof/adapter_config.json`
- `runs/phase3/proof/train_log.txt`

## Optional Unsloth (CUDA-only)

Unsloth is not the MPS scored path and is not required for `weight_change_claimed`. On a CUDA machine the loop can instead run a 2-step Unsloth + TRL LoRA SFT. The report then labels `device` `cuda` and `backend` `unsloth`. This was not run for the checked-in unsupported report.

```bash
python -m pip install -e ".[dev,train,unsloth]"
python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --device cuda
```

## License

MIT. See [LICENSE](LICENSE).
