# sft-loop

Public MIT loop for same-seed before/after PEFT supervised fine-tuning on one pinned open instruct model at or under 3B parameters.

The checked-in Phase 3 result is fail-closed `unsupported`. This environment has no CUDA device and does not have the pinned Unsloth/TRL packages installed, so no train step ran and no before/after task scores exist.

## Pinned stack

| Component | Pin |
| --- | --- |
| Unsloth | `unsloth==2026.9.12` |
| TRL | `trl==0.24.0` |
| Model | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Method | PEFT LoRA supervised fine-tuning (`sft-lora`) |

`Qwen/Qwen2.5-1.5B-Instruct` is a public Hugging Face instruction-tuned model with 1.54B parameters under Apache-2.0. 1.54B is inside the ≤3B bound.

`trl==0.24.0` is the highest release inside the range Unsloth 2026.9.12 declares: `trl>=0.18.2,!=0.19.0,<=0.24.0`.

The optional extra `train` in `pyproject.toml` installs those two pins. It was not installed for the checked-in run.

## Honesty

- Soft-PASS is unused and is refused. It is not a status and not a gate outcome.
- This repository is not an AdaptiveSandbox qualification. Reports have no `adaptiveSandboxQualified` field.
- score ≠ safety. A task score is a task metric. The report field `metric_role` is fixed to `task_score_not_safety`.
- This is not frontier RL.
- This is not full pretraining.
- No invented uplift: the checked-in Phase 3 report has no `before`, no `after`, and no `task_score`. `weight_change_claimed` is false.
- `weight_change_claimed` can be true only on a `scored` report whose `train_proof` matches a LoRA `adapter_config.json` and a TRL `SFTTrainer` log with `global_step >= 1`. That proof is not in this checkout.

## Task

Synthetic held-out task `uppercase-word-v1`. The grader is exact match on the first token of the generation (`exact_match`). Train words and eval words are disjoint. The metric is a task score, not a safety judgment.

## `runs/` path

Canonical schema: [`schema/run_report.schema.json`](schema/run_report.schema.json) (JSON Schema draft 2020-12).

Every object sets `additionalProperties` to `false`. Status is `unsupported`, `unscored`, or `scored`.

| Path | `status` | Contents |
| --- | --- | --- |
| [`runs/fixtures/unsupported.json`](runs/fixtures/unsupported.json) | `unsupported` | Scaffold fixture for the blocked shape. No `before` or `after`. |
| [`runs/fixtures/unscored.json`](runs/fixtures/unscored.json) | `unscored` | Placeholder with no scores and no weight-change claim. |
| [`runs/phase3/unsupported.json`](runs/phase3/unsupported.json) | `unsupported` | Live Phase 3 loop result for seed 0. Same seed was requested for both eval splits. Neither split ran. |

`.gitignore` ignores scratch files under `runs/` and un-ignores the fixture directory plus `runs/phase3/unsupported.json`. A future scored run may also check in `runs/phase3/proof/adapter_config.json` and `runs/phase3/proof/train_log.txt`. Adapter weights (`*.safetensors` and the `adapters/` directory) stay ignored.

## Fail-closed unsupported path

When a GPU or the train dependencies are unavailable, each command below prints one JSON report whose `status` is `unsupported`. Soft-PASS is unused and is refused.

```bash
python -m sft_loop
python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0
python -m sft_loop.train --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0
python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split before
python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split after
```

`sft_loop.train` and `sft_loop.eval` do not print a task score on this path. `weight_change_claimed` is false. `before` and `after` are absent. `python -m sft_loop.loop` is the command that wrote `runs/phase3/unsupported.json`.

| Condition | `unsupported.reason` |
| --- | --- |
| Torch, CUDA, Unsloth, or TRL is not usable | `gpu_or_train_deps_unavailable` |
| The pinned stack is usable and this invocation did not finish a train step | `train_not_run` |

Both reasons are fail-closed. `unsupported` is not a pass.

To copy the probe JSON onto a scratch path (gitignored, outside the tracked carve-outs):

```bash
python -m sft_loop --out runs/scratch/unsupported.json
```

## Reproduce

Commands that ran for the checked-in artifact, with no GPU:

```bash
python -m pip install -e ".[dev]"
pytest
python -m sft_loop
python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --out runs/phase3/unsupported.json
python -m sft_loop.train --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0
python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split before
python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split after
```

On a CUDA machine the same loop command attempts a 2-step Unsloth + TRL LoRA SFT (`sft-lora`) on `Qwen/Qwen2.5-1.5B-Instruct`, then evals the held-out words before training and again after training with the same `--seed`. Install the pinned extra first. This prints a report to stdout. It was not run here, and it is not the source of `runs/phase3/unsupported.json`.

```bash
python -m pip install -e ".[dev,train]"
python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0
```

If that command cannot finish a real step, it still prints `unsupported` and does not fill in before/after numbers. A `scored` report is eligible to be checked in only together with `runs/phase3/proof/adapter_config.json` and `runs/phase3/proof/train_log.txt`.

## License

MIT. See [LICENSE](LICENSE).
