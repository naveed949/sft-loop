# sft-loop

Public MIT scaffold for a same-seed before/after SFT loop on one pinned open instruct model at or under 3B parameters.

Training, evaluation, adapter files, and any weight-change claim are deferred to [issue #3](https://github.com/naveed949/sft-loop/issues/3). This repository checks in the report schema, the stack pin, and fail-closed fixtures.

## Pinned stack

| Component | Pin |
| --- | --- |
| Unsloth | `unsloth==2026.9.12` |
| TRL | `trl==0.24.0` |
| Model | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Method | PEFT LoRA supervised fine-tuning (`sft-lora`) |

`Qwen/Qwen2.5-1.5B-Instruct` is a public Hugging Face instruction-tuned model with 1.54B parameters under Apache-2.0. 1.54B is inside the ≤3B bound.

`trl==0.24.0` is the highest release inside the range Unsloth 2026.9.12 declares: `trl>=0.18.2,!=0.19.0,<=0.24.0`.

## Honesty

- Soft-PASS is unused and is refused. It is not a status and not a gate outcome.
- This scaffold is not an AdaptiveSandbox qualification. Reports have no `adaptiveSandboxQualified` field.
- score ≠ safety. A task score is a task metric. The report field `metric_role` is fixed to `task_score_not_safety`.
- This is not frontier RL.
- This is not full pretraining.
- No invented uplift: this scaffold states no before/after metric values. `weight_change_claimed` is `false` in schema version 1.

## `runs/` path

Canonical schema: [`schema/run_report.schema.json`](schema/run_report.schema.json) (JSON Schema draft 2020-12).

Every object sets `additionalProperties` to `false`. Checked-in portfolio fixtures:

| Path | `status` | Contents |
| --- | --- | --- |
| [`runs/fixtures/unsupported.json`](runs/fixtures/unsupported.json) | `unsupported` | Fail-closed shape for when train cannot run. No `before` or `after`. |
| [`runs/fixtures/unscored.json`](runs/fixtures/unscored.json) | `unscored` | Placeholder with no scores and no weight-change claim. |

`.gitignore` ignores scratch files written directly under `runs/` and un-ignores `runs/fixtures/` so these portfolio files stay tracked. Checkpoints, adapters, and weight files stay ignored.

The schema also describes a future `scored` before/after object (`before` and `after`, with `seed` on the report). This ticket checks in no `scored` file and no `task_score` values. A measured file belongs to issue #3.

## Fail-closed unsupported path

When a GPU or the train dependencies are unavailable, the command below prints one JSON report whose `status` is `unsupported`. Soft-PASS is unused and is refused.

```bash
python -m sft_loop
```

The process does not train. `weight_change_claimed` is false. `before` and `after` are absent. `unsupported.reason` is chosen as follows.

| Condition | `unsupported.reason` |
| --- | --- |
| Torch or CUDA is not usable | `gpu_or_train_deps_unavailable` |
| CUDA is usable, and this package still has no train step | `train_not_run` |

Both reasons are fail-closed. `unsupported` is not a pass.

To copy that JSON onto a scratch path (gitignored, outside `runs/fixtures/`):

```bash
python -m sft_loop --out runs/scratch/unsupported.json
```

## Reproduce

Commands that run in this scaffold, with no GPU:

```bash
python -m pip install -e ".[dev]"
pytest
python -m sft_loop
```

Stubs that will apply once train lands in issue #3. They are not implemented here. They do not report an uplift. `sft_loop.train` and `sft_loop.eval` are not modules in this ticket.

```bash
python -m pip install "unsloth==2026.9.12" "trl==0.24.0"
python -m sft_loop.train --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0
python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split before
python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split after
```

## License

MIT. See [LICENSE](LICENSE).
