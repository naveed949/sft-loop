---
name: verify-sft-loop
description: Use when proving an sft-loop PR, driving python -m sft_loop, checking train/eval/loop honesty, or validating schema/run_report.schema.json. Primary surface is the sft-loop CLI on a CPU Linux VM.
---

# Verify sft-loop

Drive this repo the way a user does. The user-facing surface is the CLI (`python -m sft_loop`, `.train`, `.eval`, `.loop`). There is no web UI. Proof is JSON reports, pytest output, train logs, and exit codes. Run the scripts in `scripts/`. Do not retype the pipeline.

Read `features/README.md` and drive every mapped feature that the change can touch. One convenient entry point is not enough when the map lists others.

This skill is for a CPU Linux VM. Never provision a GPU. Never install the `unsloth` extra. Never invent `before`/`after`/`task_score`. A surface that cannot run here is `unsupported` with `gpu_or_train_deps_unavailable` or `train_not_run`. Those are the only reasons in `schema/run_report.schema.json`. Never record a skip, estimate, or pass for a blocked GPU path.

Soft-PASS is unused and is refused.

Keep the map current with `/maintain-verification-skill`.

## Launch

There is no long-lived server. Launch is `scripts/verify install`. Do **not** run `python3 -m venv .venv` by hand. On Debian images without `python3-venv` that command exits 1 and leaves `.venv/bin/python` with no pip.

`verify install` owns env creation:

- If `.venv` is missing, create it.
- If `.venv` exists but `python -m pip --version` fails, delete it and recreate.
- Try stdlib `venv` first. If pip is still missing, fall back to `python3 -m pip install --user virtualenv` then `python3 -m virtualenv .venv`.
- If `.venv` already has a working pip, leave it and only install packages. That is the idempotent path.

```bash
.cursor/skills/verify-sft-loop/scripts/verify install --run-id "$VERIFY_RUN_ID"
.cursor/skills/verify-sft-loop/scripts/verify doctor --run-id "$VERIFY_RUN_ID"
```

Default `VERIFY_RUN_ID` is `cpu-linux-tiny`. Pass `--run-id` on every invocation. Ready means `doctor` exits 0, `.venv/bin/python -m pip --version` works, `schema/run_report.schema.json` exists, and `doctor.json` shows `cuda=false` and `mps=false`.

Install uses the CPU PyTorch wheel index, then the pins in `pyproject.toml` (`jsonschema`, `pytest`, `transformers`, `accelerate`, `datasets`, `peft`, `trl==0.24.0`) and `pip install -e .`. It does not install `[unsloth]`.

Do not export `VERIFY_EVIDENCE_DIR` or `VERIFY_SCRATCH_DIR` unless this invocation's path ends with the same `--run-id` (or `sft-loop-verify-<run-id>` for scratch). Stale exports leak into the next run. When those env vars are unset or do not match, the helper derives:

- evidence: `.cursor/skills/verify-sft-loop/evidence/<run-id>/`
- scratch: `/tmp/sft-loop-verify-<run-id>`

Teardown is `scripts/verify cleanup --run-id "$VERIFY_RUN_ID"`. Cleanup deletes `/tmp/sft-loop-verify-$VERIFY_RUN_ID` only. It does not delete `.venv`, `hf_cache/` (gitignored Hub cache), or evidence. Honesty tests skip `hf_cache/` so empty Hub sentinel JSON cannot fail pytest.

## Doctor

Read-only health for "is this instance worth driving?"

```bash
.cursor/skills/verify-sft-loop/scripts/verify doctor --run-id "$VERIFY_RUN_ID"
```

Require exit 0 and `evidence/$VERIFY_RUN_ID/doctor.json` with:

- Python >= 3.10
- schema file present
- `jsonschema`, `pytest`, and `sft_loop` importable
- venv pip working
- `cuda` false, `mps` false

If doctor fails, stop. Do not drive train or eval. If CUDA or MPS is suddenly true, still do not label a CPU smoke as `device: mps` or `device: cuda`. The schema only allows those labels on a real scored path.

Run doctor again after a failed drive and before a fresh shell.

## Drive

Prefer `scripts/verify`. Commands below are literal. Pass `--run-id` every time. Do not rely on leftover `VERIFY_EVIDENCE_DIR` / `VERIFY_SCRATCH_DIR` from a previous run.

```bash
export VERIFY_RUN_ID=cpu-linux-tiny
```

Two official CLI processes can run at once. They must not share scratch. Never write verify output to `runs/phase3/`. That directory is the Cap M1 portfolio.

**Full proof (one shot).**

```bash
.cursor/skills/verify-sft-loop/scripts/verify all --run-id "$VERIFY_RUN_ID"
```

That runs install, doctor, pytest, official fail-closed CLIs, CPU plumbing smoke, then cleanup. A nonzero exit from any named step fails the whole run. Cleanup still runs. `summary.json` records `failed_step`.

**Tests.**

```bash
.cursor/skills/verify-sft-loop/scripts/verify test --run-id "$VERIFY_RUN_ID"
```

**Official CLI (fail-closed on this VM).**

```bash
.cursor/skills/verify-sft-loop/scripts/verify unsupported --run-id "$VERIFY_RUN_ID"
```

This is the real user path for MPS/CUDA on a CPU machine. Expect `status: unsupported`, `weight_change_claimed: false`, no `before`/`after`, no `device`. Validate each file with `sft_loop.report.validate_report`.

Equivalent raw commands (same reports):

```bash
.venv/bin/python -m sft_loop --run-id verify-cli-default --out ".cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID/reports/cli-default.json"
.venv/bin/python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --out ".cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID/reports/loop-default.json"
.venv/bin/python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --device mps --out ".cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID/reports/loop-mps.json"
.venv/bin/python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --device cuda --out ".cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID/reports/loop-cuda.json"
.venv/bin/python -m sft_loop.train --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --out ".cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID/reports/train-default.json"
.venv/bin/python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split before --out ".cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID/reports/eval-before.json"
.venv/bin/python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split after --out ".cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID/reports/eval-after.json"
```

`--device cpu` is not a product choice. argparse `choices` are `{mps,cuda}`; the CLI exits 2. `resolve_train_target("cpu")` raises `ValueError`. Do not add CPU to the product CLI.

**CPU plumbing smoke (not a quality result, not a scored report).**

```bash
.cursor/skills/verify-sft-loop/scripts/verify smoke --run-id "$VERIFY_RUN_ID"
```

Default model is the tiny pinned plumbing model `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` at revision `ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda`. Pass `--pinned-model` only when you intentionally want `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB Hub download).

Smoke writes `smoke/plumbing.json` with `status: plumbing` (or `plumbing_failed`), `weight_change_claimed: false`, `device: cpu`, `backend: trl-peft`, and `stack.unsloth: unused`. It must never write `status: scored` or `weight_change_claimed: true`. No file under `smoke/` may validate as a scored run report.

Smoke fails closed unless all of these hold after `trainer.train()`: `train_loss` is finite, `global_step == max_steps` (requested steps, 2), and at least one saved LoRA `lora_B` tensor has nonzero max-abs. PEFT inits B to zero, so `delta W = B A` is zero at step 0 and the base model's outputs are unchanged. Record `lora_b_max_abs` and `global_step` in `plumbing.json`. A 0.0/0.0 tiny-model `exact_match` cannot prove a step ran.

If smoke cannot finish, keep the plumbing artifact and fail the step. Do not estimate scores. Do not mint a schema-scored report.

Feature recipes: `features/cli-unsupported-report.md`, `features/train-lora.md`, `features/eval-same-seed.md`, `features/loop.md`, `features/report-schema.md`.

## Evidence

Named location that must survive cleanup:

`.cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID/`

Capture all of:

- `transcript.txt` (commands, stdout, stderr, exit codes)
- `doctor.json`
- `pytest.txt`
- `reports/*.json` (official CLI, each schema-valid)
- `unsupported-validation.json`
- `smoke/plumbing.json` plus `smoke/adapter_config.json` and `smoke/train_log.txt` when train finished
- `surfaces.json` from live probes (`torch.cuda` / `torch.backends.mps`, import attempts for bitsandbytes / flash_attn / unsloth). A probe error is still `unsupported` with the error text
- `summary.json`
- `cleanup.json` after teardown

Proof standards:

- Drive the CLI or the shipped helper. Do not hand-edit JSON scores.
- Record the action and the resulting file. A passing pytest line is not proof of train.
- Side effects: adapter files live in scratch and may vanish on cleanup. `adapter_config.json` and `train_log.txt` copied into evidence must remain.
- Mocks only exist in `tests/`. Verification does not use them for smoke.
- The official dry path is not a dry-run flag. `python -m sft_loop.train` on this VM prints `unsupported` and does not call `trainer.train`. Confirm by missing `runs/phase3/proof` mutations and by `status: unsupported`.
- CPU smoke numbers cite config, seed, split, and hardware from `plumbing.json`. They are not Cap M1 quality numbers and they are not a scored report.

Never copy verify smoke JSON onto `runs/phase3/scored.json`.

## Cleanup

```bash
.cursor/skills/verify-sft-loop/scripts/verify cleanup --run-id "$VERIFY_RUN_ID"
```

Deletes only `/tmp/sft-loop-verify-$VERIFY_RUN_ID` (adapters and other scratch). Does not kill by process name. Does not delete evidence, `.venv`, or `hf_cache/`. After cleanup, `cleanup.json` must list leftover evidence files. If that list is empty, cleanup failed this skill.

Run cleanup after every failed iteration too.

## Helpers

All invocations from the repo root. Scripts are executable.

| Command | What it does |
| --- | --- |
| `scripts/verify install` | Create or repair `.venv`, CPU torch wheel, extras from `pyproject.toml` |
| `scripts/verify doctor` | Read-only readiness |
| `scripts/verify test` | `python -m pytest -q` |
| `scripts/verify unsupported` | Official CLIs, schema-validate fail-closed reports |
| `scripts/verify smoke` | CPU LoRA plumbing on the tiny pinned model; writes `plumbing.json`, never a scored report |
| `scripts/verify cleanup` | Remove scratch, keep evidence |
| `scripts/verify all` | The sequence above; any nonzero step fails the run and names that step |

Implementation: `scripts/verify_sft_loop.py`.
