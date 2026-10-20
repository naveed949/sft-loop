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

There is no long-lived server. Launch means a venv plus the repo extras from `pyproject.toml`, CPU torch only.

```bash
python3 -m venv .venv
.cursor/skills/verify-sft-loop/scripts/verify install --run-id "$VERIFY_RUN_ID"
.cursor/skills/verify-sft-loop/scripts/verify doctor --run-id "$VERIFY_RUN_ID"
```

Default `VERIFY_RUN_ID` is `cpu-linux-seed0`. Ready means `doctor` exits 0, `.venv/bin/python` exists, `schema/run_report.schema.json` exists, and `doctor.json` shows `cuda=false` and `mps=false`.

Install uses the CPU PyTorch wheel index, then the pins in `pyproject.toml` (`jsonschema`, `pytest`, `transformers`, `accelerate`, `datasets`, `peft`, `trl==0.24.0`) and `pip install -e .`. It does not install `[unsloth]`. If `python3 -m venv` lacks `ensurepip` (Debian images without `python3-venv`), install falls back to `python3 -m pip install --user virtualenv` and `python3 -m virtualenv .venv`.

Teardown is `scripts/verify cleanup --run-id "$VERIFY_RUN_ID"`. Cleanup deletes `/tmp/sft-loop-verify-$VERIFY_RUN_ID` only. It does not delete `.venv`, `hf_cache/`, or evidence.

## Doctor

Read-only health for "is this instance worth driving?"

```bash
.cursor/skills/verify-sft-loop/scripts/verify doctor --run-id "$VERIFY_RUN_ID"
```

Require exit 0 and `evidence/$VERIFY_RUN_ID/doctor.json` with:

- Python >= 3.10
- schema file present
- `jsonschema`, `pytest`, and `sft_loop` importable
- `cuda` false, `mps` false

If doctor fails, stop. Do not drive train or eval. If CUDA or MPS is suddenly true, still do not label a CPU smoke as `device: mps` or `device: cuda`. The schema only allows those labels on a real scored path.

Run doctor again after a failed drive and before a fresh shell.

## Drive

Prefer `scripts/verify`. Commands below are literal.

Set an isolated run id when two agents might overlap:

```bash
export VERIFY_RUN_ID=cpu-linux-seed0
export VERIFY_EVIDENCE_DIR=.cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID
export VERIFY_SCRATCH_DIR=/tmp/sft-loop-verify-$VERIFY_RUN_ID
```

Two official CLI processes can run at once. They must not share `VERIFY_SCRATCH_DIR`. Never write verify output to `runs/phase3/`. That directory is the Cap M1 portfolio.

**Full proof (one shot).**

```bash
.cursor/skills/verify-sft-loop/scripts/verify all --run-id "$VERIFY_RUN_ID"
```

That runs install, doctor, pytest, official fail-closed CLIs, CPU smoke train+eval+schema validation, then cleanup.

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
.venv/bin/python -m sft_loop --run-id verify-cli-default --out "$VERIFY_EVIDENCE_DIR/reports/cli-default.json"
.venv/bin/python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --out "$VERIFY_EVIDENCE_DIR/reports/loop-default.json"
.venv/bin/python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --device mps --out "$VERIFY_EVIDENCE_DIR/reports/loop-mps.json"
.venv/bin/python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --device cuda --out "$VERIFY_EVIDENCE_DIR/reports/loop-cuda.json"
.venv/bin/python -m sft_loop.train --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --out "$VERIFY_EVIDENCE_DIR/reports/train-default.json"
.venv/bin/python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split before --out "$VERIFY_EVIDENCE_DIR/reports/eval-before.json"
.venv/bin/python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split after --out "$VERIFY_EVIDENCE_DIR/reports/eval-after.json"
```

`--device cpu` is rejected by `sft_loop.stack.resolve_train_target`. Do not add it to the product CLI.

**CPU plumbing smoke (not a quality result).**

```bash
.cursor/skills/verify-sft-loop/scripts/verify smoke --run-id "$VERIFY_RUN_ID"
```

This calls `_execute_eval_trl_peft` (before, no adapter), `_execute_train_trl_peft`, then `_execute_eval_trl_peft` (after, with adapter), all with `device="cpu"`, seed `0`, pinned model `Qwen/Qwen2.5-1.5B-Instruct`, `max_steps=2`, LoRA rank 8, float32, eager attention. Baseline and candidate share that eval setup. The helper then calls `build_scored_report` **without** `device`/`backend` so the report cannot claim MPS or CUDA. Sidecar `smoke/plumbing.json` labels the metrics as plumbing.

If smoke cannot finish, write a schema-valid `unsupported` report with `train_not_run`. Do not estimate scores.

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
- `smoke/run_report.json` plus `smoke/plumbing.json`, `smoke/adapter_config.json`, `smoke/train_log.txt`
- `surfaces.json` (`supported` vs `unsupported`, never estimated)
- `summary.json`
- `cleanup.json` after teardown

Proof standards:

- Drive the CLI or the shipped helper. Do not hand-edit JSON scores.
- Record the action and the resulting file. A passing pytest line is not proof of train.
- Side effects: adapter files live in scratch and may vanish on cleanup. `adapter_config.json` and `train_log.txt` copied into evidence must remain.
- Mocks only exist in `tests/`. Verification does not use them for smoke.
- The official dry path is not a dry-run flag. `python -m sft_loop.train` on this VM prints `unsupported` and does not call `trainer.train`. Confirm by missing `runs/phase3/proof` mutations and by `status: unsupported`.
- CPU smoke numbers cite config, seed, split, and hardware from `plumbing.json`. They are not Cap M1 quality numbers.

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
| `scripts/verify install` | CPU venv, CPU torch wheel, extras from `pyproject.toml` |
| `scripts/verify doctor` | Read-only readiness |
| `scripts/verify test` | `python -m pytest -q` |
| `scripts/verify unsupported` | Official CLIs, schema-validate fail-closed reports |
| `scripts/verify smoke` | CPU LoRA SFT + same-seed eval + schema validate |
| `scripts/verify cleanup` | Remove scratch, keep evidence |
| `scripts/verify all` | The sequence above |

Implementation: `scripts/verify_sft_loop.py`.
