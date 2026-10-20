# Evidence location

Verify writes one directory per `--run-id`:

`.cursor/skills/verify-sft-loop/evidence/<run-id>/`

Cleanup removes `/tmp/sft-loop-verify-<run-id>` only. It never deletes this directory.

Do not export `VERIFY_EVIDENCE_DIR` or `VERIFY_SCRATCH_DIR` unless the path ends with that run id. Stale exports leak across runs. The helper derives both paths from `--run-id` when those env vars are unset or do not match.

`cpu-linux-tiny-lorab-20261020/` is the committed CPU Linux plumbing run. It came from `scripts/verify all --run-id cpu-linux-tiny-lorab-20261020` on a clean clone of this branch. `verify install` created `.venv` (stdlib venv lacked ensurepip; fallback was `virtualenv`). Default smoke used `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` at revision `ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda`. `--pinned-model` was not passed.

Read these first:

- `summary.json` (all steps exit 0, 72.263s, `failed_step` null)
- `transcript.txt` (commands, logs, exit codes, including the ensurepip miss then virtualenv recreate)
- `surfaces.json` (supported vs unsupported from live probes)
- `smoke/plumbing.json` (CPU smoke is a plumbing check, not a scored report)
- `cleanup.json` (scratch gone, these files still listed)

Smoke plumbing from that run (seed 0, n=8, greedy, float32, eager, 2 LoRA steps, `torch 2.14.1+cpu`, cuda false, mps false): `exact_match` 0.0 before and 0.0 after. `train_loss` 11.931159, `global_step` 2 (`== max_steps`), `lora_b_max_abs` 0.00030013470677658916, `lora_b_nonzero` 8 of 8, `device=cpu`, `backend=trl-peft`, `stack.unsloth=unused`. Those are plumbing checks. The 0.0/0.0 exact match does not prove a step ran. The nonzero `lora_B` does, because PEFT inits B to zero and `delta W = BA`.
