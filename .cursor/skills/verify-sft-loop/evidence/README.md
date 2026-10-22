# Evidence location

Verify writes one directory per `--run-id`:

`.cursor/skills/verify-sft-loop/evidence/<run-id>/`

Cleanup removes `/tmp/sft-loop-verify-<run-id>` only. It never deletes this directory.

Do not export `VERIFY_EVIDENCE_DIR` or `VERIFY_SCRATCH_DIR` unless the path ends with that run id. Stale exports leak across runs. The helper derives both paths from `--run-id` when those env vars are unset or do not match.

`cpu-linux-tiny-lorab-20261020/` is the committed CPU Linux plumbing run. It came from `scripts/verify all --run-id cpu-linux-tiny-lorab-20261020` on a clean clone of this branch. `verify install` created `.venv` (stdlib venv lacked ensurepip; fallback was `virtualenv`). Default smoke used `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` at revision `ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda`. `--pinned-model` was not passed.

`issue9-quant/` is the later `scripts/verify all --run-id issue9-quant` run on the quantization branch. Every named step exited 0, including `quant`. That step probed NF4, validated `runs/quant/`, and checked the README table. It did not reload the 1.5B weights.

`issue9-quant-r2/` is the remasurement proof after the throughput and NLL blockers. `scripts/verify all --run-id issue9-quant-r2` exited 0. The table now has scored p50/p95, mean generated tokens, and decode tokens/s from a `min_new_tokens == max_new_tokens == 8` pass.

Kaggle GPU kernels are not a local verify surface. The scripts live at `kaggle/quant_nf4` and `kaggle/quant_onnx_int8`. `verify quant` checks those directories exist and that ingest of a missing output dir is `kernel_output_missing`. It does not run the kernels.

`issue9-quant-gpu-p1/` is the Phase 1 proof. `scripts/verify all --run-id issue9-quant-gpu-p1` exited 0. `surfaces.json` lists `kaggle-quant-kernels` as unsupported and points at those dirs. `quant/kernel-missing.json` is the fail-closed ingest.

`issue9-quant-gpu-p2/` is the Phase 2 ingest proof. `scripts/verify all --run-id issue9-quant-gpu-p2` exited 0 in 34.508s. Pytest was 81 passed. `quant` validated the ingested T4 comparison, checked the README table, and still fail-closed a missing kernel dir. Downloaded outputs live at `runs/quant/kernel/`. Local verify still does not run the kernels.

`issue9-quant-t4-fp32/` is the fail-closed T4 fp32 baseline proof. `scripts/verify all --run-id issue9-quant-t4-fp32` exited 0. v1 GPU pairing is withdrawn. The replacement kernel lives at `kaggle/quant_nf4`.

Read these first:

- `summary.json` (all steps exit 0, 72.263s, `failed_step` null)
- `transcript.txt` (commands, logs, exit codes, including the ensurepip miss then virtualenv recreate)
- `surfaces.json` (supported vs unsupported from live probes)
- `smoke/plumbing.json` (CPU smoke is a plumbing check, not a scored report)
- `cleanup.json` (scratch gone, these files still listed)

Smoke plumbing from that run (seed 0, n=8, greedy, float32, eager, 2 LoRA steps, `torch 2.14.1+cpu`, cuda false, mps false): `exact_match` 0.0 before and 0.0 after. `train_loss` 11.931159, `global_step` 2 (`== max_steps`), `lora_b_max_abs` 0.00030013470677658916, `lora_b_nonzero` 8 of 8, `device=cpu`, `backend=trl-peft`, `stack.unsloth=unused`. Those are plumbing checks. The 0.0/0.0 exact match does not prove a step ran. The nonzero `lora_B` does, because PEFT inits B to zero and `delta W = BA`.
