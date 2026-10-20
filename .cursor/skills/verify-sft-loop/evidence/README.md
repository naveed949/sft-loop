# Evidence location

Verify writes one directory per `VERIFY_RUN_ID`:

`.cursor/skills/verify-sft-loop/evidence/<run-id>/`

Cleanup removes `/tmp/sft-loop-verify-<run-id>` only. It never deletes this directory.

`cpu-linux-seed0/` is the first end-to-end run of this skill on a CPU Linux VM.

Read these first:

- `summary.json` (all steps exit 0)
- `transcript.txt` (commands, logs, exit codes, including earlier failed iterations)
- `surfaces.json` (supported vs unsupported)
- `smoke/plumbing.json` (CPU smoke is a plumbing check, not a quality result)
- `smoke/run_report.json` (schema-valid scored report with no `device` label)
- `cleanup.json` (scratch gone, these files still listed)

Smoke metrics from that run (seed 0, n=8, greedy, float32, eager, 2 LoRA steps, `torch 2.14.1+cpu`, cuda false, mps false): `exact_match` 0.375 before and 0.375 after. Those are plumbing checks. They are not Cap M1 quality numbers.
