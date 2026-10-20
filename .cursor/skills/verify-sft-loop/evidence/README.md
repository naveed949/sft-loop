# Evidence location

Verify writes one directory per `--run-id`:

`.cursor/skills/verify-sft-loop/evidence/<run-id>/`

Cleanup removes `/tmp/sft-loop-verify-<run-id>` only. It never deletes this directory.

Do not export `VERIFY_EVIDENCE_DIR` or `VERIFY_SCRATCH_DIR` unless the path ends with that run id. Stale exports leak across runs. The helper derives both paths from `--run-id` when those env vars are unset or do not match.

Default run id is `cpu-linux-tiny`. That directory is the committed CPU Linux plumbing run of this skill.

Read these first:

- `summary.json` (all steps exit 0, or `failed_step` names the first nonzero step)
- `transcript.txt` (commands, logs, exit codes)
- `surfaces.json` (supported vs unsupported from live probes)
- `smoke/plumbing.json` (CPU smoke is a plumbing check, not a scored report)
- `cleanup.json` (scratch gone, these files still listed)

Smoke must never contain `status: scored` or `weight_change_claimed: true`.
