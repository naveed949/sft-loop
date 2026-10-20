# Evidence location

Verify writes one directory per `VERIFY_RUN_ID`:

`.cursor/skills/verify-sft-loop/evidence/<run-id>/`

Cleanup removes `/tmp/sft-loop-verify-<run-id>` only. It never deletes this directory.

The committed `cpu-linux-seed0/` tree is the first end-to-end run of this skill on a CPU Linux VM.
