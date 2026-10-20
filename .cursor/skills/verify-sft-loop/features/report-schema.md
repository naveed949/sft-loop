# Run report schema

Every run report must validate against `schema/run_report.schema.json`. Status is `unsupported`, `unscored`, or `scored`. `weight_change_claimed` is true only with train proof on a scored report.

## Sub-features

- `schema-unsupported` forbids `before`, `after`, `train_proof`, and a true weight claim.
- `schema-scored` requires seed, task_id, before, after, and train_proof.
- `schema-device` allows `device` `mps` or `cuda` only on scored reports, never on unsupported.
- `schema-validate-cli` is `validate_report` after every verify write.

## How to get to it (user POV)

- Print JSON from any `python -m sft_loop*` command.
- Inspect `runs/fixtures/unsupported.json`, `runs/fixtures/unscored.json`, and `runs/phase3/*.json`.
- Run `scripts/verify unsupported` and `scripts/verify smoke`.

## Driving it with verify-sft-loop

Preconditions:

- Doctor is green.
- `schema/run_report.schema.json` is present.

- **Fixtures.** They stay valid without a verify run. Do not modify them.
- **Official reports.** Run `scripts/verify unsupported`. `unsupported-validation.json` lists each evidence report with `"ok": true`.
- **Smoke report.** Run `scripts/verify smoke`. `smoke/report-validation.json` is `"ok": true` for a scored plumbing report, or the fail-closed unsupported smoke report validates.
- **Surfaces file.** Read `surfaces.json`. Every blocked GPU extra uses reason `gpu_or_train_deps_unavailable` or `train_not_run`. No invented reason strings.
- **Proof.** Re-run validation on disk:

```bash
.venv/bin/python -c "import json; from pathlib import Path; from sft_loop.report import validate_report; p=Path('.cursor/skills/verify-sft-loop/evidence/' + __import__('os').environ['VERIFY_RUN_ID'] + '/reports/loop-default.json'); validate_report(json.loads(p.read_text()))"
```

## Gotchas

- `additionalProperties` is false on every object. Extra keys fail validation.
- CPU smoke scored reports must omit `device` and `backend`. Putting `device: cpu` fails the schema. Putting `device: mps` would be a lie.
- `runs/phase3/scored.json` is Cap's MPS run. Validate it with `proof_dir=runs/phase3/proof`. Do not replace it with smoke output.
- Soft-PASS is unused and is refused. A status outside `unsupported`, `unscored`, and `scored` must fail the schema.
