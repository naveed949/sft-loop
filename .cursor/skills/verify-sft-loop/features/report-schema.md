# Run report schema

Every run report must validate against `schema/run_report.schema.json`. Status is `unsupported`, `unscored`, or `scored`. `weight_change_claimed` is true only with train proof on a scored report.

## Sub-features

- `schema-unsupported` forbids `before`, `after`, `train_proof`, and a true weight claim.
- `schema-scored` requires seed, task_id, before, after, and train_proof.
- `schema-device` allows `device` `mps` or `cuda` only on scored reports, never on unsupported.
- `schema-validate-cli` is `validate_report` after every official verify write.

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
- **Smoke artifact.** Run `scripts/verify smoke`. `smoke/plumbing.json` has `status` `plumbing` (not a schema status). No file under `smoke/` may `validate_report` as `status: scored`.
- **Surfaces file.** Read `surfaces.json`. Rows come from live `torch.cuda` / MPS probes and import attempts. Every blocked GPU extra uses reason `gpu_or_train_deps_unavailable` or `train_not_run`, with the probe error text when an import raises.
- **Proof.** Re-run validation on disk:

```bash
.venv/bin/python -c "import json; from pathlib import Path; from sft_loop.report import validate_report; p=Path('.cursor/skills/verify-sft-loop/evidence/' + __import__('os').environ['VERIFY_RUN_ID'] + '/reports/loop-default.json'); validate_report(json.loads(p.read_text()))"
```

## Gotchas

- `additionalProperties` is false on every object. Extra keys fail validation.
- Putting `device: cpu` on a schema run report fails the schema. Putting `device: mps` on CPU smoke would be a lie. Smoke therefore is not a run report.
- `runs/phase3/scored.json` is Cap's MPS run. Validate it with `proof_dir=runs/phase3/proof`. Do not replace it with smoke output.
- Soft-PASS is unused and is refused. A status outside `unsupported`, `unscored`, and `scored` must fail the schema.
- A JSON under `runs/` with no `status` is skipped only when it also has no score-like keys (`task_score`, `before`, `after`, `exact_match`). `adapter_config.json` is the skip case.
