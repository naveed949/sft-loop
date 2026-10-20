# Same-seed eval

Eval grades greedy generations on the held-out synthetic task `uppercase-word-v1` with metric `exact_match`. Before and after must use the same seed, the same words, and the same grader.

## Sub-features

- `eval-before` loads the pinned base model with no adapter.
- `eval-after` loads the adapter written by train.
- `eval-blocked-cli` is the eval CLI on a CPU VM without MPS/CUDA.
- `eval-same-seed` refuses a scored report whose `before.seed` or `after.seed` drifts.

## How to get to it (user POV)

- Run `python -m sft_loop.eval --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --split before`.
- Run the same command with `--split after` and `--adapter <dir>` after train.
- Run `scripts/verify smoke`, which evals before train and after train with seed 0.

## Driving it with verify-sft-loop

Preconditions:

- Doctor is green.
- For after-eval, smoke train has written scratch adapters or smoke already assembled `smoke/plumbing.json`.

- **Blocked CLI.** Run `scripts/verify unsupported`. `reports/eval-before.json` and `reports/eval-after.json` are `unsupported`, seed 0, no `task_score`. Detail mentions the split name.
- **Shared setup smoke.** Run `scripts/verify smoke`. When it finishes, `smoke/plumbing.json` has `before.seed == after.seed == 0`, `metric_name` `exact_match`, `n` 8, `status` `plumbing`, and `weight_change_claimed` false.
- **Plumbing citation.** Read `smoke/plumbing.json` for model id, revision, seed, split, dtype, device `cpu`, backend `trl-peft`, and hardware. Those numbers are plumbing checks, not a scored report.
- **Proof.** No file under `smoke/` validates as a scored run report. Official eval reports still validate as `unsupported`.

## Gotchas

- After-eval without an adapter is `train_not_run` on the product CLI. That is correct, not a helper bug.
- Changing seed between splits invalidates the loop. `build_scored_report` raises and the loop records `unsupported`.
- Greedy decoding (`do_sample=False`) is required. Do not enable sampling to "get a better score".
- `n` must equal the held-out word count (8). Do not shrink the eval set to make smoke faster.
- Do not compare CPU smoke plumbing `task_score` to `runs/phase3/scored.json` (0.375 on MPS) as a quality delta.
- Product `--device cpu` argparse exits 2. Smoke records `device=cpu` only on the plumbing artifact.
