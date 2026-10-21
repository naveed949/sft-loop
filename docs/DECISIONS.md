# Decisions

This log records project choices that later agents should not rediscover. Each entry states what we chose and why.

## Why the model map is in the repo (2026-10-20)

Cloud agents read Cursor rules from this repository. They do not read a developer's home directory. The per-role map therefore lives at `.cursor/rules/pstack-models.mdc`.

`# budget: medium (high)` means each role starts at high reasoning. That trades some reasoning depth for lower cost compared with the xhigh skill defaults named in poteto-mode.

Naveed picks the budget. Do not change the map or the budget unless Naveed asks.

## Quant comparison model and metrics (2026-10-20)

The comparison uses `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. That is the same model id the SFT loop already pins. A revision SHA is required so every arm loads the same bytes.

The task score is first-token exact match on `evals/quant-heldout-v1.txt`, 129 lowercase words disjoint from `TRAIN_WORDS` and `EVAL_WORDS`. n is 129, which meets the n >= 100 bar. The grader is `sft_loop.task.prediction_token`. The scored pass is greedy, `max_new_tokens` 8, EOS allowed, seed 0. A second pass measures decode tokens/s with `min_new_tokens == max_new_tokens == 8`. Warmup is 2 iterations. Repeats per item are 5. Thread count is 1, set before torch imports.

When an arm exposes logits, `nll` is mean token NLL of the target word. The prompt tokens are masked with `labels[:, :prefix_len] = -100`. `sequence_nll` is full-sequence CE including the chat template. `evals/quant-perplexity-v1.txt` is the reserved Alice extract. An arm without logits omits both NLL fields. The paired difference is int8 minus fp32 on the same item ids. Parse recomputes it and rejects a stored block that disagrees. The published int8 arm is `default_dynamic_qconfig` (per-tensor symmetric) on every `nn.Linear`, including `lm_head`.

`schema/quant_comparison.schema.json` is a sibling of the SFT run report. Quant JSON has no `weight_change_claimed` field. NF4 cannot be a measured arm in this comparison. A 1.5B arm that cannot load on this VM is `unsupported` with the probed failure. A smaller-model rerun, if added, cannot occupy `runs/quant/comparison.json`.

On this CPU VM the pinned 1.5B model loaded. fp32 and torch dynamic int8 finished the 129-item protocol, including the fixed-length decode pass. ONNX export wrote a 5.8 GB `model.onnx.data` then `quantize_dynamic` was SIGKILL 137, so that arm is `oom`. NF4 is `missing_cuda`. The README table is generated from those files. Do not type the scores by hand.

## Phase 4 is a closed sibling (2026-10-21)

Phase 4 is not a new status on `schema/run_report.schema.json` and not a task on `sft_loop.task`. The uppercase fixture stays `uppercase-word-v1` with `n` 8. A CUDA + TRL + PEFT report with eval n >= 100 cannot pass `validate_report` today: device/backend pairing is `cuda`+`unsloth` or `mps`+`trl-peft`, and `_check_same_seed` locks `n` to `len(EVAL_WORDS)`. The new code lives in `src/phase4/` and `kaggle/`. An AST fence forbids `sft_loop` and `unsloth` imports there. Product modules must not import `phase4`.

The eval task is open-book 10-K figure extraction, `tenk-extract-v1`. Source bytes are a read-only copy of `evals/golden/passages/` from capitalcite commit `34b42ee0b387738725705a0af682dd3f030a9944`. `data/parsed/**/chunks.jsonl` was absent at that commit. The split is by issuer, seed 0, so train and eval share no document and no issuer. The grader is canonical-figure exact match. No LLM-as-judge. Baseline exact match at or above 0.85 refuses training (`refused_ceiling`). Missing `GradScaler` or a CPU torch wheel writes `unsupported` and does not write an adapter.

The GPU run is two-phase because this VM has no Kaggle token. Phase 1 ships `kaggle/phase4_data/` and `kaggle/phase4_train/`. Naveed pushes the dataset, then the kernel, on a 2x T4. T4 compute capability 7.5 has no bf16; the kernel records bf16 as `unsupported` and trains with fp16 autocast plus `GradScaler`. Optional DDP is `unsupported` unless that process group actually initializes. Phase 2 commits the downloaded outputs, gitignores adapter weights, records their sha256, and writes `runs/phase4/scored.json` via `phase4.assemble`. That file does not exist in Phase 1.

LoRA here is ΔW = BA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, rank 8, alpha 16, so the scale is alpha/r = 2. Attention projections are the cheapest modules that can steer "which year-column figure" without MLP adapters on a 15 GB T4. Effective batch is 8 (batch 1 × 8 accum). The schedule is cosine with 5% warmup. Eval n=8 cannot show uplift: a Bernoulli exact-match SE is sqrt(p(1-p)/n) ≈ 0.18 at p=0.5, larger than a typical LoRA delta.

The arena cross-judge scored candidate 2 over this sibling (12 vs 10) and wanted `kaggle/tenk-extract-v1/` plus `runs/phase4/extraction.json`. Those names lose the ticket's `kaggle/phase4_data/`, `kaggle/phase4_train/`, and Phase 2 `runs/phase4/scored.json`. Keep the ticket paths. Take the honesty grafts that do not rename the handoff: refuse a live `data/parsed/**/chunks.jsonl`, pin a live capitalcite clone to `34b42ee0…`, record `split_hash` as sha256 of sorted eval ids then train ids, sort issuers before the seed-0 shuffle, and grade only a full-match bare figure.
