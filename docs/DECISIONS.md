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

The eval task is open-book 10-K figure extraction, `tenk-extract-v1`. Source bytes are a read-only copy of `evals/golden/passages/` from capitalcite commit `34b42ee0b387738725705a0af682dd3f030a9944`. That pin is exact. Do not float it to capitalcite HEAD. `data/parsed/**/chunks.jsonl` was absent at that commit. The split is by issuer string, seed 0, so train and eval share no document and no issuer *string*. Issuer *families* can still overlap. CIK `0001628280` is JPM 2025 content and sits in eval next to JPM train. The grader is canonical-figure exact match. No LLM-as-judge. Baseline exact match at or above 0.85 refuses training (`refused_ceiling`). Missing `GradScaler` or a CPU torch wheel writes `unsupported` and does not write an adapter.

The GPU run is two-phase because this VM has no Kaggle token. Phase 1 ships `kaggle/phase4_data/` and `kaggle/phase4_train/`. Naveed pushes the dataset, then the kernel, on a 2x T4. T4 compute capability 7.5 has no bf16; the kernel records bf16 as `unsupported`. Mixed precision is attempted only after a finite-loss probe. Optional DDP is `unsupported` unless that process group actually initializes. Phase 2 commits the downloaded outputs, gitignores adapter weights, records their sha256, and writes `runs/phase4/scored.json` via `phase4.assemble`. That file does not exist in Phase 1.

LoRA here is ΔW = BA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, rank 8, alpha 16, so the scale is alpha/r = 2. Attention projections are the cheapest modules that can steer "which year-column figure" without MLP adapters on a 15 GB T4. Effective batch is 8 (batch 1 × 8 accum). The schedule is cosine with 5% warmup. Eval n=8 cannot show uplift: a Bernoulli exact-match SE is sqrt(p(1-p)/n) ≈ 0.18 at p=0.5, larger than a typical LoRA delta.

The arena cross-judge scored candidate 2 over this sibling (12 vs 10) and wanted `kaggle/tenk-extract-v1/` plus `runs/phase4/extraction.json`. Those names lose the ticket's `kaggle/phase4_data/`, `kaggle/phase4_train/`, and Phase 2 `runs/phase4/scored.json`. Keep the ticket paths. Take the honesty grafts that do not rename the handoff: refuse a live `data/parsed/**/chunks.jsonl`, pin a live capitalcite clone to `34b42ee0…`, record `split_hash` as sha256 of sorted eval ids then train ids, sort issuers before the seed-0 shuffle, and grade only a full-match bare figure.

## Phase 4 T4 fp16 overflow (2026-10-21)

Kaggle run 2 at `ce57285` found the nested dataset and wrote `split_manifest.json` plus `eval_before.json`, then died `RuntimeError: non-finite loss at step 1`. No adapter was written.

`eval_before` is not a usable baseline. Score was 0/123. Every prediction was `!!!!!!!!!!!!!!!!` (16 bangs, equal to `MAX_NEW_TOKENS`). That is a collapsed decode, not a real extract. The before number is invalid regardless of training.

The kernel loaded `Qwen2.5-1.5B-Instruct` with `torch_dtype=float16` and then trained under `GradScaler`. That recipe is wrong. GradScaler needs fp32 master weights. It scales the loss so tiny fp16 *gradients* do not underflow to zero. It cannot recover a forward whose activations are already Inf. IEEE fp16's max finite value is 65504. Qwen2.5 RMSNorm, attention, and `lm_head` can emit hidden states or logits above that once the compute dtype is fp16. When the stored weights are also fp16 there is no fp32 copy to fall back to, so the first backward sees a non-finite loss.

The same T4 image already showed this on PR #14: Qwen2.5-1.5B-Instruct in fp16 scored 0/129 with non-finite NLL; CPU fp32 scored 47/129.

The kernel now loads fp32 masters (1.5B × 4 bytes ≈ 6.2 GB, fits a 15 GB T4 with LoRA and checkpointing), probes one batch under `torch.autocast(dtype=float16)` plus a hidden-state max-abs hook, and requires a sane baseline (decoded text, finite NLL) before any optimizer step. If the AMP probe overflows, `fp16.json` records `unsupported` with the overflowing modules. Training then continues in fp32 and is labeled `fp32`, not mixed precision. The ticket's fp16 requirement is that measured `unsupported` finding.

## Phase 4 run 3 ingest (2026-10-21)

Kaggle `inaveed101/phase4-tenk-lora` version 3 at `ee7f5c8511d42916c02147b68731e6c0a36cd317` finished in 5087 s on 2x T4. Tarball sha256 `421df4ada47a4b8d98c22c8c0f606cc6eef038760bc62dfb15124ae4768cf507`. Adapter `adapter_model.safetensors` is 8,745,704 B, sha256 `e94fb237398a3e496552014b5ddb8dad6556a2ac56a669d480bbf6098f5e0109`. Weights stay gitignored. The digest is in `runs/phase4/raw/adapter.sha256` and `runs/phase4/proof/adapter.sha256`.

`eval_before` is usable. n=123, seed 0. The kernel stored 79/123 under the old grader. After the `$(370)$` canonicalizer fix, assemble regrades from the raw generations and scores 81/123 = 0.6585365853658537. Baseline NLL 0.6888449788093567 is finite and is one eval row only (`nll_scope=first_eval_row`). 102 distinct predictions. Zero `!` collapses. The same item ids as `kaggle/phase4_data/eval.jsonl`. Split hash still `3a298f9e0d83b00c9d3cfac99f90a160ebef3ce28c94d926b3f5efa266c95e33`. Contamination status is `same-issuer overlap`. The six JPM 2025 eval rows stay in the score.

`fp16.json` is `unsupported` / `fp16_amp_overflow`. The AMP probe loss is non-finite. 770 hooked modules overflowed. The first is `layers.0.self_attn.o_proj.base_layer` in float16 with non-finite max-abs. AMP greedy gens were four copies of `!!!!!!!!!!!!!!!!`. The fp32 probe on the same batch was finite (loss 10.357911109924316, zero overflow modules). Training used fp32 masters, autocast off, no GradScaler. `env.json` and `trainer_history.json` both say `mixed_precision=fp32`.

`eval_after` is 90/123 = 0.7317073170731707 on the same items, seed, greedy decode, and T4. The primary interval is a filing-cluster bootstrap (10000 rounds, seed 0, 10 `doc_id` clusters): mean delta 0.07317073170731707, 95% CI [0.0, 0.11214953271028037]. That interval covers 0. The verdict is `no uplift`. The secondary item bootstrap is the same mean, 95% CI [0.0, 0.14634146341463414], also covering 0. The item-level estimate is still positive. Say it this way: inconclusive / no uplift at the filing level, with a positive item-level estimate.

Of the remaining gains, 5 are format-equivalent figure strings and 10 are substantive. There are 6 losses. Net new correct figures are +4. The substantive interval covers 0. This is one kernel run, not a five-trial repeat. Do not retune toward a larger number.

Train loss fell from 1.745 to 0.00369 by step 7. Eval ran only at step 200. That is an overfitting risk. Leave the schedule alone.

The kernel is a manual PEFT loop. Label it `manual-peft`, not `trl-peft`. `try_ddp` has no success path. Record DDP as `unsupported` with that reason. The stored `hit` field on two before-rows was 0 for `$(370)$` and `$(545)$`. Assemble ignores stored `hit` and regrades.

Memory at batch 1, seq 1024. Checkpointing off peaked at 13,905,077,760 B. On peaked at 8,269,284,864 B. Saved 5,635,792,896 B. Both fit the 15,636,037,632 B T4. Profiler peak after train was 8,416,858,112 B. `aten::mm` was 76.23% of self CUDA time in the profiler table.

bf16 remains `unsupported` (T4 cc 7.5). DDP remains `unsupported` because the kernel is a single-process `main()` and never initializes a process group. `phase4.assemble` wrote `runs/phase4/scored.json` from the raw per-item JSON. That file exists.

## Clustered 10-K rows are not i.i.d. (2026-10-21)

107 of 123 eval rows come from three filings: `3M_2018_10K` (47), `AMERICANEXPRESS_2022_10K` (30), and `BOEING_2018_10K` (30). An item bootstrap treats those 107 rows as independent coin flips. They are not. Rows from one 10-K share the same prose, the same table layout, and the same year-column habit. The effective sample size is closer to the number of filings than to 123.

Kish's design-effect form is n_eff = n / (1 + (m - 1) ρ), where m is the mean cluster size and ρ is intra-cluster correlation. With m ≈ 12 and even a modest ρ, n_eff drops well below 123. A 10-filing resample is the honest interval. That is why `assemble.py` sets `uplift` from `cluster_ci`, and keeps `paired_ci` as the secondary item interval.

Issuer-level splits matter for the same reason. A 10-K is not a random document. The same issuer repeats language, units, and table stubs across years. The seed-0 split used issuer *strings*, so `0001628280` (JPM 2025) landed in eval while `JPM` / `0000019617` landed in train. Chunk overlap was empty, so the old check said `ok`. The model could still see JPM house style. `ISSUER_FAMILIES` is for contamination only. Do not add `0001628280` to `ISSUER_ALIASES`. That would reshuffle the locked split. Do not drop the six JPM eval rows from the main score. Report the overlap and a sensitivity that excludes them.

An exact-match gain can be format, not knowledge. The prompt asks for a bare canonical figure. A before-row that prints `$(370)$` and an after-row that prints `-370` are the same number. The old canonicalizer stripped `$` only at the start of the string, so the parentheses never flipped the sign. That hole made two already-correct figures look like misses, and it made format cleanup look like new facts. After the fix, 5 of the remaining gains still match the target figure inside extra words or units. Net new correct figures are +4, and that interval covers 0.

The next honest artifact is a proposal only. Do not run it on this GPU pass. Pre-register a filing-grouped train/eval split with no shared issuers, the cluster interval as the primary test, a format-versus-substantive breakdown, and at least two seeds. Then run that protocol.
