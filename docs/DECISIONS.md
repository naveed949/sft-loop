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

On this CPU VM the pinned 1.5B model loaded. fp32 and torch dynamic int8 finished the 129-item protocol, including the fixed-length decode pass. ONNX export wrote a 5.8 GB `model.onnx.data` then `quantize_dynamic` was SIGKILL 137, so that arm is `oom`. NF4 is `missing_cuda` on this host. The README table is generated from those files. Do not type the scores by hand.

## Quant GPU pairing and Kaggle kernels (2026-10-21)

NF4 can be a measured arm when a Kaggle T4 kernel output is ingested. The same-hardware baseline is fp32 on that T4, not CPU fp32 and not T4 fp16. `outcomes` owns `int8-minus-fp32` on the CPU host and `nf4-minus-t4-fp32` / `nf4-fp16-minus-t4-fp32` on one `hardware.accelerator`. A GPU number is never subtracted from a CPU number. A cross-hardware view is `kind: not_paired` with reason `cross_hardware`.

GPU measured arms require `hardware.accelerator`, `peak_gpu_bytes`, a kernel URL/version placeholder, and fp16 compute. NF4 also records BitsAndBytesConfig `load_in_4bit`, `bnb_4bit_quant_type=nf4`, double quant on, and fp16 compute.

Local `python -m sft_loop.quant probe-nf4` stays unsupported on this CPU VM. The only measured GPU path is `python -m sft_loop.quant ingest`. A missing kernel directory is `unsupported` / `kernel_output_missing` with no metric keys.

Kernels live at `kaggle/quant_nf4` (NvidiaTeslaT4) and `kaggle/quant_onnx_int8` (CPU only). They pin deps, download the public Hub revision, and write env, logs, progress, and per-arm items. Local verify cannot run them. Downloaded outputs are committed under `runs/quant/kernel/`. The table is regenerated from ingest. `python -m sft_loop.quant run` reuses measured kernel arms so it cannot overwrite them with a local probe.

## Quant T4 ingest and ONNX cp313 pin (2026-10-21)

Kaggle kernel `https://www.kaggle.com/code/inaveed101/sft-loop-quant-nf4` version 1 is withdrawn. Both T4 arms scored 0/129, every item emitted 8 tokens, and NLL was non-finite. CPU fp32 on the same model, items, and grader scored 47/129. That is a harness or numerics bug, not a pairing. The v1 numbers stay under `runs/quant/kernel/quant_nf4/` as evidence. They are not a baseline.

The replacement kernel runs fp32-on-T4 first (`t4-fp32`, about 6.2 GB weights, fits a 15 GB T4). It fail-closes if that baseline scores 0 or has non-finite NLL. Ingest refuses the same state unless `finding` is `collapse`. fp16 stays its own arm. If it overflows, it is a measured `finding: overflow`, not a pairing baseline. NF4 compute dtype is float32; nf4-fp16 compute runs only if fp16 is healthy. Pairing is NF4 versus `t4-fp32`.

The kernel pins `transformers==5.20.2` and `tokenizers==0.23.3` to match the CPU arm. It writes `prompt_hashes.json` and a two-item numerics probe (`fp32_eager`, `fp16_eager`, `fp16_sdpa`) with `torch.isfinite` on logits and per-layer hidden max abs.

Expected T4 wall is about 2 hours. Budget 2.5 hours. Stay under 3 GPU hours.

PR #15's Phase 4 T4 kernel hit a non-finite training loss at step 1 in fp16 on the same accelerator. That is a training-loop observation, not this eval harness, and it is consistent with the fp16 overflow hypothesis.

The ONNX kernel failed during pip. `onnx==1.17.0` has no cp313 wheel; the image is Python 3.13.15. The pin is now `onnx==1.18.0`. That kernel is being rerun separately. The ONNX arm stays `unsupported` / `probed_failure` until those outputs land.

## Quant ONNX torch 2.11 exporter: dynamo=False, not onnx-ir (2026-10-21)

ONNX Kaggle rerun v2 at `2eede76` installed `onnx==1.18.0`, `onnxruntime==1.20.1`, `onnxscript==0.2.0`, `transformers==4.56.2` and then failed in `torch.onnx.export`:

`ModuleNotFoundError: No module named 'onnx_ir'`

The host was 4 CPUs and `ram_bytes=33658318848` (33.658 GB). Python 3.13.15, torch 2.11.0. v2 never reached graph export; peak RSS during export is unmeasured.

Chose **(b) `dynamo=False`**, not **(a) pin `onnx-ir` plus a dynamo-compatible onnxscript**.

torch 2.11's `torch.onnx.export` defaults `dynamo=True`. That branch is the only one that does `from torch.onnx._internal.exporter import _compat`. `_compat` → `_core` → `_schemas` → `_lazy_import("onnx_ir")`. The TorchScript branch (`else`) imports `torch.onnx._internal.torchscript_exporter.utils` and never touches `onnx_ir`.

Rejected (a) because v2 already had `onnxscript==0.2.0` and still died on `onnx_ir`. The dynamo path needs both, and the versions have to match torch 2.11's private API (`onnxscript._framework_apis.torch_2_11`). pytorch/pytorch#179280 on 2.11.0 needed `onnx-ir` plus onnxscript 0.6.2, not 0.2.0. Pinning that pair is a second moving dependency after the 1.17.0 wheel miss. `dynamic_axes` / `do_constant_folding` are the TorchScript flags.

Local proof, same torch 2.11.0+cpu, pins `onnx==1.18.0` and `onnxscript==0.2.0`, no `onnx-ir`. This VM has Python 3.12.3, not 3.13; the wheel's `torch/onnx/__init__.py` and `_schemas.py` match the Kaggle traceback line for line.

```
python -c "import torch.onnx._internal.exporter._core"
# ModuleNotFoundError: No module named 'onnx_ir'
```

`torch.onnx.export(..., dynamo=False)` on a one-add module wrote a 184-byte ONNX file. After that call, `onnx_ir` and `torch.onnx._internal.exporter._compat` were still absent from `sys.modules`. `dynamo=True` still raised `No module named 'onnxscript'` or `onnx_ir` depending on whether 0.2.0 was installed.

Export peak RAM vs 33.6 GB: v2 died at import, so there is no Kaggle export peak to report. The earlier local 1.5B export on this 16.781 GB host wrote a 5.8 GB `model.onnx.data` and then took SIGKILL 137 in `quantize_dynamic`. Export of the same model already fit in 16.781 GB, so 33.658 GB (2.0×) has headroom for export. The remaining RAM risk is quantize, not `torch.onnx.export`. The kernel writes `export_mem.json` at `export_start`, `export_done` (file sizes included), and `quantize_done` with `peak_rss_bytes`, `host_ram_bytes`, and `fraction_of_host`. Compare the next run's `export_done` peak against 33658318848.

The ONNX arm stays `unsupported` / `probed_failure` until a later rerun lands. v2 and v3 logs are under `runs/quant/kernel/quant_onnx_int8/`.

## Quant T4 v2 ingest and ONNX 4D mask (2026-10-21)

Kaggle `https://www.kaggle.com/code/inaveed101/sft-loop-quant-nf4` version 2 ran the fail-closed kernel at `38bd88b5720b2f2e0ca9502b30180764ad90228a`. Wall 1972 s on NvidiaTeslaT4. Tarball sha256 `ecb97484656bc7c7e0b4a758653f5a59528dee3919523641650dcb2f291358e8`.

`t4-fp32` scored 47/129. That matches CPU fp32 from PR #13 on the same items, grader, seed, and revision. Mean generated tokens also match (3.248062). Target NLL means match to 1e-5 (CPU 0.79975264, T4 0.79975481). The same-hardware baseline is real.

`metrics.json` from the kernel had no `exact_match` field. Ingest already derives `task_score` from `items[].correct`. The kernel now writes `exact_match` as that same ratio. The committed v2 metrics files were backfilled from their items.

Headline pairing is NF4 minus `t4-fp32` on one T4, n=129, bootstrap 1000, seed 0:

- exact match 79/129 versus 47/129. Score delta +0.24806202, se 0.041224898, CI 0.17054264 to 0.3255814.
- target NLL mean 0.51175187 versus 0.79975481. NLL delta -0.28800294.
- sequence NLL mean 4.7892442 versus 4.6356639. Sequence NLL delta +0.15358034.
- fixed-length decode 15.304685 tok/s versus 25.135479 tok/s (`min_new_tokens=max_new_tokens=8`).
- peak GPU 1224451584 versus 6238088192 bytes. On-disk 1143625737 versus 3098955668 bytes.

This is one Kaggle run. The protocol warms up 2 times and repeats each item 5 times. `limiter.kind` is `gpu`. All 129 NF4 and t4-fp32 rows have finite NLL and emit 8 decode tokens. The work is in `items.json`. Do not pair T4 tok/s with CPU tok/s.

CPU int8 from PR #13 (same host as CPU fp32) scored 0/129. Target NLL mean rose to 10.032154. Sequence NLL rose to 8.068833. On-disk fell to 2477926394. Peak RSS rose to 15464464384. Decode rose to 6.4966602 tok/s. Per-tensor int8 on every `nn.Linear`, including `lm_head`, destroyed first-token exact match and made NLL much worse.

NF4 stores each 64-wide block as a 4-bit codebook index plus a scale. Double quant (`bnb_4bit_use_double_quant=true`) quantizes those scales too, about 4.125 bits per parameter. That is more compression than int8 (8 bits plus qconfig metadata) and much less than fp32 (32 bits). The extra scale quantization is more error than single-level NF4, not less.

Quantization error and this task do not move together. NF4 raised exact match and lowered target-token NLL, and it raised full-sequence NLL. Int8 lowered exact match and raised both NLL numbers. Sequence NLL is the closer stand-in for perplexity. A first-token exact-match gain is not a perplexity win. Do not read the NF4 score delta as a general quality claim. The field-by-field check is in the later "NF4 79/129 is first-token exact match" entry.

### Why fp16 failed

`probe/numerics.json` on `anchor` and `beacon`:

- `fp32_eager`: `logits_finite=true`, hidden max abs about 6636, logits max abs about 30.55.
- `fp16_eager`: `logits_finite=false`, `hidden_finite=false`. Layer 0 max abs 0.1787, then NaN from layer 1. This is overflow in eager fp16, not a template mismatch.
- `fp16_sdpa`: `logits_finite=true`, hidden max abs about 6644 (under fp16 max 65504), logits max abs about 30.59.

The scored fp16 arm used `attn_implementation=eager`, so it overflowed. 0/129, every item emitted 8 tokens, NLL stripped as non-finite, `finding: overflow`. `nf4-fp16` was skipped. PR #15 Phase 4 hit a non-finite train loss at step 1 in fp16 on the same T4. That is the same class of bug.

### ONNX v3 and the 4D mask

Kaggle `https://www.kaggle.com/code/inaveed101/sft-loop-quant-onnx-int8` version 3, sha256 `02d6e74e87fae1be20aa6f30310347ec3979ce2019099552ee70e10800681855`, wall 266 s. `export_mem.json` peaked at 10222952448 bytes at `export_start` (0.3037 of 33658318848).

`dynamo=False` worked. TorchScript then died in transformers 4.56.2 `create_causal_mask` → `eager_mask` → `sdpa_mask_recent_torch` → vmap `and_mask` / `inner_mask` with `RuntimeError: unordered_map::at`.

The failed premise after v1 and v2 was "export works once the exporter is callable." v3 falsified that. The first wrong step is asking the model to build a 2D pad mask under JIT.

Chose a wrapper that builds a 4D additive causal mask from the 2D pad mask and passes that in. `masking_utils._preprocess_mask_arguments` returns a 4D mask as-is (line 709). Rejected a transformers bump or legacy-mask flag because the ONNX kernel stays on 4.56.2 and the 4D early return is already there.

Local proof, torch 2.11.0+cpu, transformers 4.56.2, tiny `Qwen2Config` (hidden 64, 2 layers), `dynamo=False`:

- 2D wrapper: `RuntimeError: unordered_map::at` at `masking_utils.py:142` `inner_mask`.
- 4D wrapper: wrote `/tmp/qwen2-4d.onnx` (524851 bytes).

The ONNX arm stays `unsupported` / `probed_failure` until the 4D kernel reruns.

## NF4 79/129 is first-token exact match, not a scorer bug (2026-10-21)

The +25 point NF4 minus t4-fp32 exact-match delta is the wrong sign for lossy 4-bit storage of the same weights. That made it a measurement-bug candidate until the run-3 artifacts were compared field by field. No GPU rerun.

### Prompt parity

One tokenizer is loaded before any arm. `dump_prompt_hashes` runs once. `render_prompt` is `apply_chat_template(..., add_generation_prompt=True)` for every arm. Scored `generate_fn` kwargs are `max_new_tokens=8`, `do_sample=False`, `pad_token_id=tokenizer.pad_token_id`. `min_new_tokens` is unset on the scored path. The decode path is a second call with `min_new_tokens=8`. Temperature is unset. Stop is EOS or `max_new_tokens`. transformers 5.20.2 and tokenizers 0.23.3 match. There is no per-item per-arm prompt hash in the v2 tarball. The shared file is `runs/quant/kernel/quant_nf4/prompt_hashes.json`.

### Scoring parity

`prediction_token` strips the decode and takes the first whitespace token. Gold is `item_id.upper()`. Ingest `task_score` is the mean of `items[].correct` via `_summarize_items`. It does not re-grade text. Kernel `metrics.exact_match` is the same ratio. Both arms use that path.

### Flips

| class | n | item ids |
| --- | --- | --- |
| both right | 45 |  |
| both wrong | 48 |  |
| fp32 wrong, nf4 right | 34 | anchor, blanket, carpet, celery, cherry, cricket, custom, desert, ember, glacier, grape, hammer, horse, knight, ladder, meadow, mirror, orange, orchid, otter, palace, pebble, pepper, plum, prairie, quartz, rocket, scarlet, silver, stable, thunder, velvet, violet, window |
| fp32 right, nf4 wrong | 2 | canyon, wizard |

Net +32 equals 79 minus 47. Twenty of those 36 flips change `generated_tokens`, so the decoded sequences differed. The 15 same-length fp32-wrong/nf4-right items still dropped target NLL (mean 0.853 → 0.298). No flip has NF4 worse target NLL and NF4 `correct=true`. CPU fp32 from PR #13 matches t4-fp32 on all 129 `correct` flags and all 129 `generated_tokens`. Mean NLL matches to 1e-5. The 47/129 baseline is the same set.

Raw generation strings are not in `items.json`, `metrics.json`, or the kernel log. The 15 requested verbatim generations cannot be printed. PR #13 CPU items have the same schema. A scorer change cannot rescore any arm. The #13 numbers do not change.

### PR #15 79/123

Coincidence. PR #15 eval_before is 79/123 on a 10-K figure extract task (`capitalcite`, n=123). This pairing is 79/129 on `quant-heldout-v1` uppercase words. Same numerator, different n, different labels.

### Mechanism

Not a prompt mismatch. Not `min_new_tokens=8` on the scored path. Not a leading-space or case artifact that one arm hits and the other does not. The first-token grader and teacher-forced target NLL move together. NF4 mean target NLL is 0.51175 versus 0.79975. Sequence NLL is 4.789 versus 4.636, the sign lossy quant predicts. Exact match is a different functional. The 1.5B instruct model often emits a short chatty prefix under fp32 (2–3 tokens, first token ≠ gold). NF4 more often emits the uppercase word and EOS. That can raise first-token exact match while making the full chat-template sequence less likely.

The table keeps 79/129 and 47/129. Do not read the score delta as a general quality claim. Sequence NLL is the closer stand-in for perplexity, and it got worse.

`assert_prompt_and_generation_parity` now fails ingest when two GPU arms in one kernel dir differ in prompt hashes or scored generation config. The next NF4 kernel writes per-arm `generation_config.json`, copies `prompt_hashes.json`, and writes `predictions.json` so a later rescore does not need a GPU rerun. Run-3 `generation_config.json` files are a backfill from `kernel.py`. pad and eos ids were not recorded, so those fields are null.
