# Decisions

This log records project choices that later agents should not rediscover. Each entry states what we chose and why.

## Why the model map is in the repo (2026-10-20)

Cloud agents read Cursor rules from this repository. They do not read a developer's home directory. The per-role map therefore lives at `.cursor/rules/pstack-models.mdc`.

`# budget: medium (high)` means each role starts at high reasoning. That trades some reasoning depth for lower cost compared with the xhigh skill defaults named in poteto-mode.

Naveed picks the budget. Do not change the map or the budget unless Naveed asks.

## Quant comparison model and metrics (2026-10-20)

The comparison uses `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. That is the same model id the SFT loop already pins. A revision SHA is required so every arm loads the same bytes.

The task score is case-sensitive first-token exact match on `evals/quant-heldout-v1.txt`, 129 lowercase words disjoint from `TRAIN_WORDS` and `EVAL_WORDS`. n is 129, which meets the n >= 100 bar. The grader is `sft_loop.task.prediction_token` compared to `item_id.upper()`. The scored pass is greedy, `max_new_tokens` 8, EOS allowed, seed 0. A second pass measures decode tokens/s with `min_new_tokens == max_new_tokens == 8`. Warmup is 2 iterations. Repeats per item are 5. Thread count is 1, set before torch imports.

When an arm exposes logits, `nll` is mean token NLL of the target word. The prompt tokens are masked with `labels[:, :prefix_len] = -100`. `sequence_nll` is full-sequence CE including the chat template. `evals/quant-perplexity-v1.txt` is the reserved Alice extract. An arm without logits omits both NLL fields. The paired difference is int8 minus fp32 on the same item ids. Parse recomputes it and rejects a stored block that disagrees. The published int8 arm is `default_dynamic_qconfig` (per-tensor symmetric) on every `nn.Linear`, including `lm_head`.

`schema/quant_comparison.schema.json` is a sibling of the SFT run report. Quant JSON has no `weight_change_claimed` field. NF4 cannot be a measured arm in this comparison. A 1.5B arm that cannot load on this VM is `unsupported` with the probed failure. A smaller-model rerun, if added, cannot occupy `runs/quant/comparison.json`.

On this CPU VM the pinned 1.5B model loaded. fp32 and torch dynamic int8 finished the 129-item protocol, including the fixed-length decode pass. ONNX export wrote a 5.8 GB `model.onnx.data` then `quantize_dynamic` was SIGKILL 137, so that arm is `oom`. NF4 is `missing_cuda` on this host. The README table is generated from those files. Do not type the scores by hand.

## Quant GPU pairing and Kaggle kernels (2026-10-21)

NF4 can be a measured arm when a Kaggle T4 kernel output is ingested. The same-hardware baseline is fp32 on that T4, not CPU fp32 and not T4 fp16. `outcomes` owns `int8-minus-fp32` on the CPU host and `nf4-minus-t4-fp32` / `nf4-fp16-minus-t4-fp32` on one `hardware.accelerator`. A GPU number is never subtracted from a CPU number. (Superseded 2026-10-22: `assemble_comparison` never writes `kind: not_paired`. `cross_hardware_view` exists in the module and is unused. Cross-hardware pairs are simply absent from `outcomes`.)

GPU measured arms require `hardware.accelerator`, `peak_gpu_bytes`, and a kernel URL/version placeholder. `t4-fp32` and `nf4` compute dtype is float32. (Superseded 2026-10-22: measured arms do not require fp16 compute. `nf4-fp16` is the only NF4 arm that would use float16, and it stays unsupported when fp16 overflows.)

Local `python -m sft_loop.quant probe-nf4` stays unsupported on this CPU VM. The only measured GPU path is `python -m sft_loop.quant ingest`. A missing kernel directory is `unsupported` / `kernel_output_missing` with no metric keys.

Kernels live at `kaggle/quant_nf4` (NvidiaTeslaT4) and `kaggle/quant_onnx_int8` (CPU only). They pin deps, download the public Hub revision, and write env, logs, progress, and per-arm items. Local verify cannot run them. Downloaded outputs are committed under `runs/quant/kernel/`. The table is regenerated from ingest. `python -m sft_loop.quant run` reuses a measured arm from `comparison.json` or the per-arm projection. It never replaces that arm with `probe_nf4` or any unsupported result. When no measured arm is stored, every T4 arm including NF4 ingests from `runs/quant/kernel/quant_nf4`. `probe_nf4` is only the local CLI probe.

## Quant T4 ingest and ONNX cp313 pin (2026-10-21)

Kaggle kernel `https://www.kaggle.com/code/inaveed101/sft-loop-quant-nf4` version 1 is withdrawn. Both T4 arms scored 0/129, every item emitted 8 tokens, and NLL was non-finite. CPU fp32 on the same model, items, and grader scored 47/129. That is a harness or numerics bug, not a pairing. (Superseded 2026-10-22: the v1 kernel files are not in the tree. Canonical T4 evidence is kernel v4 under `runs/quant/kernel/quant_nf4/`.)

The replacement kernel runs fp32-on-T4 first (`t4-fp32`, about 6.2 GB weights, fits a 15 GB T4). It fail-closes if that baseline scores 0 or has non-finite NLL. Ingest refuses the same state unless `finding` is `collapse`. fp16 stays its own arm. (Superseded 2026-10-22: fp16 overflow is recorded `unsupported` / `probed_failure` with detail `fp16 overflow on T4, 0/129`, not `measured` with `finding: overflow`. A missing numerics probe fails closed. It is still not a pairing baseline.) NF4 compute dtype is float32; nf4-fp16 compute runs only if fp16 is healthy. Pairing is NF4 versus `t4-fp32`.

The kernel pins `transformers==5.20.2` and `tokenizers==0.23.3` to match the CPU arm. It writes `prompt_hashes.json` and a two-item numerics probe (`fp32_eager`, `fp16_eager`, `fp16_sdpa`) with `torch.isfinite` on logits and per-layer hidden max abs.

Expected T4 wall is about 2 hours. Budget 2.5 hours. Stay under 3 GPU hours.

PR #15's Phase 4 T4 kernel hit a non-finite training loss at step 1 in fp16 on the same accelerator. That is a training-loop observation, not this eval harness, and it is consistent with the fp16 overflow hypothesis.

The ONNX kernel failed during pip. `onnx==1.17.0` has no cp313 wheel; the image is Python 3.13.15. The pin is now `onnx==1.18.0`. (Superseded 2026-10-22: ONNX kernel v4 landed and is `measured` with `finding: collapse`. See the 2026-10-22 ONNX entry.)

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

(Superseded 2026-10-22: the 4D-mask kernel reran as v4 and is measured collapse. v2 and v3 logs were overwritten by that download under `runs/quant/kernel/quant_onnx_int8/`.)

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

The scored fp16 arm used `attn_implementation=eager`, so it overflowed. 0/129, every item emitted 8 tokens, NLL stripped as non-finite. (Superseded 2026-10-22: ingest records that arm `unsupported` / `probed_failure`, detail `fp16 overflow on T4, 0/129`. Kernel items stay in `runs/quant/kernel/quant_nf4/fp16/` as evidence. The comparison no longer stores a measured 0.0 score.) `nf4-fp16` was skipped. PR #15 Phase 4 hit a non-finite train loss at step 1 in fp16 on the same T4. That is the same class of bug.

### ONNX v3 and the 4D mask

Kaggle `https://www.kaggle.com/code/inaveed101/sft-loop-quant-onnx-int8` version 3, sha256 `02d6e74e87fae1be20aa6f30310347ec3979ce2019099552ee70e10800681855`, wall 266 s. `export_mem.json` peaked at 10222952448 bytes at `export_start` (0.3037 of 33658318848).

`dynamo=False` worked. TorchScript then died in transformers 4.56.2 `create_causal_mask` → `eager_mask` → `sdpa_mask_recent_torch` → vmap `and_mask` / `inner_mask` with `RuntimeError: unordered_map::at`.

The failed premise after v1 and v2 was "export works once the exporter is callable." v3 falsified that. The first wrong step is asking the model to build a 2D pad mask under JIT.

Chose a wrapper that builds a 4D additive causal mask from the 2D pad mask and passes that in. `masking_utils._preprocess_mask_arguments` returns a 4D mask as-is (line 709). Rejected a transformers bump or legacy-mask flag because the ONNX kernel stays on 4.56.2 and the 4D early return is already there.

Local proof, torch 2.11.0+cpu, transformers 4.56.2, tiny `Qwen2Config` (hidden 64, 2 layers), `dynamo=False`:

- 2D wrapper: `RuntimeError: unordered_map::at` at `masking_utils.py:142` `inner_mask`.
- 4D wrapper: wrote `/tmp/qwen2-4d.onnx` (524851 bytes).

(Superseded 2026-10-22: the 4D kernel reran as v4. See the 2026-10-22 ONNX entry.)

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

Net +32 equals 79 minus 47. Twenty of those 36 flips change `generated_tokens`, so the decoded sequences differed. The 15 same-length fp32-wrong/nf4-right items still dropped target NLL (mean 0.853 → 0.298). No flip has NF4 worse target NLL and NF4 `correct=true`. Canyon is the reverse case. Target NLL improved (`1.063` to `0.828`) while `correct` went false at the same length. Mean NLL over the gold word is not the greedy first token. CPU fp32 from PR #13 matches t4-fp32 on all 129 `correct` flags and all 129 `generated_tokens`. Mean NLL matches to 1e-5. The 47/129 baseline is the same set.

Raw generation strings are not in `items.json`, `metrics.json`, or the kernel log. The 15 requested verbatim generations cannot be printed. PR #13 CPU items have the same schema. A scorer change cannot rescore any arm. The #13 numbers do not change.

### PR #15 79/123

Disproven as the same measurement. PR #15 eval_before is 79/123 (`0.64227642`) on `tenk-extract-v1`. Item-id overlap with `quant-heldout-v1` is 0. This pairing is 79/129 (`0.6124031`) on uppercase words. Same numerator, different n, different labels, different task.

### Mechanism

Not a prompt mismatch. Not `min_new_tokens=8` on the scored path. The first-token grader and teacher-forced target NLL move together on the 34 gains. NF4 mean target NLL is 0.51175 versus 0.79975. Sequence NLL rose on all 129 items (4.789 versus 4.636). That is the sign lossy quant predicts. The sequence mean is dominated by the shared chat template, so a better gold-word NLL does not pull it down. Exact match is a different functional. Retracted. The chatty-prefix mechanism was wrong. The later run-4 entry shows 33 of the 34 gains are title-case versus ALL-CAPS.

The table keeps 79/129 and 47/129 as the task-as-defined score. Do not read the score delta as a general quality claim. Sequence NLL is the closer stand-in for perplexity, and it got worse.

`assert_prompt_and_generation_parity` now fails ingest when two GPU arms in one kernel dir differ in prompt hashes or scored generation config. The NF4 kernel writes per-arm `generation_config.json`, copies `prompt_hashes.json`, and writes `predictions.json` so a later rescore does not need a GPU rerun.

## NF4 79/129 is casing, not a chatty prefix (2026-10-22)

Kaggle `https://www.kaggle.com/code/inaveed101/sft-loop-quant-nf4` version 4 ran the kernel at `d0f7517428cb8f23ef74ad49a625f663cd640721`. Status COMPLETE. Tarball sha256 `db52542b02f0aae1d22ddf8666bb34962eb0921ed90abfdbed6607ef19278aef`. The tarball includes `predictions.json`. A token grep of that file found 0. Run 4 is the canonical NF4 evidence.

Run 4 matches run 3 on all 129 items per arm for `correct`, `generated_tokens`, target NLL, and sequence NLL. Latency and tok/s differ because this is a new wall-clock.

### Both metrics

The prompt is `Uppercase the word: {word}`. The grader is `prediction_token == item_id.upper()` and is case-sensitive. Case-folded match is `prediction_token.casefold() == item_id.casefold()`. CPU #13 fp32 and int8 have no raw generations, so case-folded is n/a.

| arm | strict | case-folded | n | raw text |
| --- | --- | --- | --- | --- |
| CPU fp32 (#13) | 47 | n/a | 129 | no |
| CPU int8 (#13) | 0 | n/a | 129 | no |
| t4-fp32 (run 4) | 47 | 126 | 129 | yes |
| nf4 (run 4) | 79 | 126 | 129 | yes |
| fp16 (run 4) | 0 | 0 | 129 | yes (`!!!!!!!!`) |

Paired NF4 minus t4-fp32, n=129, bootstrap 1000, seed 0:

- strict exact match 79/129 versus 47/129. Score delta +0.24806202, se 0.041224898, CI 0.17054264 to 0.3255814.
- case-folded match 126/129 versus 126/129. Delta 0.0, se 0.01556432, CI -0.031007752 to 0.031007752.
- target NLL mean 0.51175187 versus 0.79975481. NLL delta -0.28800294.
- sequence NLL mean 4.7892442 versus 4.6356639. Sequence NLL delta +0.15358034. Sequence NLL rose on all 129 items.

int8 minus CPU fp32 stays -47/129 strict. Case-folded cannot be computed. Sequence NLL rose to 8.068833.

### Flip table, split

| class | n | of which case-only | other |
| --- | --- | --- | --- |
| both right | 45 |  |  |
| both wrong | 48 |  |  |
| fp32 wrong, nf4 right | 34 | 33 | 1 (`otter`) |
| fp32 right, nf4 wrong | 2 | 2 (`canyon`, `wizard`) | 0 |

Case-only means the first tokens differ only by case. fp32 emits `Anchor`, `Blanket`, `Carpet`. NF4 emits `ANCHOR`, `BLANKET`, `CARPET`. The gold is `ANCHOR`. The reverse canyon and wizard flips are `CANYON` to `Canyon` and `WIZARD` to `Wizard`.

The one non-case gain is `otter`. fp32 emitted `The uppercase version of "otter"`. NF4 emitted `OTTER`. That is a real chatty prefix on one item. It is not the 33-item pattern.

Case-folded misses. t4-fp32: `drawer` and `otter` start with `The`, `zebra` is `ZEBRAS`. NF4: `helmet` is `Helmets`, `pocket` is `Pockets`, `zebra` is `ZEBRAS`. Shared miss is `zebra` only.

### Retract the chatty-prefix claim

The earlier mechanism said fp32 often emits a short chatty prefix and NF4 more often emits the uppercase word. That is false for 33 of 34 gains. Both arms usually emit the word. They disagree on title case versus ALL-CAPS.

### Why quantization can flip `Anchor` versus `ANCHOR`

Qwen2.5 tokenizes `Anchor` and `ANCHOR` as different first tokens. After the chat template, the model is close to a tie between those two logits. Greedy decode takes the argmax. NF4 replaces each 64-wide block with a 4-bit codebook index plus a double-quantized scale, about 4.125 bits per parameter. That weight noise moves the two logits by a small amount. On a near-tie, a small move flips the winner. The gold is the ALL-CAPS token, so the case-sensitive grader records a win when the flip goes that way. Teacher-forced target NLL still uses the gold token, so it can fall even when greedy decode used to pick the title-case sibling.

This is a real, narrow instruction-following difference on one synthetic uppercase task. It is not a quality gain.

### The eval is too easy for quantization quality

Case-folded match is 126/129 on both T4 arms. Both models know the word. A 126/129 ceiling leaves almost no room to measure whether 4-bit storage hurt recall. First-token exact match then mostly measures a casing near-tie. Target NLL and sequence NLL still move. Sequence NLL rose on every item under NF4. That is the sign lossy quant predicts, and it is the number to keep in front.

Ingest now requires `metrics.grader_case_sensitive` to be true. Protocol and measured arms declare the same field. When `predictions.json` exists, ingest attaches per-item `case_folded` and summarizes `case_folded_match`. Pairing reports a case-folded delta and CI when both arms have that field. The NF4 kernel writes the declaration. The ONNX kernel now writes the same files. Run 4 itself has no raw text; see the next entry.

## ONNX int8 run 4 is measured collapse, not a generate-loop bug (2026-10-22)

Kaggle `https://www.kaggle.com/code/inaveed101/sft-loop-quant-onnx-int8` version 4 ran the 4D-mask kernel at `66b9b865`. Status COMPLETE. Wall 10,753 s on 4 CPUs, 33.66 GB RAM. Tarball sha256 `c823a4568de66df52483beeaa3ff6be0ee2431a8f28a02bde16943dd2d66a447`. Token grep of the outputs found 0. The 1,779,618,939 B `model.int8.onnx` is not in the tarball; `ONNX_DIR_MANIFEST.tsv` records sha256 `f2eeb25be0943b9d99533db8f123243f37dfdc9979f6fe3f7b801cb7b2e817f9`. Pins: torch 2.11.0+cpu, onnx 1.18.0, onnxruntime 1.20.1, transformers 4.56.2.

`items.json` has 0/129 `correct`, finite `nll` and `sequence_nll` on every row, and no raw generated text. Case-folded match is therefore n/a. Do not invent strings.

### Why 0/129 is not a decode-loop-only bug

The checklist claim to test was "dynamic int8 ONNX collapsed first-token exact match." A generate-loop bug (mask, past_key_values, position_ids, EOS) would leave teacher-forced NLL close to CPU fp32 while scoring 0. That is not what happened.

| arm | strict | case-folded | raw text | target NLL | sequence NLL | mean gen tokens |
| --- | --- | --- | --- | --- | --- | --- |
| CPU fp32 (#13) | 47/129 | n/a | no | 0.79975 | 4.63564 | 3.248 |
| CPU int8 (#13) | 0/129 | n/a | no | 10.03215 | 8.06883 | 7.946 |
| ONNX int8 (run 4) | 0/129 | n/a | no | 7.24983 | 6.85203 | 7.938 |

ONNX target NLL minus CPU fp32 is +6.450 on all 129 items. Sequence NLL minus CPU fp32 is +2.216 on all 129 items. ONNX is slightly less bad than torch dynamic int8 (NLL −2.782, better on 109/129). Generated lengths are `{4,6,7,8}`, not a uniform 8-token overflow. All NLL values are finite. Teacher-forced NLL is one full forward of prefix plus gold target. It does not use the greedy loop. A graph that still knew the gold word would not raise NLL on every item.

ONNX is not a pairing. `outcomes` still owns only `int8-minus-fp32` and `nf4-minus-t4-fp32`. The NLL comparison above is not a subtracted score. ONNX cites transformers 4.56.2. CPU fp32 and CPU int8 cite 5.20.2. Teacher-forced NLL still does not use the greedy loop. The collapse class still has a same-stack sibling in CPU int8 (NLL 10.03 on 5.20.2).

### Tiny-model proof of the loop

`kaggle/quant_onnx_int8/check_generate_loop.py` exports the same 4D-mask, `use_cache=False`, `dynamo=False` wrapper on `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` with the Kaggle pins (torch 2.11.0+cpu, transformers 4.56.2, onnx 1.18.0, onnxruntime 1.20.1). fp32 ONNX first-token argmax matches the torch wrapper. Teacher NLL matches to 0.0. ONNX generate ids match the wrapper loop and diverge from `model.generate` (KV cache / different mask path). The shipped loop is faithful to the exported graph. It is not a systematic mask, PKV, or position_ids bug. int8 on the tiny random model can near-tie; that is not the 1.5B story. No Kaggle rerun.

### Why dynamic int8 ONNX vs NF4

ONNX uses `onnxruntime.quantization.quantize_dynamic(..., QInt8)` on the exported graph. Torch int8 uses `default_dynamic_qconfig` on every `nn.Linear`, including `lm_head`. Both are per-tensor-class 8-bit weight quant of the output projection. Both destroy first-token exact match and raise both NLL numbers. NF4 stores each 64-wide block as a 4-bit codebook index plus a double-quantized scale, about 4.125 bits per parameter, and keeps a usable logit distribution on this task (case-folded 126/129; sequence NLL rose only +0.154).

On-disk: fp32 3,098,955,668 B, torch int8 2,477,926,394 B, ONNX int8 1,779,618,939 B, NF4 1,143,625,737 B. ONNX is more compact than torch int8 and still about 1.56× NF4. Compression and quality do not move together. The int8 collapse is the same class on two runtimes. It is not NF4 "winning at quality."

### Latency and RSS

Peak RSS is 28,993,339,392 B at `quantize_done` (0.8614 of 33,658,314,752 B). The limiter is RSS. Export plus `quantize_dynamic` keeps the fp32 graph, the int8 graph, and the ORT session in RAM. That is why 33.66 GB was required after the 16.78 GB host SIGKILL 137'd. Scoring then reuses that peak. The 1.78 GB on-disk file is not the working set.

scored p50 is 5,701.72 ms and p95 is 8,204.22 ms for a greedy pass that usually emits 8 tokens. Decode is 1.366 tok/s with `min_new_tokens=max_new_tokens=8`. The exported graph has `use_cache=False`. Each step re-runs the full growing sequence. Eight no-cache forwards of 1.5B on 4 CPUs is the limiter, not a 5–9 s token cost. cpu_time 21,471 s on 4 CPUs is about 2.0× wall. Do not pair these tok/s numbers with T4 or with torch int8 (6.50 tok/s), which uses a different generate path. The kernel does not cap ORT intra-op at protocol `threads=1`.

`129 × 2 × 7 × 5.7017247 s = 10,297.3 s`. Stored `wall_s` is `10,753.151`. That product is not an identity. `score()` also runs teacher-forced NLL. The product uses the cross-item p50, not each timed call. Using the per-item generate means times 7 for both passes is about 10,612 s, still about 141 s short of wall. Export and quantize sit before that clock. Do not treat the product as a closed check.

This is one Kaggle run. Protocol warmup is 2 and repeats are 5. The 0/129 generate score is consistent with the NLL collapse and unconfirmed as text. Run 4 `metrics.json` has `exact_match: 0.0` and `grader_case_sensitive: true` stamped after download. The `66b9b865` writer did not emit those keys or `predictions.json`. The case-sensitive compare is in that blob (`prediction_token(...) == item_id.upper()`). The next kernel writes `onnx-int8/predictions.json` so a later run can report case-folded match.

Ingest stores a 0-score ONNX arm as `measured` with `finding: collapse` when mean teacher NLL is above 2.0 (run 4 is 7.25). A 0-score arm whose mean teacher NLL stays near CPU fp32 0.80 is `unsupported`, a probable generate-loop bug. Finite NLL alone is not enough.

The top-level `*.onnx` rule is not enough. `!runs/quant/kernel/**` re-includes that directory. Later ignore rules block `*.onnx`, `*.onnx.data`, `inner.model.*`, and `onnx__*` under that tree. `scripts/verify` fails if any tracked git blob under `runs/` exceeds 5 MB. The check uses `git ls-files -s` and `git cat-file -s`, not the worktree, so a deleted checkout file cannot hide a large index blob. Run 4 left 7.11 GB of leftover external tensors next to `model.int8.onnx` (283 non-onnx files in `ONNX_DIR_MANIFEST.tsv`, including two 933,494,784 B shards). Those files were not attached. The 1.78 GB graph stayed out because `EXCLUDED_OVER_25MB.tsv` dropped the 87 rows over 25 MiB. The kernel now deletes leftover external tensors after quantize. Commit the small outputs, not the graph.

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

107 of 123 eval rows come from three filings: `3M_2018_10K` (47), `AMERICANEXPRESS_2022_10K` (30), and `BOEING_2018_10K` (30). An item bootstrap treats those 107 rows as independent coin flips. They are not. Rows from one 10-K share the same prose, the same table layout, and the same year-column habit.

The design effect applies to the per-arm hit rates, not to the paired deltas on this run. `phase4.assemble.oneway_icc` computes the one-way ICC as (MSB - MSW) / (MSB + (n0 - 1) MSW) and Kish n_eff as n / (1 + (m - 1) ρ) with m = n / k. On the run-3 paired deltas the ICC is -0.006157097732266388, so Kish n_eff is 132.1976806490039. Clustering barely inflates the variance of the deltas. The hit-level ICC is 0.20247794544213105 before training (n_eff 37.40875020997287) and 0.17951443549408733 after (n_eff 40.613989466886466). Those numbers live in `runs/phase4/scored.json` `clustering_effect` and are recomputed on every assemble.

The cluster interval still covers 0 because only 10 clusters drive the bootstrap and 3 dominate it. 403 of 10000 draws come out ≤ 0 (`cluster_ci.n_le_zero`). That is few-cluster instability, not a large Kish design effect on the deltas. `assemble.py` still sets `uplift` from `cluster_ci` and keeps `paired_ci` as the secondary item interval.

Issuer-level splits matter for the same reason. A 10-K is not a random document. The same issuer repeats language, units, and table stubs across years. The seed-0 split used issuer *strings*, so `0001628280` (JPM 2025) landed in eval while `JPM` / `0000019617` landed in train. Chunk overlap was empty, so the old check said `ok`. The model could still see JPM house style. `ISSUER_FAMILIES` is for contamination only. Do not add `0001628280` to `ISSUER_ALIASES`. That would reshuffle the locked split. Do not drop the six JPM eval rows from the main score. Report the overlap and a sensitivity that excludes them.

Family lookup must not treat a leading digit as a CIK. `3M_2018_10K` starts with `3`, so the old `doc_id[:1].isdigit()` path used the whole stem as a family and left those 47 rows ungrouped. A CIK is a 10-digit accession prefix. `0000072971` is Wells Fargo, not 3M. After that remap, contamination is still only the JPM family. The JPM sensitivity is unchanged (6 rows held out, n=117).

An exact-match gain can be format, not knowledge. The prompt asks for a bare canonical figure. A before-row that prints `$(370)$` and an after-row that prints `-370` are the same number. The old canonicalizer stripped `$` only at the start of the string, so the parentheses never flipped the sign. That hole made two already-correct figures look like misses, and it made format cleanup look like new facts. After the fix, 5 of the remaining gains still match the target figure inside extra words or units. Net new correct figures are +4, and that interval covers 0.

The next honest artifact is a proposal only. Do not run it on this GPU pass. Pre-register a filing-grouped train/eval split with no shared issuers, the cluster interval as the primary test, a format-versus-substantive breakdown, and at least two seeds. Then run that protocol.
