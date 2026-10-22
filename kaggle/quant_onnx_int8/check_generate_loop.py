"""Compare torch vs ONNX generate and NLL on a tiny Qwen2.

Matches kaggle/quant_onnx_int8/kernel.py: 4D additive mask, dynamo=False,
greedy loop with growing input_ids, no KV cache.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Uppercase the word: anchor"
TARGET = "ANCHOR"
MAX_NEW = 8
TOKENIZER_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TOKENIZER_REV = "ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda"


def additive_causal_mask(attention_mask, dtype):
    batch, seqlen = attention_mask.shape
    causal = torch.tril(
        torch.ones((seqlen, seqlen), dtype=torch.bool, device=attention_mask.device)
    )
    keep = attention_mask.to(torch.bool)[:, None, None, :] & causal
    fill = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, seqlen, seqlen), dtype=dtype, device=attention_mask.device)
    return mask.masked_fill(~keep, fill)


def render(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


class LogitsOnly(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, attention_mask):
        mask4 = additive_causal_mask(
            attention_mask, next(self.inner.parameters()).dtype
        )
        return self.inner(
            input_ids=input_ids,
            attention_mask=mask4,
            use_cache=False,
        ).logits


def export_session(model, tokenizer, *, quantize: bool):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    wrapper = LogitsOnly(model)
    wrapper.eval()
    sample = tokenizer(render(tokenizer, PROMPT), return_tensors="pt")
    work = Path(tempfile.mkdtemp(prefix="onnx-loop-"))
    fp_path = work / "model.onnx"
    out_path = work / "model.int8.onnx" if quantize else fp_path
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    if quantize:
        quantize_dynamic(str(fp_path), str(out_path), weight_type=QuantType.QInt8)
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    return sess, out_path


def torch_generate(model, tokenizer):
    text = render(tokenizer, PROMPT)
    encoded = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return out[0][encoded["input_ids"].shape[-1] :].tolist()


def torch_wrapper_generate(model, tokenizer):
    text = render(tokenizer, PROMPT)
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    new_tokens = []
    with torch.no_grad():
        for _ in range(MAX_NEW):
            mask4 = additive_causal_mask(
                attention_mask, next(model.parameters()).dtype
            )
            logits = model(
                input_ids=input_ids,
                attention_mask=mask4,
                use_cache=False,
            ).logits
            next_id = int(logits[0, -1].argmax())
            new_tokens.append(next_id)
            next_tok = torch.tensor([[next_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)], dim=1
            )
            if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
                break
    return new_tokens


def onnx_generate(sess, tokenizer):
    text = render(tokenizer, PROMPT)
    encoded = tokenizer(text, return_tensors="np")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    new_tokens = []
    for _ in range(MAX_NEW):
        logits = sess.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        next_id = int(np.argmax(logits[0, -1]))
        new_tokens.append(next_id)
        input_ids = np.concatenate([input_ids, [[next_id]]], axis=1)
        attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
        if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
            break
    return new_tokens


def first_logit_argmax(model, sess, tokenizer):
    text = render(tokenizer, PROMPT)
    pt = tokenizer(text, return_tensors="pt")
    np_enc = tokenizer(text, return_tensors="np")
    with torch.no_grad():
        mask4 = additive_causal_mask(pt["attention_mask"], next(model.parameters()).dtype)
        torch_logits = model(
            input_ids=pt["input_ids"],
            attention_mask=mask4,
            use_cache=False,
        ).logits[0, -1]
    torch_id = int(torch_logits.argmax())
    onnx_logits = sess.run(
        ["logits"],
        {"input_ids": np_enc["input_ids"], "attention_mask": np_enc["attention_mask"]},
    )[0]
    onnx_id = int(np.argmax(onnx_logits[0, -1]))
    return torch_id, onnx_id


def teacher_nll(model, sess, tokenizer):
    prefix = render(tokenizer, PROMPT)
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"][0].tolist()
    full_pt = tokenizer(prefix + TARGET, return_tensors="pt")
    full_np = tokenizer(prefix + TARGET, return_tensors="np")
    labels = full_pt["input_ids"][0].tolist()
    gold = labels[1:]
    keep = [i for i, _ in enumerate(gold) if (i + 1) >= len(prefix_ids)]
    with torch.no_grad():
        mask4 = additive_causal_mask(
            full_pt["attention_mask"], next(model.parameters()).dtype
        )
        t_logits = model(
            input_ids=full_pt["input_ids"],
            attention_mask=mask4,
            use_cache=False,
        ).logits[0, :-1]
        t_logp = torch.log_softmax(t_logits, dim=-1)
        t_nll = float(-t_logp[keep, torch.tensor(gold)[keep]].mean()) if keep else 0.0
    o_logits = sess.run(
        ["logits"],
        {"input_ids": full_np["input_ids"], "attention_mask": full_np["attention_mask"]},
    )[0][0, :-1]
    shifted = o_logits - o_logits.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    o_nll = float(-logp[keep, np.array(gold)[keep]].mean()) if keep else 0.0
    return t_nll, o_nll


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REV)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REV,
        attn_implementation="eager",
    )
    model.eval()
    torch_ids = torch_generate(model, tokenizer)
    wrap_ids = torch_wrapper_generate(model, tokenizer)
    print("torch.generate     ", torch_ids)
    print("torch wrapper loop ", wrap_ids, "match generate", torch_ids == wrap_ids)
    for label, quantize in (("fp32-onnx", False), ("int8-onnx", True)):
        sess, path = export_session(model, tokenizer, quantize=quantize)
        onnx_ids = onnx_generate(sess, tokenizer)
        t_first, o_first = first_logit_argmax(model, sess, tokenizer)
        t_nll, o_nll = teacher_nll(model, sess, tokenizer)
        print(f"=== {label} size={path.stat().st_size} ===")
        print("onnx generate      ", onnx_ids)
        print("first-token argmax torch", t_first, "onnx", o_first, "match", t_first == o_first)
        print("teacher target nll torch", t_nll, "onnx", o_nll, "absdelta", abs(o_nll - t_nll))
        print("onnx == wrapper loop", onnx_ids == wrap_ids)
        print("onnx == torch.generate", onnx_ids == torch_ids)


if __name__ == "__main__":
    main()
