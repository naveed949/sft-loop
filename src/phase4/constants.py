"""Pins for the Phase 4 10-K extraction sibling. Not imported by sft_loop."""

from __future__ import annotations

CAPITALCITE_COMMIT = "34b42ee0b387738725705a0af682dd3f030a9944"
CAPITALCITE_REPO = "https://github.com/naveed949/capitalcite"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SMOKE_MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
SMOKE_MODEL_REVISION = "ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda"
TASK_ID = "tenk-extract-v1"
METRIC_NAME = "exact_match"
SPLIT_SEED = 0
MIN_EVAL = 100
MIN_TRAIN = 8
MIN_DISTINCT_NUMS = 3
CEILING = 0.85
NEAR_DUP_THRESHOLD = 0.8
SHINGLE_N = 5
LORA_R = 8
LORA_ALPHA = 16
LORA_SCALE = LORA_ALPHA / LORA_R
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MAX_STEPS = 200
SMOKE_STEPS = 2
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
EFFECTIVE_BATCH = PER_DEVICE_BATCH * GRAD_ACCUM
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULE = "cosine"
CHUNK_RADIUS = 700
MAX_CHUNK_CHARS = 2200
BOOTSTRAP_ROUNDS = 10000
LOCKED_SPLIT_HASH = "3a298f9e0d83b00c9d3cfac99f90a160ebef3ce28c94d926b3f5efa266c95e33"
LOCKED_EVAL_SHA256 = "09a6e33011761caf02a5eb3f732d172f778c234ed29d434af1298360d3e6a459"
LOCKED_TRAIN_SHA256 = "c73756013ef53e615cc774e8bf5e354033346e0812a325d5c72ea1f262f06abc"
BACKEND = "manual-peft"
DDP_UNSUPPORTED_REASON = (
    "kernel is a single-process main() and never initializes a torch.distributed process group"
)
VERDICT_DETAIL_NO_FILING_UPLIFT = (
    "inconclusive / no uplift at the filing level, with a positive item-level estimate"
)
ISSUER_ALIASES = {
    "0000019617": "JPM",
    "JPM": "JPM",
    "JPMORGAN": "JPM",
}
ISSUER_FAMILIES = {
    "0000019617": "JPM",
    "JPM": "JPM",
    "JPMORGAN": "JPM",
    "0001628280": "JPM",
    "0000831001": "AMEX",
    "AMERICANEXPRESS": "AMEX",
    "0000072971": "3M",
    "3M": "3M",
    "BOEING": "BOEING",
    "GS": "GS",
}


def issuer_family(issuer: str, doc_id: str = "") -> str:
    if issuer in ISSUER_FAMILIES:
        return ISSUER_FAMILIES[issuer]
    key = ""
    if doc_id[:1].isdigit():
        key = doc_id.split("-")[0]
    elif doc_id:
        key = doc_id.split("_")[0]
    return ISSUER_FAMILIES.get(key, issuer or doc_id)
