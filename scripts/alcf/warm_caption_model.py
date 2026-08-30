"""One-time warmup: download Qwen/Qwen3-VL-8B-Instruct (processor + weights)
into HF_HOME so a later multi-rank job finds it already cached instead of 8
ranks racing to download the same large repo simultaneously (exactly the
class of race that hit the V-JEPA torch.hub cache earlier).

CPU-only -- this just needs the files on disk, not a working GPU model.
See scripts/alcf/sjob_warm_caption_model.
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

print(f"=== warming processor: {MODEL_NAME} ===", flush=True)
AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"=== warming model: {MODEL_NAME} ===", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cpu")
model.eval()

print("WARM_DONE", flush=True)
