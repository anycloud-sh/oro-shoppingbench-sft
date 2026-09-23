"""Unwrapped model-card loading and native Transformers generation on a GPU."""

import base64
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "oro-ai/qwen3-4b-shoppingbench-sft"
REVISION = "2fbf90fc867bb7768a9e034fb8680a830a1d6373"
OUTPUT = Path("/mnt/output")
OUTPUT.mkdir(parents=True, exist_ok=True)
inputs_bytes = base64.b64decode(os.environ["ORO_INPUTS_B64"], validate=True)
(OUTPUT / "inputs.jsonl").write_bytes(inputs_bytes)
rows = [json.loads(line) for line in inputs_bytes.splitlines()]

assert torch.cuda.is_available(), "CUDA is required"
start = time.monotonic()
a = torch.ones((32, 32), device="cuda")
b = a @ a
torch.cuda.synchronize()
assert b.is_cuda and bool(torch.all(b == 32).item()), "CUDA matrix result mismatch"
del a, b
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="sdpa",
)
assert all(p.device.type == "cuda" for p in model.parameters())
model.eval()
loaded = time.monotonic()
results = []
for row in rows:
    chat = tokenizer.apply_chat_template(
        row["messages"], tools=row["tools"], add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to("cuda")
    before = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **chat, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    tokens = output[0, chat["input_ids"].shape[1]:].tolist()
    result = {
        "id": row["id"],
        "raw_text": tokenizer.decode(tokens, skip_special_tokens=True),
        "generated_token_ids": tokens,
        "input_tokens": chat["input_ids"].shape[1],
        "elapsed_seconds": round(time.monotonic() - before, 3),
    }
    results.append(result)
    print(json.dumps(result), flush=True)
(OUTPUT / "reference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
metadata = {
    "model": MODEL, "model_revision": REVISION,
    "dataset_revision": rows[0]["source"]["revision"],
    "source_revision": "d2dd7d0b70e8fe4494787398795502b54757ee73",
    "inputs_sha256": hashlib.sha256(inputs_bytes).hexdigest(),
    "torch": torch.__version__, "transformers": transformers.__version__,
    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    "cuda_matrix_check": "passed", "all_parameters_on_cuda": True,
    "model_load_seconds": round(loaded - start, 3),
    "total_seconds": round(time.monotonic() - start, 3),
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "generation": {"max_new_tokens": 512, "do_sample": False, "seed": 42},
}
(OUTPUT / "reference-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata), flush=True)
