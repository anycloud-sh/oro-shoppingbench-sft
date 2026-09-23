# ORO ShoppingBench SFT on AnyCloud

Run reproducible batch inference with ORO's ShoppingBench Qwen3-4B SFT model on cloud GPUs.

This AnyCloud-powered Job uses the [public merged model](https://huggingface.co/oro-ai/qwen3-4b-shoppingbench-sft).
It generates the next assistant turn from shopping-conversation prefixes and
returns the raw text plus parsed tool calls. ORO owns the upstream model.

## Run

Requires a configured AnyCloud API, a Lambda credential named `lambda`, and an
AWS storage credential named `artifact-storage`. Use your saved credential names
and storage region if they differ. This creates a uniquely named output Bucket:

```bash
export ORO_OUTPUT="oro-sft-$(date +%s)-$RANDOM"
anycloud job ghcr.io/anycloud-sh/oro-shoppingbench-sft:0.1.0-2fbf90f-r1 \
  --credentials lambda --gpu-type a10 --gpus all \
  --output-bucket "$ORO_OUTPUT" \
  --output-storage-credentials artifact-storage --output-storage-region us-east-1
```

After the Job completes, download its concrete predictions:

```bash
anycloud bucket download "$ORO_OUTPUT" predictions.jsonl ./predictions.jsonl \
  --credentials artifact-storage --region us-east-1
```

Three [source-derived inputs](examples/shopping.jsonl) produce `predictions.jsonl`,
`summary.json`, and a copy of `inputs.jsonl`. The default dental-elastics prompt
produces a `find_product` call. [Read the actual outputs](validation/lambda-a10/default/output/predictions.jsonl).
The Bucket retains these files after compute cleanup. Reruns start from the
beginning; this short inference Job does not save training checkpoints.

## Use your own input

From a checkout of this repository, use the included custom example or edit it
first with your own JSONL conversation prefixes and tool schemas:

```bash
export ORO_INPUT="${ORO_OUTPUT}-input"
anycloud bucket create "$ORO_INPUT" --credentials artifact-storage --region us-east-1
anycloud bucket upload "$ORO_INPUT" examples/custom.jsonl queries.jsonl \
  --credentials artifact-storage --region us-east-1
anycloud job ghcr.io/anycloud-sh/oro-shoppingbench-sft:0.1.0-2fbf90f-r1 \
  --credentials lambda --gpu-type a10 --gpus all \
  --input-bucket "$ORO_INPUT" \
  --input-storage-credentials artifact-storage --input-storage-region us-east-1 \
  --output-bucket "${ORO_OUTPUT}-custom" \
  --output-storage-credentials artifact-storage --output-storage-region us-east-1 \
  -- --input /mnt/input/queries.jsonl
```

Each row needs a unique `id`, `messages` ending in a user/tool turn, and `tools`.
The default is greedy decoding with 512 new tokens; `--max-new-tokens` accepts
1–2048. Invalid input fails before loading weights. Per-row generation/parsing
errors are retained and fail the Job. Raw text is preserved; parsing reads
complete `<tool_call>` blocks after the leading reasoning section. Schema
validity checks names and argument types; calls are not executed and are not
scored for shopping correctness.

## Validation and requirements

Validated with AnyCloud 0.1.64 on Lambda A10 (24 GB), Linux/amd64. Default and
custom-input outputs match the unwrapped reference token-for-token and survive
compute cleanup. [Evidence and checks](validation/lambda-a10/checks.json).
The default run took **36.5 seconds** for model loading and generation and used
**8.82 GB** peak allocated GPU memory. Provisioning and image pulling add time.
Cold downloads: approximately 3.35 GB of image layers and 8.1 GB of model files.

Release digest: `sha256:fe0609eead68e26d567543bfad135324eebfcb8607f5bee4f859fae5a57075d3`.
Model revision: `2fbf90fc867bb7768a9e034fb8680a830a1d6373`;
[source](https://github.com/ORO-AI/shoppingbench-trajectory-primitive) revision:
`d2dd7d0b70e8fe4494787398795502b54757ee73`.
Runtime: PyTorch 2.6.0 / CUDA 12.4, Transformers 4.53.1, BF16, SDPA.
Weights download from ORO at runtime. Code/model: Apache-2.0; corpus examples:
CC BY 4.0. Revisions, attribution, and input adaptations are in [NOTICE](NOTICE).
CPU checks: install `jsonschema==4.24.0`, then `python -m unittest discover -s tests -v`.
