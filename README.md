# ORO ShoppingBench SFT on AnyCloud

Run reproducible batch inference with ORO's ShoppingBench Qwen3-4B SFT model on cloud GPUs.

This AnyCloud-powered artifact preserves the creator's public merged model and
tokenizer. Each input is a shopping-conversation prefix plus tool schemas; the
output contains the raw next assistant generation and parsed tool calls.

## Validation status

The unwrapped upstream path passed on a Lambda NVIDIA A10:
[reference evidence](validation/reference-metadata.json). The three source-derived
inputs generated complete tool calls, using 8.82 GB of peak allocated GPU memory.
Model loading and generation took 40.9 seconds, excluding provisioning, image
pulling, and Python dependency installation. GPU capacity retry delayed that run.

The portable image is being built and validated. A release command will be added
after the exact candidate digest passes the GPU and output-parity gates.

## Input and output

- Inputs: JSONL objects with unique `id`, `messages`, and `tools`; see
  [the source-derived examples](examples/shopping.jsonl).
- Outputs: `predictions.jsonl`, `summary.json`, and the supplied `inputs.jsonl`.
- Raw generations retain the model's reasoning and tool-call text. Parsing only
  reads complete `<tool_call>` blocks after the leading reasoning section.
- Tool names and arguments are checked against each input's supplied schemas.
  No tool is executed, and no ShoppingBench task-success score is claimed.
- Invalid input fails before loading weights. Generation or parsing failures are
  preserved per row and make the Job fail. There is no silent truncation.

The default uses greedy decoding and at most 512 new tokens per input. Input
prefixes exclude the target assistant turn. Custom inputs must end in a user or
tool turn; see [the custom example](examples/custom.jsonl).

## Provenance

- [ORO SFT model](https://huggingface.co/oro-ai/qwen3-4b-shoppingbench-sft), revision
  `2fbf90fc867bb7768a9e034fb8680a830a1d6373`.
- [Upstream code](https://github.com/ORO-AI/shoppingbench-trajectory-primitive),
  revision `d2dd7d0b70e8fe4494787398795502b54757ee73`.
- [Public corpus](https://huggingface.co/datasets/oro-ai/sn15-shoppingbench-sft-15k),
  revision `ca63926329b53af58b5122fab5bc24d156aa9ea7`.
- Runtime: PyTorch 2.6.0 / CUDA 12.4, Transformers 4.53.1, BF16, SDPA.

Weights download from the public creator release at runtime. Approximately
8.1 GB of model files are required. Code/model licensing is Apache-2.0; dataset
examples are CC BY 4.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Development

CPU input/parser tests require `jsonschema==4.24.0` and run with
`python -m unittest discover -s tests -v`. Image builds run on hosted Linux;
GPU validation runs on Lambda through AnyCloud.
