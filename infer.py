"""AnyCloud-powered inference of ORO's public merged SFT model."""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from jsonschema import Draft202012Validator

MODEL = "oro-ai/qwen3-4b-shoppingbench-sft"
MODEL_REVISION = "2fbf90fc867bb7768a9e034fb8680a830a1d6373"
SOURCE_REVISION = "d2dd7d0b70e8fe4494787398795502b54757ee73"
DATASET_REVISION = "ca63926329b53af58b5122fab5bc24d156aa9ea7"
DEFAULT_INPUTS = Path(__file__).parent / "examples/shopping.jsonl"


def tool_definitions(tools):
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a non-empty list")
    definitions = {}
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("each tool must be an object")
        definition = tool.get("function", tool)
        if not isinstance(definition, dict):
            raise ValueError("tool function must be an object")
        name = definition.get("name")
        if not isinstance(name, str) or not name or name in definitions:
            raise ValueError("tool names must be unique non-empty strings")
        parameters = definition.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("tool parameters must be a JSON schema object")
        Draft202012Validator.check_schema(parameters)
        definitions[name] = parameters
    return definitions


def read_inputs(path):
    raw = Path(path).read_bytes()
    rows = []
    ids = set()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {number}: expected an object")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError(f"line {number}: id must be a unique non-empty string")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{identifier}: messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system",
                "user",
                "assistant",
                "tool",
            }:
                raise ValueError(f"{identifier}: invalid message role")
            if message.get("content") is not None and not isinstance(
                message["content"], str
            ):
                raise ValueError(f"{identifier}: message content must be text or null")
        if messages[-1]["role"] not in {"user", "tool"}:
            raise ValueError(
                f"{identifier}: prefix must end before the target assistant turn"
            )
        tool_definitions(row.get("tools"))
        ids.add(identifier)
        rows.append(row)
    if not rows:
        raise ValueError("input contains no examples")
    return rows, raw


def parse_tool_calls(raw_text, tools):
    definitions = tool_definitions(tools)
    text = raw_text
    if text.startswith("<think>"):
        if "</think>" not in text:
            raise ValueError("unterminated reasoning section")
        text = text.split("</think>", 1)[1]
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not blocks:
        raise ValueError("no complete tool call generated")
    if text.count("<tool_call>") != len(blocks) or text.count("</tool_call>") != len(
        blocks
    ):
        raise ValueError("incomplete tool-call delimiters")
    calls = []
    for block in blocks:
        call = json.loads(block)
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError("tool call must have a name and arguments object")
        if call["name"] not in definitions:
            raise ValueError(f"unknown tool: {call['name']}")
        if not isinstance(call.get("arguments"), dict):
            raise ValueError("tool-call arguments must be an object")
        Draft202012Validator(definitions[call["name"]]).validate(call["arguments"])
        calls.append(call)
    return calls


def run(args):
    rows, raw_inputs = read_inputs(args.input)
    # Input errors fail before GPU allocation or weight downloads.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    started = time.monotonic()
    a = torch.ones((32, 32), device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    if not b.is_cuda or not bool(torch.all(b == 32).item()):
        raise RuntimeError("CUDA matrix check failed")
    del a, b
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    if not all(p.device.type == "cuda" for p in model.parameters()):
        raise RuntimeError("Model parameters were not all loaded onto CUDA")
    model.eval()
    loaded = time.monotonic()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs.jsonl").write_bytes(raw_inputs)
    successful = 0
    with (out / "predictions.jsonl").open("w") as stream:
        for row in rows:
            result = {
                "id": row["id"],
                "raw_text": None,
                "tool_calls": [],
                "error": None,
            }
            try:
                chat = tokenizer.apply_chat_template(
                    row["messages"],
                    tools=row["tools"],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to("cuda")
                if (
                    chat["input_ids"].shape[1] + args.max_new_tokens
                    > model.config.max_position_embeddings
                ):
                    raise ValueError(
                        "input plus generation budget exceeds the model context window"
                    )
                before = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(
                        **chat,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                torch.cuda.synchronize()
                tokens = generated[0, chat["input_ids"].shape[1] :].tolist()
                result.update(
                    {
                        "raw_text": tokenizer.decode(tokens, skip_special_tokens=True),
                        "generated_token_ids": tokens,
                        "input_tokens": chat["input_ids"].shape[1],
                        "elapsed_seconds": round(time.monotonic() - before, 3),
                    }
                )
                if not tokens or tokens[-1] != tokenizer.eos_token_id:
                    raise ValueError(
                        "generation ended without EOS; increase --max-new-tokens"
                    )
                result["tool_calls"] = parse_tool_calls(
                    result["raw_text"], row["tools"]
                )
                successful += 1
            except Exception as error:
                # Retain the failing row and raw generation; the process fails
                # after writing the batch so failures never become silent skips.
                result["error"] = f"{type(error).__name__}: {error}"
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "dataset_revision": (
            DATASET_REVISION
            if all(
                row.get("source", {}).get("dataset")
                == "oro-ai/sn15-shoppingbench-sft-15k"
                and row["source"].get("revision") == DATASET_REVISION
                for row in rows
            )
            else None
        ),
        "inputs_sha256": hashlib.sha256(raw_inputs).hexdigest(),
        "input_count": len(rows),
        "valid_tool_call_rows": successful,
        "failed_rows": len(rows) - successful,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_matrix_check": "passed",
        "all_parameters_on_cuda": True,
        "model_load_seconds": round(loaded - started, 3),
        "total_seconds": round(time.monotonic() - started, 3),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "seed": 42,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if successful == len(rows) else 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate the next ShoppingBench SFT tool call for each conversation prefix."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/output"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 2048:
        parser.error("--max-new-tokens must be between 1 and 2048")
    try:
        return run(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
