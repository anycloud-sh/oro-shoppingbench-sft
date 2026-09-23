"""Check retained CUDA evidence and exact generation parity before promotion."""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "oro-ai/qwen3-4b-shoppingbench-sft"
MODEL_REVISION = "2fbf90fc867bb7768a9e034fb8680a830a1d6373"
SOURCE_REVISION = "d2dd7d0b70e8fe4494787398795502b54757ee73"
PACKAGE = "ghcr.io/anycloud-sh/oro-shoppingbench-sft"


def verify(root, image):
    from infer import parse_tool_calls

    assert re.fullmatch(re.escape(PACKAGE) + r"@sha256:[0-9a-f]{64}", image), (
        "expected exact repository digest"
    )
    reference = {
        row["id"]: row
        for row in map(
            json.loads, (ROOT / "validation/reference.jsonl").read_text().splitlines()
        )
    }
    results = {}
    for mode, fixture in [("default", "shopping.jsonl"), ("custom", "custom.jsonl")]:
        folder = root / mode
        status = json.loads((folder / "workload.json").read_text())
        assert status["state"] == "completed" and status["cleanedAt"], (
            "compute cleanup not confirmed"
        )
        assert status["imageDigest"] == image.split("@", 1)[1], "ran another image"
        summary = json.loads((folder / "output/summary.json").read_text())
        assert summary["model"] == MODEL and summary["model_revision"] == MODEL_REVISION
        assert summary["source_revision"] == SOURCE_REVISION
        assert (
            summary["cuda_matrix_check"] == "passed"
            and summary["all_parameters_on_cuda"] is True
        )
        assert summary["gpu"] == "NVIDIA A10", "reference parity was established on A10"
        assert summary["torch"] == "2.6.0+cu124" and summary["transformers"] == "4.53.1"
        expected_bytes = (ROOT / "examples" / fixture).read_bytes()
        assert (folder / "output/inputs.jsonl").read_bytes() == expected_bytes
        assert summary["inputs_sha256"] == hashlib.sha256(expected_bytes).hexdigest()
        inputs = [json.loads(line) for line in expected_bytes.splitlines()]
        actual = [
            json.loads(line)
            for line in (folder / "output/predictions.jsonl").read_text().splitlines()
        ]
        assert (
            len(inputs)
            == len(actual)
            == summary["input_count"]
            == summary["valid_tool_call_rows"]
        )
        assert summary["failed_rows"] == 0
        assert summary["generation"] == {
            "max_new_tokens": 512,
            "do_sample": False,
            "seed": 42,
        }
        for source, prediction in zip(inputs, actual):
            assert prediction["id"] == source["id"] and prediction["error"] is None
            expected = reference[source["id"]]
            assert (
                prediction["generated_token_ids"] == expected["generated_token_ids"]
            ), "generation token mismatch"
            assert prediction["raw_text"] == expected["raw_text"], (
                "raw generation mismatch"
            )
            assert prediction["tool_calls"] == parse_tool_calls(
                prediction["raw_text"], source["tools"]
            )
            assert prediction["tool_calls"][0]["name"] == "find_product"
        results[mode] = {
            "rows": len(actual),
            "token_parity": "exact",
            "persisted_after_cleanup": True,
        }
    return results


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("image")
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence, args.image), indent=2))
