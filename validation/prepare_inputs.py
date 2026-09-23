"""Select three distinct, source-derived prefixes without assistant targets."""

import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

DATASET = "oro-ai/sn15-shoppingbench-sft-15k"
REVISION = "ca63926329b53af58b5122fab5bc24d156aa9ea7"
FILENAME = "oro_sft_eval_v0.jsonl"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILENAME}"


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    with urlopen(URL, timeout=90) as response:
        for index, line in enumerate(response):
            source = json.loads(line)
            end = next(i for i, m in enumerate(source["messages"]) if m["role"] == "assistant")
            messages = source["messages"][:end]
            query = next(m["content"] for m in messages if m["role"] == "user")
            if query in seen:
                continue
            seen.add(query)
            rows.append({
                "id": f"oro-eval-row-{index}",
                "messages": messages,
                "tools": source["tools"],
                "source": {
                    "dataset": DATASET,
                    "revision": REVISION,
                    "file": FILENAME,
                    "row_index": index,
                    "row_sha256": hashlib.sha256(line).hexdigest(),
                    "prefix_end_exclusive": end,
                },
            })
            if len(rows) == 3:
                break
    if len(rows) != 3:
        raise RuntimeError("Expected three distinct public shopping prompts")
    (out / "inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"selected": [r["id"] for r in rows], "dataset_revision": REVISION}))


if __name__ == "__main__":
    main()
