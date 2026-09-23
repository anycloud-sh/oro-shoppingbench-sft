"""Validate an immutable candidate through the selected AnyCloud API."""

import argparse
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cli(*args, check=True, capture=False):
    return subprocess.run(
        ["anycloud", *args], check=check, capture_output=capture, text=True, timeout=180
    )


def status(job):
    data = json.loads(cli("status", job, "--json", capture=True).stdout)
    # Support the current Workload field and the pinned v0.1.64 release.
    return data.get("workload") or data["deployment"]


def wait(job, cleanup=False, timeout=2400):
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        state = status(job)
        if state["state"] != previous:
            print(job, state["state"], flush=True)
            previous = state["state"]
        if (
            state.get("cleanedAt")
            if cleanup
            else state["state"]
            in {"completed", "failed", "errored", "invalid", "terminated"}
        ):
            return state
        time.sleep(15)
    raise TimeoutError(
        f"{job}: timeout awaiting {'cleanup' if cleanup else 'completion'}"
    )


def cleanup(ledger):
    if not ledger.exists():
        return
    resource = json.loads(ledger.read_text())
    for job in resource["jobs"]:
        listed = json.loads(cli("list", "--json", "--filter", job, capture=True).stdout)
        if any(row["id"] == job for row in listed):
            cli("terminate", job)
            wait(job, cleanup=True, timeout=600)
    for bucket in resource["buckets"]:
        cli(
            "bucket",
            "delete",
            bucket,
            "--credentials",
            "artifact-storage",
            "--region",
            "us-east-1",
        )


def run(args):
    image = args.image
    if not re.fullmatch(
        r"ghcr.io/anycloud-sh/oro-shoppingbench-sft@sha256:[0-9a-f]{64}", image
    ):
        raise ValueError("An exact ORO candidate digest is required")
    out = args.evidence.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "resources.json"
    if ledger.exists():
        raise ValueError("Evidence directory already has a resource ledger")
    suffix = uuid.uuid4().hex[:12]
    resource = {"jobs": [], "buckets": []}

    def record():
        ledger.write_text(json.dumps(resource, indent=2) + "\n")

    record()
    try:
        input_bucket = "anycloud-oro-input-" + suffix
        resource["buckets"].append(input_bucket)
        record()
        cli(
            "bucket",
            "create",
            input_bucket,
            "--credentials",
            "artifact-storage",
            "--region",
            "us-east-1",
        )
        cli(
            "bucket",
            "upload",
            input_bucket,
            str(ROOT / "examples/custom.jsonl"),
            "queries.jsonl",
            "--credentials",
            "artifact-storage",
            "--region",
            "us-east-1",
        )
        for mode in ["default", "custom"]:
            job = "oro-" + mode + "-" + suffix
            output_bucket = "anycloud-" + job
            folder = out / mode
            folder.mkdir()
            resource["jobs"].append(job)
            resource["buckets"].append(output_bucket)
            record()
            command = [
                "job",
                image,
                "--id",
                job,
                "--credentials",
                "lambda",
                "--gpu-type",
                "a10",
                "--gpus",
                "all",
                "--output-bucket",
                output_bucket,
                "--output-storage-credentials",
                "artifact-storage",
                "--output-storage-region",
                "us-east-1",
            ]
            if mode == "custom":
                command += [
                    "--input-bucket",
                    input_bucket,
                    "--input-storage-credentials",
                    "artifact-storage",
                    "--input-storage-region",
                    "us-east-1",
                    "--",
                    "--input",
                    "/mnt/input/queries.jsonl",
                ]
            (folder / "command.json").write_text(
                json.dumps(["anycloud", *command], indent=2) + "\n"
            )
            cli(*command)
            result = wait(job)
            logs = cli("logs", job, capture=True, check=False)
            (folder / "job.log").write_text(logs.stdout + logs.stderr)
            result = wait(job, cleanup=True, timeout=600)
            public = {
                key: result.get(key)
                for key in [
                    "id",
                    "state",
                    "image",
                    "imageDigest",
                    "completedAt",
                    "cleanedAt",
                ]
            }
            (folder / "workload.json").write_text(json.dumps(public, indent=2) + "\n")
            cli(
                "bucket",
                "download",
                output_bucket,
                "",
                str(folder / "output"),
                "--recursive",
                "--credentials",
                "artifact-storage",
                "--region",
                "us-east-1",
            )
            if result["state"] != "completed":
                raise RuntimeError(f"{job} ended {result['state']}")
        import sys

        sys.path.insert(0, str(ROOT))
        from verify import verify

        checks = verify(out, image)
        (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    finally:
        cleanup(ledger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if args.cleanup:
        cleanup(args.evidence / "resources.json")
    else:
        run(args)
