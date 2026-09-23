"""Verify public pulling and repository identity for the exact candidate."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

image, artifact_revision, output = sys.argv[1:]
package = "anycloud-sh/oro-shoppingbench-sft"
assert re.fullmatch(re.escape("ghcr.io/" + package) + r"@sha256:[0-9a-f]{64}", image)
digest = image.split("@", 1)[1]
with urlopen(
    f"https://ghcr.io/token?service=ghcr.io&scope=repository:{package}:pull", timeout=60
) as response:
    token = json.load(response)["token"]
headers = {
    "Authorization": "Bearer " + token,
    "Accept": "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json",
}
with urlopen(
    Request(f"https://ghcr.io/v2/{package}/manifests/{digest}", headers=headers),
    timeout=60,
) as response:
    raw = response.read()
assert "sha256:" + hashlib.sha256(raw).hexdigest() == digest
manifest = json.loads(raw)
config_digest = manifest["config"]["digest"]
with urlopen(
    Request(f"https://ghcr.io/v2/{package}/blobs/{config_digest}", headers=headers),
    timeout=60,
) as response:
    config = json.load(response)
labels = config["config"]["Labels"]
repository = "https://github.com/" + package
assert labels["org.opencontainers.image.source"] == repository
assert labels["org.opencontainers.image.revision"] == artifact_revision
metadata_headers = {
    "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
    "Accept": "application/vnd.github+json",
    "User-Agent": "oro-artifact-validation",
}
with urlopen(
    Request(
        "https://api.github.com/orgs/anycloud-sh/packages/container/oro-shoppingbench-sft",
        headers=metadata_headers,
    ),
    timeout=60,
) as response:
    metadata = json.load(response)
assert metadata["visibility"] == "public"
assert metadata["repository"]["html_url"] == repository
# A fresh Docker config proves the pull uses no saved registry credential.
with tempfile.TemporaryDirectory() as directory:
    subprocess.run(
        ["docker", "--config", directory, "pull", "--platform", "linux/amd64", image],
        check=True,
        timeout=900,
    )
result = {
    "image": image,
    "artifact_revision": artifact_revision,
    "anonymous_pull": True,
    "repository_linked": True,
    "repository": repository,
    "platform": "linux/amd64",
}
Path(output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
