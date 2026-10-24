"""Download a pinned public checkpoint as inert data; never execute repo code."""
import hashlib
import json
from pathlib import Path
import urllib.request

repo = "sup-computer/gatsby-nanogpt-3"
revision = "fcefc5fbc149f9763bd79475fa9ee793abb62ee3"
expected_sha256 = "15d50d8d21ef28f6d7a67ee5e937abdcc9fd8e2fe295959a0680b591e82f5f7b"
expected_bytes = 132227430
url = f"https://huggingface.co/{repo}/resolve/{revision}/ckpt.pt"
root = Path(__file__).parent / "public-state"
root.mkdir(exist_ok=True)
path = root / "gatsby-nanogpt-3-ckpt.pt"
for attempt in range(4):
    offset = path.stat().st_size if path.exists() else 0
    if offset == expected_bytes:
        break
    assert offset < expected_bytes
    request = urllib.request.Request(url + "?download=true", headers={"Range": f"bytes={offset}-"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            if response.status == 206:
                assert response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
            else:
                assert offset == 0 and response.status == 200
            with path.open("ab") as output:
                while chunk := response.read(256 * 1024):
                    output.write(chunk)
        print("received_bytes", path.stat().st_size, flush=True)
    except (TimeoutError, OSError) as error:
        print("resume_after", type(error).__name__, path.stat().st_size, flush=True)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == expected_sha256 and path.stat().st_size == expected_bytes
manifest = {
    "repository": repo, "revision": revision, "url": url,
    "sha256": digest, "bytes": path.stat().st_size,
    "scope": "Existing public checkpoint, downloaded only; no model instantiated or training run",
}
(root / "source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2), flush=True)

# Restrict loading to data types accepted by PyTorch's weights-only loader.
import torch
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
print("top_level_keys", list(checkpoint), flush=True)
if "optimizer" in checkpoint:
    optimizer = checkpoint["optimizer"]
    print("optimizer_groups", optimizer.get("param_groups"), flush=True)
    states = optimizer.get("state", {})
    print("state_count", len(states), flush=True)
    if states:
        first = next(iter(states.values()))
        print("first_state", {k: {"shape": list(v.shape), "dtype": str(v.dtype)} if isinstance(v, torch.Tensor) else str(v) for k,v in first.items()}, flush=True)
print("iteration", checkpoint.get("iter_num"), flush=True)
