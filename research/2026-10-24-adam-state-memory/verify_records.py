#!/usr/bin/env python3
"""Read-only checks of the lightweight published records. Does not run experiments."""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent

def verify():
    expected = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    actual = {str(p.relative_to(ROOT)): p for p in ROOT.rglob("*")
              if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
    assert set(actual) == set(expected), "File list differs from SHA256SUMS"
    for name, path in actual.items():
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected[name], name
        text = data.decode("utf-8")
        assert "\0" not in text, name
        if path.suffix == ".json":
            json.loads(text)
        if path.suffix == ".py":
            ast.parse(text, filename=name)
    provenance = json.loads((ROOT / "PUBLICATION_PROVENANCE.json").read_text())
    for item in provenance["files"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["published_sha256"], item["path"]
    summary = json.loads((ROOT / "summary.json").read_text())
    assert summary["strict_long_run_fp32_bitwise_under_30_gib"]["achieved"] is False
    assert summary["publication_reran_experiments"] is False
    assert summary["budget"]["source_derived_text_parameters"] == 26895998464
    assert not summary["approximate_candidate"]["target_device_peak_proven"]
    assert not summary["finite_factor_replay"]["long_run_fixed_memory_solution"]
    assert summary["stage13"]["actual_states_at_target_fraction"] == 0
    for name in ["README.md", "REPRODUCE.md", "ARTIFACT_INDEX.md"]:
        text = (ROOT / name).read_text()
        for href in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if "://" not in href and not href.startswith("#"):
                assert (ROOT / href.split("#", 1)[0]).exists(), (name, href)
    print(json.dumps({"status": "passed", "verified_files": len(actual),
                      "scope": "Hashes, UTF-8, JSON, Python syntax, publication provenance, key conclusion flags and entrypoint links only; not a numerical experiment rerun."}, ensure_ascii=False))

if __name__ == "__main__":
    verify()
