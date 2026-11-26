# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject oversized catalogs without materializing their entire JSON document."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import skillevaluator.embedding.registry as registry_module


def test_oversized_catalog_serialization_stops_at_the_byte_budget(tmp_path: Path) -> None:
    # Measure in a separate real process so other tests/threads cannot affect
    # tracemalloc's peak. The stored vectors are valid; only their total JSON
    # size exceeds the configured catalog publication budget.
    script = r"""
import json
import tracemalloc
from pathlib import Path
from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.registry import EmbeddingRegistry, RegistryEntry
import skillevaluator.embedding.registry as registry_module

registry_module.MAX_CATALOG_BYTES = 1024 * 1024
client = EmbeddingClient(model="test-model", api_key="test-only", base_url="https://example.com/v1")
registry = EmbeddingRegistry(client)
vector = [1.0 / 7.0] * 8192
for index in range(64):
    path = f"skill-{index:03}"
    key = f"skill:{path}"
    registry._entries[key] = RegistryEntry(
        entry_id=key, name=path, description="A catalog skill", path=path,
        content_type="skill", content_fingerprint="a" * 64, embedding=vector,
    )
results = []
for existing in (False, True):
    target = Path("existing.json" if existing else "new.json")
    if existing:
        target.write_bytes(b"previous catalog\n")
    tracemalloc.start()
    try:
        registry.save_catalog(target)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("Oversized catalog was accepted")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results.append({
        "peak": peak, "error": message,
        "content": target.read_text() if target.exists() else None,
    })
print(json.dumps({"results": results, "files": sorted(p.name for p in Path.cwd().iterdir())}))
"""
    source_root = str(Path(registry_module.__file__).resolve().parents[2])
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": source_root},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    data = json.loads(process.stdout)

    assert [result["error"] for result in data["results"]] == [
        "Catalog size limit exceeded (1048576 bytes)",
        "Catalog size limit exceeded (1048576 bytes)",
    ]
    assert [result["content"] for result in data["results"]] == [None, "previous catalog\n"]
    assert data["files"] == ["existing.json"]
    # The 10 MiB document needs over 19 MiB if serialized in one operation.
    # Allow ample headroom above the 1 MiB cap for one vector and validation.
    assert max(result["peak"] for result in data["results"]) < 8 * 1024 * 1024
