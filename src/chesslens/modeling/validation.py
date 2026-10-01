"""Validation helpers for modeling dataset publication and reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"JSON file not found: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON payload is not an object: {path.as_posix()}")
    return payload


def validate_manifest_identity(
    *,
    manifest: dict[str, Any],
    expected_paths: dict[tuple[str, ...], Any],
) -> None:
    issues: list[str] = []

    for path, expected_value in expected_paths.items():
        current: Any = manifest
        for key in path:
            if not isinstance(current, dict) or key not in current:
                issues.append("missing identity field " + ".".join(path))
                current = None
                break
            current = current[key]
        if current is None:
            continue
        if current != expected_value:
            issues.append(
                "identity mismatch for "
                + ".".join(path)
                + f": expected {expected_value!r}, got {current!r}"
            )

    if issues:
        raise RuntimeError(
            "Existing modeling dataset manifest identity mismatch: " + "; ".join(issues)
        )


def validate_relative_artifact_paths(dataset_root: Path, relative_paths: list[str]) -> None:
    for rel in relative_paths:
        rel_path = Path(rel)
        if rel_path.is_absolute():
            raise RuntimeError(f"Manifest contains absolute artifact path: {rel}")
        if ".." in rel_path.parts:
            raise RuntimeError(f"Manifest contains unsafe relative artifact path: {rel}")

        full_path = dataset_root / rel_path
        if not full_path.exists():
            raise RuntimeError(f"Manifest artifact path does not exist: {rel}")
