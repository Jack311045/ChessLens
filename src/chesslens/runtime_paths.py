"""Environment-based runtime path resolution helpers.

These helpers intentionally keep machine-specific locations out of tracked config by
allowing commands to resolve local paths from environment variables at runtime.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class RuntimePathResolutionError(RuntimeError):
    """Raised when environment-based path resolution is ambiguous or invalid."""


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def resolve_external_data_root_env() -> Path | None:
    """Return CHESSLENS_DATA_ROOT when configured."""
    value = _env("CHESSLENS_DATA_ROOT")
    if value is None:
        return None
    return Path(value)


def _discover_complete_collection_roots(data_root: Path) -> list[tuple[str, Path]]:
    collections_root = data_root / "processed" / "collections"
    if not collections_root.is_dir():
        return []

    discovered: list[tuple[str, Path]] = []
    for candidate in sorted(collections_root.iterdir()):
        if not candidate.is_dir():
            continue
        manifest_path = candidate / "_collection_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("status") != "complete":
            continue
        collection_id = payload.get("collection_id")
        if isinstance(collection_id, str) and collection_id.strip():
            discovered.append((collection_id.strip(), candidate))

    return discovered


def resolve_collection_root_override_or_env(collection_root_override: str | None) -> Path | None:
    """Resolve collection root from CLI override, existing env, or derived env pair.

    Precedence:
    1. CLI override
    2. CHESSLENS_COLLECTION_ROOT
    3. CHESSLENS_DATA_ROOT + CHESSLENS_COLLECTION_ID ->
       <data_root>/processed/collections/<collection_id>
    4. If CHESSLENS_DATA_ROOT is set without CHESSLENS_COLLECTION_ID:
       - exactly one complete collection -> use it
       - multiple complete collections -> raise ambiguity error
    """
    if collection_root_override is not None and collection_root_override.strip():
        return Path(collection_root_override)

    explicit_collection_root = _env("CHESSLENS_COLLECTION_ROOT")
    if explicit_collection_root is not None:
        return Path(explicit_collection_root)

    data_root = resolve_external_data_root_env()
    collection_id = _env("CHESSLENS_COLLECTION_ID")
    if data_root is None:
        return None

    if collection_id is not None:
        return data_root / "processed" / "collections" / collection_id

    complete_collections = _discover_complete_collection_roots(data_root)
    if not complete_collections:
        return None
    if len(complete_collections) == 1:
        return complete_collections[0][1]

    candidate_ids = ", ".join(
        sorted(collection_id for collection_id, _ in complete_collections)
    )
    raise RuntimePathResolutionError(
        "Ambiguous collection selection under CHESSLENS_DATA_ROOT. "
        "Set CHESSLENS_COLLECTION_ID or CHESSLENS_COLLECTION_ROOT explicitly. "
        f"Candidate complete collection IDs: {candidate_ids}"
    )


def resolve_duckdb_path_override_or_env(
    duckdb_path_override: str | None,
    *,
    default_path: str | None,
) -> Path | None:
    """Resolve DuckDB path from CLI override or env conventions."""
    if duckdb_path_override is not None and duckdb_path_override.strip():
        return Path(duckdb_path_override)

    explicit = _env("CHESSLENS_DUCKDB_PATH")
    if explicit is not None:
        return Path(explicit)

    # Compatibility alias for Tier-2 sampled warehouse snapshots.
    tier2_alias = _env("CHESSLENS_TIER2_DB_PATH")
    if tier2_alias is not None:
        return Path(tier2_alias)

    if default_path is None:
        return None
    return Path(default_path)


def resolve_warehouse_provenance_path_override_or_env(
    warehouse_provenance_path_override: str | None,
) -> Path | None:
    """Resolve warehouse provenance artifact path from CLI override or env."""
    if (
        warehouse_provenance_path_override is not None
        and warehouse_provenance_path_override.strip()
    ):
        return Path(warehouse_provenance_path_override)

    env_value = _env("CHESSLENS_WAREHOUSE_PROVENANCE_PATH")
    if env_value is None:
        return None
    return Path(env_value)


def resolve_modeling_manifest_override_or_env(
    modeling_manifest_override: str | None,
) -> Path | None:
    """Resolve Phase 2.1 modeling manifest path from CLI override or env."""
    if modeling_manifest_override is not None and modeling_manifest_override.strip():
        return Path(modeling_manifest_override)

    env_value = _env("CHESSLENS_MODELING_MANIFEST_PATH")
    if env_value is None:
        # Backward compatibility with older local runbooks.
        env_value = _env("CHESSLENS_MODELING_MANIFEST")
    if env_value is None:
        return None
    return Path(env_value)
