"""Environment-based runtime path resolution helpers.

These helpers intentionally keep machine-specific locations out of tracked config by
allowing commands to resolve local paths from environment variables at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path


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


def resolve_collection_root_override_or_env(collection_root_override: str | None) -> Path | None:
    """Resolve collection root from CLI override, existing env, or derived env pair.

    Precedence:
    1. CLI override
    2. CHESSLENS_COLLECTION_ROOT
    3. CHESSLENS_DATA_ROOT + CHESSLENS_COLLECTION_ID ->
       <data_root>/processed/collections/<collection_id>
    """
    if collection_root_override is not None and collection_root_override.strip():
        return Path(collection_root_override)

    explicit_collection_root = _env("CHESSLENS_COLLECTION_ROOT")
    if explicit_collection_root is not None:
        return Path(explicit_collection_root)

    data_root = resolve_external_data_root_env()
    collection_id = _env("CHESSLENS_COLLECTION_ID")
    if data_root is None or collection_id is None:
        return None

    return data_root / "processed" / "collections" / collection_id


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
