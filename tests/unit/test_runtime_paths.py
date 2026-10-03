from __future__ import annotations

import json
from pathlib import Path

import pytest

from chesslens.runtime_paths import (
    RuntimePathResolutionError,
    resolve_collection_root_override_or_env,
    resolve_duckdb_path_override_or_env,
    resolve_external_data_root_env,
    resolve_modeling_manifest_override_or_env,
    resolve_warehouse_provenance_path_override_or_env,
)

RUNTIME_PATH_ENV_VARS: tuple[str, ...] = (
    "CHESSLENS_COLLECTION_ID",
    "CHESSLENS_COLLECTION_ROOT",
    "CHESSLENS_DATA_ROOT",
    "CHESSLENS_DUCKDB_PATH",
    "CHESSLENS_MODELING_MANIFEST_PATH",
    "CHESSLENS_MODELING_MANIFEST",
    "CHESSLENS_TIER2_DB_PATH",
    "CHESSLENS_WAREHOUSE_PROVENANCE_PATH",
)


def _clear_runtime_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in RUNTIME_PATH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _write_complete_collection_manifest(data_root: Path, collection_id: str) -> None:
    collection_root = data_root / "processed" / "collections" / collection_id
    collection_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection_id": collection_id,
        "status": "complete",
        "counts": {
            "accepted_games": 1,
            "emitted_moves": 1,
            "error_records": 0,
        },
        "shards": [],
    }
    manifest_path = collection_root / "_collection_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def test_collection_root_resolves_from_data_root_and_collection_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_path_env(monkeypatch)
    data_root = tmp_path / "external_data"
    monkeypatch.setenv("CHESSLENS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("CHESSLENS_COLLECTION_ID", "collection-abc")

    resolved = resolve_collection_root_override_or_env(None)

    assert resolved == data_root / "processed" / "collections" / "collection-abc"


def test_collection_root_raises_ambiguity_when_multiple_complete_collections_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_path_env(monkeypatch)
    data_root = tmp_path / "external_data"
    _write_complete_collection_manifest(data_root, "collection-a")
    _write_complete_collection_manifest(data_root, "collection-b")
    monkeypatch.setenv("CHESSLENS_DATA_ROOT", str(data_root))

    with pytest.raises(RuntimePathResolutionError, match="Candidate complete collection IDs"):
        resolve_collection_root_override_or_env(None)


def test_collection_root_resolves_single_complete_collection_without_collection_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_path_env(monkeypatch)
    data_root = tmp_path / "external_data"
    collection_id = "collection-unique"
    _write_complete_collection_manifest(data_root, collection_id)
    monkeypatch.setenv("CHESSLENS_DATA_ROOT", str(data_root))

    resolved = resolve_collection_root_override_or_env(None)

    assert resolved == data_root / "processed" / "collections" / collection_id


def test_collection_root_override_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_path_env(monkeypatch)
    cli_override = tmp_path / "cli_collection"
    explicit_env_root = tmp_path / "env_collection"
    data_root = tmp_path / "external_data"

    monkeypatch.setenv("CHESSLENS_COLLECTION_ROOT", str(explicit_env_root))
    monkeypatch.setenv("CHESSLENS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("CHESSLENS_COLLECTION_ID", "collection-abc")

    resolved = resolve_collection_root_override_or_env(str(cli_override))

    assert resolved == cli_override


def test_duckdb_path_resolution_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_path_env(monkeypatch)

    assert resolve_duckdb_path_override_or_env(None, default_path=None) is None

    tier2 = tmp_path / "tier2.duckdb"
    monkeypatch.setenv("CHESSLENS_TIER2_DB_PATH", str(tier2))
    assert resolve_duckdb_path_override_or_env(None, default_path=None) == tier2

    explicit = tmp_path / "explicit.duckdb"
    monkeypatch.setenv("CHESSLENS_DUCKDB_PATH", str(explicit))
    assert resolve_duckdb_path_override_or_env(None, default_path=None) == explicit

    _clear_runtime_path_env(monkeypatch)
    assert resolve_duckdb_path_override_or_env(
        None,
        default_path="data/tmp/warehouse.duckdb",
    ) == Path(
        "data/tmp/warehouse.duckdb",
    )


def test_modeling_and_provenance_path_env_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_path_env(monkeypatch)
    modeling_manifest = tmp_path / "modeling_manifest.json"
    warehouse_provenance = tmp_path / "warehouse_provenance.json"

    monkeypatch.setenv("CHESSLENS_MODELING_MANIFEST_PATH", str(modeling_manifest))
    monkeypatch.setenv("CHESSLENS_WAREHOUSE_PROVENANCE_PATH", str(warehouse_provenance))

    assert resolve_modeling_manifest_override_or_env(None) == modeling_manifest
    assert resolve_warehouse_provenance_path_override_or_env(None) == warehouse_provenance


def test_external_data_root_ignores_blank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_path_env(monkeypatch)
    monkeypatch.setenv("CHESSLENS_DATA_ROOT", "   ")

    assert resolve_external_data_root_env() is None
