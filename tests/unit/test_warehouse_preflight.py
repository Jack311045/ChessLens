from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chesslens.warehouse.preflight import (
    DatasetPreflightError,
    register_bronze_views,
    validate_published_dataset_root,
)


def _write_part(base: Path, dataset_name: str, row_count: int) -> None:
    target = base / dataset_name / "source_month=2013-01" / "part-000000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"id": list(range(row_count))})
    pq.write_table(table, target)


def _write_manifest(
    base: Path,
    *,
    accepted_games: int,
    emitted_moves: int,
    error_records: int,
) -> None:
    payload = {
        "dataset_id": "test-dataset",
        "configuration_hash": "abc123",
        "manifest_version": "1.0.0",
        "status": "complete",
        "run_id": "test-run",
        "source": {
            "source_month": "2013-01",
            "archive_filename": "test.pgn.zst",
            "archive_sha256": "f" * 64,
        },
        "counts": {
            "accepted_games": accepted_games,
            "emitted_moves": emitted_moves,
            "error_records": error_records,
            "rejected_games": 0,
        },
        "datasets": {
            "games": {"row_count": accepted_games},
            "moves": {"row_count": emitted_moves},
            "ingestion_errors": {"row_count": error_records},
        },
    }
    (base / "_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _build_dataset_root(tmp_path: Path, *, games: int, moves: int, errors: int) -> Path:
    dataset_root = tmp_path / "dataset"
    _write_part(dataset_root, "games", games)
    _write_part(dataset_root, "moves", moves)
    _write_part(dataset_root, "ingestion_errors", errors)
    _write_manifest(
        dataset_root,
        accepted_games=games,
        emitted_moves=moves,
        error_records=errors,
    )
    return dataset_root


def _fetch_count(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    row = connection.execute(query).fetchone()
    if row is None:
        raise AssertionError("Expected one-row count query result")
    return int(row[0])


def test_validate_published_dataset_root_success(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path, games=3, moves=8, errors=0)

    result = validate_published_dataset_root(dataset_root)

    assert result.accepted_games_manifest == 3
    assert result.emitted_moves_manifest == 8
    assert result.error_records_manifest == 0
    assert result.accepted_games_physical == 3
    assert result.emitted_moves_physical == 8
    assert result.error_records_physical == 0


def test_validate_published_dataset_root_requires_complete_status(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path, games=2, moves=4, errors=0)
    manifest_path = dataset_root / "_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = "running"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetPreflightError, match="status must be 'complete'"):
        validate_published_dataset_root(dataset_root)


def test_validate_published_dataset_root_detects_count_mismatch(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path, games=4, moves=5, errors=0)
    manifest_path = dataset_root / "_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["counts"]["emitted_moves"] = 6
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetPreflightError, match="Manifest/physical row count mismatch"):
        validate_published_dataset_root(dataset_root)


def test_validate_published_dataset_root_requires_all_dataset_parts(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path, games=2, moves=3, errors=0)
    missing = dataset_root / "ingestion_errors"
    for parquet_file in missing.rglob("*.parquet"):
        parquet_file.unlink()

    with pytest.raises(DatasetPreflightError, match="Required published parquet dataset"):
        validate_published_dataset_root(dataset_root)


def test_register_bronze_views_creates_duckdb_views(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path, games=3, moves=7, errors=1)
    duckdb_path = tmp_path / "warehouse.duckdb"
    temp_directory = tmp_path / "duckdb_tmp"

    register_bronze_views(
        dataset_root=dataset_root,
        duckdb_path=duckdb_path,
        memory_limit="512MB",
        temp_directory=temp_directory,
    )

    connection = duckdb.connect(str(duckdb_path))
    try:
        games_count = _fetch_count(connection, "select count(*) from main.bronze_games")
        moves_count = _fetch_count(connection, "select count(*) from main.bronze_moves")
        errors_count = _fetch_count(connection, "select count(*) from main.bronze_ingestion_errors")
        manifest_count = _fetch_count(connection, "select count(*) from main.bronze_manifest")
    finally:
        connection.close()

    assert games_count == 3
    assert moves_count == 7
    assert errors_count == 1
    assert manifest_count == 1
