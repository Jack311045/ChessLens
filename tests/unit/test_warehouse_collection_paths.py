from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chesslens.ingestion.collection_manifest import COLLECTION_MANIFEST_FILENAME
from chesslens.warehouse.collection import validate_collection_root
from chesslens.warehouse.preflight import DatasetPreflightError


def _write_part(member_root: Path, dataset_name: str, row_count: int) -> None:
    target = member_root / dataset_name / "source_month=2013-01" / "part-000000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"id": list(range(row_count))})
    pq.write_table(table, target)


def _write_member_dataset(member_root: Path, *, games: int, moves: int, errors: int) -> None:
    _write_part(member_root, "games", games)
    _write_part(member_root, "moves", moves)
    _write_part(member_root, "ingestion_errors", errors)


def _write_collection_manifest(
    collection_root: Path,
    *,
    dataset_relpath: str,
    accepted_games: int,
    emitted_moves: int,
    error_records: int,
) -> None:
    collection_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": "1.0.0",
        "collection_id": "collection-fixture-001",
        "status": "complete",
        "parent_archive_filename": "lichess_2013_01.pgn.zst",
        "parent_archive_sha256": "f" * 64,
        "source_month": "2013-01",
        "shard_manifest_identity_hash": "abc123",
        "games_per_shard": 100,
        "counts": {
            "accepted_games": accepted_games,
            "rejected_games": 0,
            "emitted_moves": emitted_moves,
            "error_records": error_records,
        },
        "timing": {
            "started_at_utc": "2024-01-01T00:00:00+00:00",
            "updated_at_utc": "2024-01-01T00:00:01+00:00",
        },
        "shards": [
            {
                "shard_index": 0,
                "dataset_id": "dataset-a",
                "dataset_relpath": dataset_relpath,
                "start_global_index": 0,
                "end_global_index": accepted_games,
                "scanned_games": accepted_games,
                "accepted_games": accepted_games,
                "rejected_games": 0,
                "emitted_moves": emitted_moves,
                "error_records": error_records,
                "output_bytes": 0,
                "processing_active_seconds": 0.25,
                "peak_rss_bytes": 1024,
                "status": "complete",
            }
        ],
    }
    (collection_root / COLLECTION_MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_validate_collection_root_supports_windows_absolute_relocation(tmp_path: Path) -> None:
    data_root = tmp_path / "external_data"
    collection_root = data_root / "processed" / "collections" / "collection-fixture-001"
    member_relpath = Path("shards") / "000000" / "datasets" / "dataset-a"
    member_root = collection_root / member_relpath
    _write_member_dataset(member_root, games=3, moves=5, errors=1)

    stale_windows_abs = r"C:\legacy\snapshot\shards\000000\datasets\dataset-a"
    _write_collection_manifest(
        collection_root,
        dataset_relpath=stale_windows_abs,
        accepted_games=3,
        emitted_moves=5,
        error_records=1,
    )

    result = validate_collection_root(collection_root, external_data_root=data_root)

    assert result.member_count == 1
    assert result.accepted_games == 3
    assert result.emitted_moves == 5
    assert result.error_records == 1


def test_validate_collection_root_rejects_absolute_paths_without_shards(tmp_path: Path) -> None:
    collection_root = tmp_path / "processed" / "collections" / "collection-fixture-001"
    _write_collection_manifest(
        collection_root,
        dataset_relpath=r"C:\legacy\dataset-a",
        accepted_games=1,
        emitted_moves=1,
        error_records=0,
    )

    with pytest.raises(DatasetPreflightError, match="cannot be relocated"):
        validate_collection_root(collection_root)


def test_validate_collection_root_rejects_parent_traversal_segments(tmp_path: Path) -> None:
    collection_root = tmp_path / "processed" / "collections" / "collection-fixture-001"
    _write_collection_manifest(
        collection_root,
        dataset_relpath="shards/../../escape",
        accepted_games=1,
        emitted_moves=1,
        error_records=0,
    )

    with pytest.raises(DatasetPreflightError, match="parent traversal"):
        validate_collection_root(collection_root)


def test_validate_collection_root_rejects_paths_outside_external_data_root(tmp_path: Path) -> None:
    external_data_root = tmp_path / "external_data"
    collection_root = (
        tmp_path / "other_data" / "processed" / "collections" / "collection-fixture-001"
    )
    _write_collection_manifest(
        collection_root,
        dataset_relpath="shards/000000/datasets/dataset-a",
        accepted_games=1,
        emitted_moves=1,
        error_records=0,
    )

    with pytest.raises(DatasetPreflightError, match="outside CHESSLENS_DATA_ROOT"):
        validate_collection_root(collection_root, external_data_root=external_data_root)


def test_validate_collection_root_reports_missing_member_after_relocation(tmp_path: Path) -> None:
    data_root = tmp_path / "external_data"
    collection_root = data_root / "processed" / "collections" / "collection-fixture-001"
    _write_collection_manifest(
        collection_root,
        dataset_relpath="shards/000000/datasets/missing-dataset",
        accepted_games=1,
        emitted_moves=1,
        error_records=0,
    )

    with pytest.raises(DatasetPreflightError, match="missing after relocation resolution"):
        validate_collection_root(collection_root, external_data_root=data_root)
