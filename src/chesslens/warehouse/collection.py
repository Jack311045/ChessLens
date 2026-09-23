"""Collection-aware DuckDB bronze registration for the dbt warehouse.

A collection unions only the dataset roots explicitly listed in its manifest (never
a broad filesystem glob), so old datasets, fixtures, reruns, or unrelated months can
never leak in. The synthesized ``bronze_manifest`` mirrors the single-dataset shape
so the existing dbt models and tests run unchanged over the unioned data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from chesslens.ingestion.collection_manifest import (
    COLLECTION_MANIFEST_FILENAME,
    load_collection_manifest,
    parse_collection_entries,
    validate_collection_reconciliation,
)
from chesslens.warehouse.preflight import DatasetPreflightError


@dataclass(frozen=True)
class CollectionPreflightResult:
    collection_root: Path
    manifest_path: Path
    member_count: int
    accepted_games: int
    emitted_moves: int
    error_records: int


def _member_parquet_files(member_root: Path, dataset_name: str) -> list[Path]:
    files = sorted((member_root / dataset_name).glob("source_month=*/part-*.parquet"))
    if not files:
        raise DatasetPreflightError(
            f"Collection member missing {dataset_name} parquet: {member_root.as_posix()}"
        )
    return files


def _sql_file_list(paths: list[str]) -> str:
    escaped = [p.replace("'", "''") for p in paths]
    return "[" + ", ".join(f"'{p}'" for p in escaped) + "]"


def validate_collection_root(collection_root: Path) -> CollectionPreflightResult:
    if not collection_root.is_dir():
        raise DatasetPreflightError(
            f"CHESSLENS_COLLECTION_ROOT must be a directory: {collection_root.as_posix()}"
        )
    manifest_path = collection_root / COLLECTION_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise DatasetPreflightError(
            f"Collection manifest not found: {manifest_path.as_posix()}"
        )
    manifest = load_collection_manifest(manifest_path)
    validate_collection_reconciliation(manifest, require_complete=True)

    entries = parse_collection_entries(manifest)
    games_files: list[str] = []
    moves_files: list[str] = []
    errors_files: list[str] = []
    for entry in entries:
        member_root = collection_root / entry.dataset_relpath
        if not member_root.is_dir():
            raise DatasetPreflightError(
                f"Collection member dataset missing: {member_root.as_posix()}"
            )
        games_files += [p.resolve().as_posix() for p in _member_parquet_files(member_root, "games")]
        moves_files += [p.resolve().as_posix() for p in _member_parquet_files(member_root, "moves")]
        errors_files += [
            p.resolve().as_posix()
            for p in _member_parquet_files(member_root, "ingestion_errors")
        ]

    counts = manifest["counts"]
    connection = duckdb.connect(database=":memory:")
    try:
        physical_games = _count(connection, games_files)
        physical_moves = _count(connection, moves_files)
        physical_errors = _count(connection, errors_files)
    finally:
        connection.close()

    if physical_games != int(counts["accepted_games"]):
        raise DatasetPreflightError("Collection games row count does not match manifest")
    if physical_moves != int(counts["emitted_moves"]):
        raise DatasetPreflightError("Collection moves row count does not match manifest")
    if physical_errors != int(counts["error_records"]):
        raise DatasetPreflightError("Collection error row count does not match manifest")

    return CollectionPreflightResult(
        collection_root=collection_root,
        manifest_path=manifest_path,
        member_count=len(entries),
        accepted_games=physical_games,
        emitted_moves=physical_moves,
        error_records=physical_errors,
    )


def _count(connection: duckdb.DuckDBPyConnection, files: list[str]) -> int:
    row = connection.execute(
        f"SELECT COUNT(*) FROM read_parquet({_sql_file_list(files)})"
    ).fetchone()
    if row is None:
        raise DatasetPreflightError("Unable to count collection parquet rows")
    return int(row[0])


def register_collection_bronze_views(
    *,
    collection_root: Path,
    duckdb_path: Path,
    memory_limit: str,
    temp_directory: Path,
) -> CollectionPreflightResult:
    result = validate_collection_root(collection_root)
    manifest = load_collection_manifest(result.manifest_path)
    entries = parse_collection_entries(manifest)

    games_files: list[str] = []
    moves_files: list[str] = []
    errors_files: list[str] = []
    for entry in entries:
        member_root = collection_root / entry.dataset_relpath
        games_files += [p.resolve().as_posix() for p in _member_parquet_files(member_root, "games")]
        moves_files += [p.resolve().as_posix() for p in _member_parquet_files(member_root, "moves")]
        errors_files += [
            p.resolve().as_posix()
            for p in _member_parquet_files(member_root, "ingestion_errors")
        ]

    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)

    synthesized_manifest = _synthesized_manifest_dict(manifest, result)
    manifest_json_path = temp_directory / "collection_bronze_manifest.json"
    manifest_json_path.write_text(json.dumps(synthesized_manifest), encoding="utf-8")

    connection = duckdb.connect(database=str(duckdb_path))
    try:
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(
            f"SET temp_directory = '{temp_directory.resolve().as_posix()}'"
        )
        connection.execute(
            "CREATE OR REPLACE VIEW main.bronze_games AS "
            f"SELECT * FROM read_parquet({_sql_file_list(games_files)}, filename = true)"
        )
        connection.execute(
            "CREATE OR REPLACE VIEW main.bronze_moves AS "
            f"SELECT * FROM read_parquet({_sql_file_list(moves_files)}, filename = true)"
        )
        connection.execute(
            "CREATE OR REPLACE VIEW main.bronze_ingestion_errors AS "
            f"SELECT * FROM read_parquet({_sql_file_list(errors_files)}, filename = true)"
        )
        manifest_json_sql = manifest_json_path.resolve().as_posix().replace("'", "''")
        connection.execute(
            "CREATE OR REPLACE VIEW main.bronze_manifest AS "
            f"SELECT * FROM read_json_auto('{manifest_json_sql}')"
        )
    finally:
        connection.close()

    return result


def _synthesized_manifest_dict(
    manifest: dict[str, Any],
    result: CollectionPreflightResult,
) -> dict[str, Any]:
    now = datetime.now(tz=UTC).isoformat(timespec="seconds")
    counts = manifest["counts"]
    return {
        "dataset_id": manifest["collection_id"],
        "configuration_hash": manifest.get("shard_manifest_identity_hash", "collection"),
        "status": "complete",
        "manifest_version": manifest.get("manifest_version", "1.0.0"),
        "run_id": manifest["collection_id"],
        "started_at_utc": manifest.get("timing", {}).get("started_at_utc", now),
        "finished_at_utc": manifest.get("timing", {}).get("updated_at_utc", now),
        "source": {
            "source_month": manifest["source_month"],
            "archive_filename": manifest["parent_archive_filename"],
            "archive_sha256": manifest["parent_archive_sha256"],
        },
        "counts": {
            "accepted_games": int(counts["accepted_games"]),
            "rejected_games": int(counts["rejected_games"]),
            "emitted_moves": int(counts["emitted_moves"]),
            "error_records": int(counts["error_records"]),
        },
        "datasets": {
            "games": {"row_count": result.accepted_games},
            "moves": {"row_count": result.emitted_moves},
            "ingestion_errors": {"row_count": result.error_records},
        },
    }
