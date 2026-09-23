"""Preflight validation and DuckDB bronze view registration for Phase 1.2b."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb


class DatasetPreflightError(RuntimeError):
    """Raised when a published dataset root fails required checks."""


@dataclass(frozen=True)
class DatasetPreflightResult:
    dataset_root: Path
    manifest_path: Path
    accepted_games_manifest: int
    emitted_moves_manifest: int
    error_records_manifest: int
    accepted_games_physical: int
    emitted_moves_physical: int
    error_records_physical: int


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise DatasetPreflightError(
            f"Missing required environment variable: {name}. "
            "Set it to the published Phase 1.2a dataset directory."
        )
    return value.strip()


def _as_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetPreflightError(f"Manifest field {field_name} must be an integer") from exc


def _query_count(connection: duckdb.DuckDBPyConnection, query: str, *, field_name: str) -> int:
    row = connection.execute(query).fetchone()
    if row is None:
        raise DatasetPreflightError(f"Unable to read count for {field_name}")
    return _as_int(row[0], field_name=field_name)


def _read_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "_manifest.json"
    if not manifest_path.exists():
        raise DatasetPreflightError(
            f"Published dataset root is missing manifest: {manifest_path.as_posix()}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetPreflightError("Manifest must be a JSON object")
    return payload


def _dataset_glob(dataset_root: Path, dataset_name: str) -> str:
    return (dataset_root / dataset_name / "source_month=*" / "part-*.parquet").as_posix()


def _require_parquet_parts(dataset_root: Path, dataset_name: str) -> None:
    pattern = dataset_root / dataset_name
    files = sorted(pattern.glob("source_month=*/part-*.parquet"))
    if not files:
        raise DatasetPreflightError(
            "Required published parquet dataset is missing parts: "
            f"{(dataset_root / dataset_name).as_posix()}"
        )


def validate_published_dataset_root(dataset_root: Path) -> DatasetPreflightResult:
    if not dataset_root.exists():
        raise DatasetPreflightError(
            "CHESSLENS_DATASET_ROOT does not exist: " f"{dataset_root.as_posix()}"
        )
    if not dataset_root.is_dir():
        raise DatasetPreflightError(
            "CHESSLENS_DATASET_ROOT must be a directory: " f"{dataset_root.as_posix()}"
        )

    manifest = _read_manifest(dataset_root)
    if manifest.get("status") != "complete":
        raise DatasetPreflightError(
            "Published dataset manifest status must be 'complete' before dbt runs"
        )

    for dataset_name in ("games", "moves", "ingestion_errors"):
        _require_parquet_parts(dataset_root, dataset_name)

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise DatasetPreflightError("Manifest must include counts object")

    manifest_games = _as_int(counts.get("accepted_games"), field_name="counts.accepted_games")
    manifest_moves = _as_int(counts.get("emitted_moves"), field_name="counts.emitted_moves")
    manifest_errors = _as_int(counts.get("error_records"), field_name="counts.error_records")

    connection = duckdb.connect(database=":memory:")
    try:
        physical_games = _query_count(
            connection,
            f"SELECT COUNT(*) FROM read_parquet('{_dataset_glob(dataset_root, 'games')}')",
            field_name="physical.games",
        )
        physical_moves = _query_count(
            connection,
            f"SELECT COUNT(*) FROM read_parquet('{_dataset_glob(dataset_root, 'moves')}')",
            field_name="physical.moves",
        )
        physical_errors = _query_count(
            connection,
            (
                "SELECT COUNT(*) FROM "
                f"read_parquet('{_dataset_glob(dataset_root, 'ingestion_errors')}')"
            ),
            field_name="physical.ingestion_errors",
        )
    finally:
        connection.close()

    mismatches: list[str] = []
    if manifest_games != physical_games:
        mismatches.append(
            f"accepted_games mismatch: manifest={manifest_games}, physical={physical_games}"
        )
    if manifest_moves != physical_moves:
        mismatches.append(
            f"emitted_moves mismatch: manifest={manifest_moves}, physical={physical_moves}"
        )
    if manifest_errors != physical_errors:
        mismatches.append(
            f"error_records mismatch: manifest={manifest_errors}, physical={physical_errors}"
        )
    if mismatches:
        raise DatasetPreflightError(
            "Manifest/physical row count mismatch: " + "; ".join(mismatches)
        )

    return DatasetPreflightResult(
        dataset_root=dataset_root,
        manifest_path=dataset_root / "_manifest.json",
        accepted_games_manifest=manifest_games,
        emitted_moves_manifest=manifest_moves,
        error_records_manifest=manifest_errors,
        accepted_games_physical=physical_games,
        emitted_moves_physical=physical_moves,
        error_records_physical=physical_errors,
    )


def _sql_escape_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def register_bronze_views(
    *,
    dataset_root: Path,
    duckdb_path: Path,
    memory_limit: str,
    temp_directory: Path,
) -> None:
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(database=str(duckdb_path))
    try:
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(f"SET temp_directory = '{_sql_escape_path(temp_directory)}'")

        games_glob = _dataset_glob(dataset_root, "games").replace("'", "''")
        moves_glob = _dataset_glob(dataset_root, "moves").replace("'", "''")
        errors_glob = _dataset_glob(dataset_root, "ingestion_errors").replace("'", "''")
        manifest_file = _sql_escape_path(dataset_root / "_manifest.json")

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW main.bronze_games AS
            SELECT *
            FROM read_parquet('{games_glob}', filename = true)
            """
        )
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW main.bronze_moves AS
            SELECT *
            FROM read_parquet('{moves_glob}', filename = true)
            """
        )
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW main.bronze_ingestion_errors AS
            SELECT *
            FROM read_parquet('{errors_glob}', filename = true)
            """
        )
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW main.bronze_manifest AS
            SELECT *
            FROM read_json_auto('{manifest_file}')
            """
        )
    finally:
        connection.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate published dataset root and register DuckDB bronze views"
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Published dataset root. Defaults to CHESSLENS_DATASET_ROOT.",
    )
    parser.add_argument(
        "--duckdb-path",
        default=None,
        help=(
            "DuckDB database path for dbt. Defaults to CHESSLENS_DUCKDB_PATH or "
            "data/tmp/chesslens_warehouse.duckdb."
        ),
    )
    parser.add_argument(
        "--memory-limit",
        default=None,
        help="DuckDB memory limit. Defaults to CHESSLENS_DUCKDB_MEMORY_LIMIT or 4GB.",
    )
    parser.add_argument(
        "--temp-directory",
        default=None,
        help=(
            "DuckDB temp directory. Defaults to CHESSLENS_DUCKDB_TEMP_DIR or "
            "data/tmp/duckdb_temp."
        ),
    )
    parser.add_argument(
        "--skip-register-views",
        action="store_true",
        help="Validate only, without registering bronze views in DuckDB.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_root_raw = args.dataset_root if args.dataset_root is not None else _require_env(
        "CHESSLENS_DATASET_ROOT"
    )
    dataset_root = Path(dataset_root_raw)

    result = validate_published_dataset_root(dataset_root)

    duckdb_path = Path(
        args.duckdb_path
        or os.environ.get("CHESSLENS_DUCKDB_PATH", "data/tmp/chesslens_warehouse.duckdb")
    )
    memory_limit = args.memory_limit or os.environ.get("CHESSLENS_DUCKDB_MEMORY_LIMIT", "4GB")
    temp_directory = Path(
        args.temp_directory
        or os.environ.get("CHESSLENS_DUCKDB_TEMP_DIR", "data/tmp/duckdb_temp")
    )

    if not args.skip_register_views:
        register_bronze_views(
            dataset_root=dataset_root,
            duckdb_path=duckdb_path,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
        )

    summary = {
        "dataset_root": result.dataset_root.resolve().as_posix(),
        "manifest_path": result.manifest_path.resolve().as_posix(),
        "counts": {
            "accepted_games_manifest": result.accepted_games_manifest,
            "accepted_games_physical": result.accepted_games_physical,
            "emitted_moves_manifest": result.emitted_moves_manifest,
            "emitted_moves_physical": result.emitted_moves_physical,
            "error_records_manifest": result.error_records_manifest,
            "error_records_physical": result.error_records_physical,
        },
        "duckdb_path": duckdb_path.resolve().as_posix(),
        "duckdb_memory_limit": memory_limit,
        "duckdb_temp_directory": temp_directory.resolve().as_posix(),
        "registered_bronze_views": not args.skip_register_views,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
