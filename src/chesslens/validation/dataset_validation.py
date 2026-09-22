"""DuckDB-backed validation for staged Parquet ingestion datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from chesslens.validation.schemas import (
    GAME_ARROW_SCHEMA,
    INGESTION_ERROR_ARROW_SCHEMA,
    MOVE_ARROW_SCHEMA,
)

_SOURCE_MONTH_REGEX = "source_month=([^/\\\\]+)"


class DatasetValidationError(RuntimeError):
    """Raised when staged dataset checks fail."""


@dataclass(frozen=True)
class DatasetValidationResult:
    games_rows: int
    moves_rows: int
    ingestion_error_rows: int
    checks: dict[str, int]


def _dataset_glob(dataset_root: Path, dataset_name: str) -> str:
    return (dataset_root / dataset_name / "source_month=*" / "part-*.parquet").as_posix()


def _sorted_parquet_files(dataset_root: Path, dataset_name: str) -> list[Path]:
    files = sorted((dataset_root / dataset_name).glob("source_month=*/part-*.parquet"))
    if not files:
        raise DatasetValidationError(f"No parquet files found for dataset: {dataset_name}")
    return files


def _validate_schema(dataset_root: Path, dataset_name: str) -> None:
    expected_schema_map = {
        "games": GAME_ARROW_SCHEMA,
        "moves": MOVE_ARROW_SCHEMA,
        "ingestion_errors": INGESTION_ERROR_ARROW_SCHEMA,
    }
    expected = expected_schema_map[dataset_name]
    files = _sorted_parquet_files(dataset_root, dataset_name)
    for parquet_path in files:
        actual = pq.read_schema(parquet_path)
        if not actual.equals(expected, check_metadata=False):
            raise DatasetValidationError(
                f"Schema mismatch in {parquet_path}: expected {expected}, got {actual}"
            )


def _scalar_query(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    value = connection.execute(query).fetchone()
    if value is None:
        raise DatasetValidationError(f"No result for scalar query: {query}")
    return int(value[0])


def validate_staged_dataset(
    *,
    dataset_root: Path,
    expected_source_month: str,
    expected_games: int,
    expected_moves: int,
    expected_errors: int,
) -> DatasetValidationResult:
    """Validate row counts, relational invariants, and schemas before publication."""

    for dataset_name in ("games", "moves", "ingestion_errors"):
        _validate_schema(dataset_root, dataset_name)

    games_glob = _dataset_glob(dataset_root, "games")
    moves_glob = _dataset_glob(dataset_root, "moves")
    errors_glob = _dataset_glob(dataset_root, "ingestion_errors")

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            f"CREATE VIEW games AS SELECT * FROM read_parquet('{games_glob}', filename=true)"
        )
        connection.execute(
            f"CREATE VIEW moves AS SELECT * FROM read_parquet('{moves_glob}', filename=true)"
        )
        connection.execute(
            f"CREATE VIEW ingestion_errors AS "
            f"SELECT * FROM read_parquet('{errors_glob}', filename=true)"
        )

        games_rows = _scalar_query(connection, "SELECT COUNT(*) FROM games")
        moves_rows = _scalar_query(connection, "SELECT COUNT(*) FROM moves")
        error_rows = _scalar_query(connection, "SELECT COUNT(*) FROM ingestion_errors")

        checks: dict[str, int] = {
            "games_count_mismatch": int(games_rows != expected_games),
            "moves_count_mismatch": int(moves_rows != expected_moves),
            "errors_count_mismatch": int(error_rows != expected_errors),
            "duplicate_game_id_rows": _scalar_query(
                connection,
                """
                SELECT COUNT(*)
                FROM (
                    SELECT game_id
                    FROM games
                    GROUP BY game_id
                    HAVING COUNT(*) > 1
                ) AS dup
                """,
            ),
            "duplicate_move_key_rows": _scalar_query(
                connection,
                """
                SELECT COUNT(*)
                FROM (
                    SELECT game_id, ply
                    FROM moves
                    GROUP BY game_id, ply
                    HAVING COUNT(*) > 1
                ) AS dup
                """,
            ),
            "missing_move_game_fk_rows": _scalar_query(
                connection,
                """
                SELECT COUNT(*)
                FROM moves m
                LEFT JOIN games g ON m.game_id = g.game_id
                WHERE g.game_id IS NULL
                """,
            ),
            "non_contiguous_ply_games": _scalar_query(
                connection,
                """
                SELECT COUNT(*)
                FROM (
                    SELECT game_id
                    FROM moves
                    GROUP BY game_id
                    HAVING
                        MIN(ply) <> 0
                        OR MAX(ply) + 1 <> COUNT(*)
                        OR COUNT(DISTINCT ply) <> COUNT(*)
                ) AS bad
                """,
            ),
            "ply_count_mismatch_games": _scalar_query(
                connection,
                """
                SELECT COUNT(*)
                FROM (
                    SELECT g.game_id
                    FROM games g
                    LEFT JOIN (
                        SELECT game_id, COUNT(*) AS move_rows
                        FROM moves
                        GROUP BY game_id
                    ) m ON g.game_id = m.game_id
                    WHERE g.ply_count <> COALESCE(m.move_rows, 0)
                ) AS bad
                """,
            ),
            "game_source_month_value_mismatch": _scalar_query(
                connection,
                f"""
                SELECT COUNT(*)
                FROM games
                WHERE source_month <> '{expected_source_month}'
                """,
            ),
            "game_source_month_partition_mismatch": _scalar_query(
                connection,
                f"""
                SELECT COUNT(*)
                FROM games
                WHERE regexp_extract(filename, '{_SOURCE_MONTH_REGEX}', 1) <> source_month
                """,
            ),
            "move_partition_mismatch": _scalar_query(
                connection,
                f"""
                SELECT COUNT(*)
                FROM moves
                WHERE regexp_extract(filename, '{_SOURCE_MONTH_REGEX}', 1)
                    <> '{expected_source_month}'
                """,
            ),
            "error_partition_mismatch": _scalar_query(
                connection,
                f"""
                SELECT COUNT(*)
                FROM ingestion_errors
                WHERE regexp_extract(filename, '{_SOURCE_MONTH_REGEX}', 1)
                    <> '{expected_source_month}'
                """,
            ),
        }

        failing_checks = {name: count for name, count in checks.items() if count != 0}
        if failing_checks:
            raise DatasetValidationError(f"Dataset validation failed: {failing_checks}")

        return DatasetValidationResult(
            games_rows=games_rows,
            moves_rows=moves_rows,
            ingestion_error_rows=error_rows,
            checks=checks,
        )
    finally:
        connection.close()
