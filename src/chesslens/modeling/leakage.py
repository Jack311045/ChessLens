"""Position-overlap leakage measurements for temporal splits."""

from __future__ import annotations

from typing import Any

import duckdb


def _single_int(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = connection.execute(sql).fetchone()
    if row is None:
        return 0
    return int(row[0])


def compute_position_overlap_metrics(
    *,
    connection: duckdb.DuckDBPyConnection,
    relation_sql: str,
) -> dict[str, Any]:
    """Compute distinct and row-level overlap metrics across temporal splits."""

    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        split_counts[split] = {
            "rows": _single_int(
                connection,
                (f"SELECT COUNT(*) FROM ({relation_sql}) t WHERE t.temporal_split = '{split}'"),
            ),
            "distinct_positions": _single_int(
                connection,
                (
                    f"SELECT COUNT(DISTINCT position_id) FROM ({relation_sql}) t "
                    f"WHERE t.temporal_split = '{split}'"
                ),
            ),
        }

    overlaps: dict[str, dict[str, float | int | str]] = {}
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for first, second in pairs:
        key = f"{first}_{second}"
        first_positions_sql = (
            "WITH first_positions AS ("
            f"  SELECT DISTINCT position_id FROM ({relation_sql}) "
            f"  WHERE temporal_split = '{first}'"
            "), second_positions AS ("
            f"  SELECT DISTINCT position_id FROM ({relation_sql}) "
            f"  WHERE temporal_split = '{second}'"
            ") "
            "SELECT COUNT(*) FROM ("
            "  SELECT fp.position_id FROM first_positions fp "
            "  INNER JOIN second_positions sp ON fp.position_id = sp.position_id"
            ")"
        )
        distinct_overlap = _single_int(
            connection,
            first_positions_sql,
        )

        row_overlap_sql = (
            "WITH first_positions AS ("
            f"  SELECT DISTINCT position_id FROM ({relation_sql}) "
            f"  WHERE temporal_split = '{first}'"
            ") "
            f"SELECT COUNT(*) FROM ({relation_sql}) second_rows "
            f"WHERE second_rows.temporal_split = '{second}' "
            "AND second_rows.position_id IN (SELECT position_id FROM first_positions)"
        )
        row_overlap_in_second = _single_int(
            connection,
            row_overlap_sql,
        )

        second_distinct = split_counts[second]["distinct_positions"]
        second_rows = split_counts[second]["rows"]
        overlaps[key] = {
            "distinct_position_overlap": distinct_overlap,
            "row_overlap_in_second_split": row_overlap_in_second,
            "distinct_overlap_pct_of_second_distinct": (
                round((distinct_overlap / second_distinct) * 100.0, 6)
                if second_distinct > 0
                else 0.0
            ),
            "row_overlap_pct_of_second_rows": (
                round((row_overlap_in_second / second_rows) * 100.0, 6) if second_rows > 0 else 0.0
            ),
            "distinct_overlap_denominator": f"distinct_positions_{second}",
            "row_overlap_denominator": f"rows_{second}",
        }

    by_move_band_rows = connection.execute(
        "WITH base AS ("
        f"  SELECT ply, position_id, temporal_split FROM ({relation_sql})"
        "), train_positions AS ("
        "  SELECT DISTINCT position_id FROM base WHERE temporal_split = 'train'"
        "), test_rows AS ("
        "  SELECT "
        "    CASE "
        "      WHEN ply < 10 THEN 'ply_0_9' "
        "      WHEN ply < 40 THEN 'ply_10_39' "
        "      ELSE 'ply_40_plus' "
        "    END AS move_band, "
        "    position_id "
        "  FROM base "
        "  WHERE temporal_split = 'test'"
        ") "
        "SELECT "
        "  move_band, "
        "  COUNT(*) AS test_rows, "
        "  SUM(CASE WHEN position_id IN ("
        "    SELECT position_id FROM train_positions"
        "  ) THEN 1 ELSE 0 END) AS overlap_rows "
        "FROM test_rows "
        "GROUP BY move_band "
        "ORDER BY move_band"
    ).fetchall()

    by_move_band: list[dict[str, float | int | str]] = []
    for move_band, test_rows, overlap_rows in by_move_band_rows:
        test_rows_int = int(test_rows)
        overlap_rows_int = int(overlap_rows)
        by_move_band.append(
            {
                "move_band": str(move_band),
                "test_rows": test_rows_int,
                "overlap_rows_with_train": overlap_rows_int,
                "overlap_pct_of_test_rows": (
                    round((overlap_rows_int / test_rows_int) * 100.0, 6)
                    if test_rows_int > 0
                    else 0.0
                ),
            }
        )

    return {
        "split_counts": split_counts,
        "position_overlaps": overlaps,
        "test_overlap_by_move_band": by_move_band,
    }


def compute_novel_position_test_count(
    *,
    connection: duckdb.DuckDBPyConnection,
    relation_sql: str,
) -> int:
    return _single_int(
        connection,
        (
            "WITH base AS ("
            f"  SELECT position_id, temporal_split FROM ({relation_sql})"
            "), train_positions AS ("
            "  SELECT DISTINCT position_id FROM base WHERE temporal_split = 'train'"
            ") "
            "SELECT COUNT(*) FROM base "
            "WHERE temporal_split = 'test' "
            "AND position_id NOT IN (SELECT position_id FROM train_positions)"
        ),
    )
