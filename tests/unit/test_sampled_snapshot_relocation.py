from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from chesslens.warehouse.sampled_snapshot import (
    inspect_sampled_snapshot,
    relocate_sampled_snapshot,
)

COLLECTION_ID = "56c008fed930b6f9883e688af135a334931b69abcb43db78a84558a3a07512eb"


def _write_sample_collection_payloads(collection_root: Path) -> dict[str, int]:
    dataset_root = (
        collection_root
        / "shards"
        / "shard-00000"
        / "datasets"
        / "dataset-sample"
    )

    games_target = dataset_root / "games" / "source_month=2017-01" / "part-000000.parquet"
    moves_target = dataset_root / "moves" / "source_month=2017-01" / "part-000000.parquet"
    games_target.parent.mkdir(parents=True, exist_ok=True)
    moves_target.parent.mkdir(parents=True, exist_ok=True)

    games_table = pa.table(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "source_month": ["2017-01", "2017-01", "2017-01", "2017-01"],
            "white_rating": [1500, 1600, 1700, 1800],
            "black_rating": [1400, 1500, 1600, 1700],
            "time_control_raw": ["300+0", "300+0", "600+0", "600+0"],
            "eco": ["C20", "C20", "D00", "A10"],
            "opening": ["KP", "KP", "QP", "English"],
            "result": ["1-0", "0-1", "1/2-1/2", "1-0"],
            "termination": ["Normal", "Normal", "Normal", "Normal"],
            "ply_count": [2, 2, 2, 2],
            "schema_version": ["1.1.0", "1.1.0", "1.1.0", "1.1.0"],
        }
    )
    pq.write_table(games_table, games_target)

    moves_table = pa.table(
        {
            "game_id": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
            "ply": [0, 1, 0, 1, 0, 1, 0, 1],
            "source_month": ["2017-01"] * 8,
            "position_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
            "pre_move_fen": ["fen"] * 8,
            "normalized_pre_move_fen": ["nfen"] * 8,
            "side_to_move": ["w", "b", "w", "b", "w", "b", "w", "b"],
            "played_move_uci": ["e2e4", "e7e5", "d2d4", "d7d5", "c2c4", "e7e6", "g1f3", "g8f6"],
            "played_move_san": ["e4", "e5", "d4", "d5", "c4", "e6", "Nf3", "Nf6"],
            "halfmove_clock": [0] * 8,
            "fullmove_number": [1, 1, 1, 1, 1, 1, 1, 1],
            "clock_annotation_raw": [None] * 8,
            "schema_version": ["1.1.0"] * 8,
        }
    )
    pq.write_table(moves_table, moves_target)

    manifest = {
        "collection_id": COLLECTION_ID,
        "status": "complete",
        "counts": {
            "accepted_games": 4,
            "emitted_moves": 8,
            "rejected_games": 0,
            "error_records": 0,
        },
    }
    (collection_root / "_collection_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (collection_root / "_collection_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    return {
        "full_games": 4,
        "full_moves": 8,
        "sampled_games": 2,
        "sampled_moves": 4,
    }


def _create_sampled_snapshot_with_stale_paths(tmp_path: Path) -> dict[str, Any]:
    source_duckdb = tmp_path / "source" / "chesslens_collection_2017_sample_10pct.duckdb"
    source_duckdb.parent.mkdir(parents=True, exist_ok=True)

    old_collection_root = (
        tmp_path
        / "old_workspace"
        / "data"
        / "processed"
        / "collections"
        / COLLECTION_ID
    )
    new_collection_root = tmp_path / "external" / "processed" / "collections" / COLLECTION_ID

    counts = _write_sample_collection_payloads(new_collection_root)
    _write_sample_collection_payloads(old_collection_root)

    old_games = (
        old_collection_root
        / "shards"
        / "shard-00000"
        / "datasets"
        / "dataset-sample"
        / "games"
        / "source_month=2017-01"
        / "part-000000.parquet"
    ).as_posix()
    old_moves = (
        old_collection_root
        / "shards"
        / "shard-00000"
        / "datasets"
        / "dataset-sample"
        / "moves"
        / "source_month=2017-01"
        / "part-000000.parquet"
    ).as_posix()
    old_manifest = (old_collection_root / "_collection_manifest.json").as_posix()

    catalog = source_duckdb.stem
    connection = duckdb.connect(str(source_duckdb))
    try:
        connection.execute("CREATE TABLE main.sampled_game_keys (game_id VARCHAR)")
        connection.execute("INSERT INTO main.sampled_game_keys VALUES ('g1'), ('g2')")
        connection.execute("CREATE TABLE main.sampled_game_indices (source_game_index BIGINT)")
        connection.execute("INSERT INTO main.sampled_game_indices VALUES (1), (2)")
        connection.execute("CREATE TABLE main.sample_counts (sampled_games BIGINT)")
        connection.execute("INSERT INTO main.sample_counts VALUES (2)")

        connection.execute(
            f"CREATE VIEW main.bronze_games_full AS SELECT * FROM read_parquet('{old_games}')"
        )
        connection.execute(
            f"CREATE VIEW main.bronze_moves_full AS SELECT * FROM read_parquet('{old_moves}')"
        )
        connection.execute(
            "CREATE VIEW main.bronze_manifest_full AS "
            f"SELECT * FROM read_json_auto('{old_manifest}')"
        )

        connection.execute(
            """
            CREATE VIEW main.bronze_games AS
            SELECT *
            FROM main.bronze_games_full
            WHERE mod(hash(game_id), 10000) < 10000
              AND game_id IN (SELECT game_id FROM main.sampled_game_keys)
            """
        )
        connection.execute(
            """
            CREATE VIEW main.bronze_moves AS
            SELECT *
            FROM main.bronze_moves_full
            WHERE mod(hash(game_id), 10000) < 10000
                            AND game_id IN (SELECT game_id FROM main.sampled_game_keys)
            """
        )
        connection.execute(
            """
            CREATE VIEW main.bronze_manifest AS
            SELECT CAST(collection_id AS VARCHAR) AS dataset_id
            FROM main.bronze_manifest_full
            """
        )

        connection.execute(
            f"""
            CREATE VIEW main.stg_games AS
            SELECT
                game_id,
                source_month,
                white_rating,
                black_rating,
                time_control_raw,
                eco,
                opening,
                result,
                termination,
                ply_count,
                schema_version
            FROM {catalog}.main.bronze_games
            """
        )
        connection.execute(
            f"""
            CREATE VIEW main.stg_moves AS
            SELECT
                game_id,
                ply,
                source_month,
                position_id,
                pre_move_fen,
                normalized_pre_move_fen,
                side_to_move,
                played_move_uci,
                played_move_san,
                halfmove_clock,
                fullmove_number,
                clock_annotation_raw,
                schema_version
            FROM {catalog}.main.bronze_moves
            """
        )
        connection.execute(
            f"""
            CREATE VIEW main.int_move_context AS
            SELECT
                m.game_id,
                m.ply,
                m.source_month,
                m.position_id,
                m.pre_move_fen,
                m.normalized_pre_move_fen,
                m.side_to_move,
                m.played_move_uci,
                m.played_move_san,
                m.halfmove_clock,
                m.fullmove_number,
                m.clock_annotation_raw,
                g.white_rating,
                g.black_rating,
                g.time_control_raw,
                g.eco,
                g.opening,
                g.result,
                g.termination,
                g.ply_count AS game_ply_count,
                g.schema_version
            FROM {catalog}.main.stg_moves AS m
            INNER JOIN {catalog}.main.stg_games AS g
                ON m.game_id = g.game_id
            """
        )
    finally:
        connection.close()

    sampling_evidence_path = tmp_path / "sampling_evidence.json"
    sampling_evidence = {
        "sampling": {
            "parent_collection_id": COLLECTION_ID,
            "parent_full_counts": {
                "games": counts["full_games"],
                "moves": counts["full_moves"],
            },
            "sampled_counts": {
                "games": counts["sampled_games"],
                "moves": counts["sampled_moves"],
            },
        }
    }
    sampling_evidence_path.write_text(
        json.dumps(sampling_evidence),
        encoding="utf-8",
    )

    return {
        "source_duckdb": source_duckdb,
        "new_collection_root": new_collection_root,
        "sampling_evidence_path": sampling_evidence_path,
        **counts,
    }


def test_inspect_sampled_snapshot_detects_stale_physical_paths(tmp_path: Path) -> None:
    payload = _create_sampled_snapshot_with_stale_paths(tmp_path)

    inspection = inspect_sampled_snapshot(
        duckdb_path=payload["source_duckdb"],
        expected_collection_root=payload["new_collection_root"],
    )

    assert inspection.deterministic_sample_detected is True
    assert COLLECTION_ID in inspection.collection_ids_from_paths
    assert "bronze_games_full" in inspection.stale_paths_by_view
    assert "bronze_moves_full" in inspection.stale_paths_by_view


def test_relocate_sampled_snapshot_preserves_sampling_semantics(tmp_path: Path) -> None:
    payload = _create_sampled_snapshot_with_stale_paths(tmp_path)
    target_directory = tmp_path / "repaired"

    result = relocate_sampled_snapshot(
        source_duckdb=payload["source_duckdb"],
        target_directory=target_directory,
        collection_root=payload["new_collection_root"],
        collection_id=COLLECTION_ID,
        sampling_evidence_path=payload["sampling_evidence_path"],
        expected_sampled_games=payload["sampled_games"],
        expected_sampled_moves=payload["sampled_moves"],
        expected_full_games=payload["full_games"],
        expected_full_moves=payload["full_moves"],
    )

    assert result.preserved_filename is True
    assert result.target_duckdb.name == payload["source_duckdb"].name
    assert "bronze_games_full" in result.replaced_views
    assert "bronze_moves_full" in result.replaced_views
    assert result.stg_games_rows == payload["sampled_games"]
    assert result.int_move_context_rows == payload["sampled_moves"]
    assert result.collection_id == COLLECTION_ID

    inspection = inspect_sampled_snapshot(
        duckdb_path=result.target_duckdb,
        expected_collection_root=payload["new_collection_root"],
    )
    assert "bronze_games_full" not in inspection.stale_paths_by_view
    assert "bronze_moves_full" not in inspection.stale_paths_by_view

    bronze_games_sql = " ".join(inspection.view_sql_by_name["bronze_games"].split())
    assert "sampled_game_keys" in bronze_games_sql
