from __future__ import annotations

from pathlib import Path

import duckdb

from chesslens.ingestion.collection_manifest import (
    load_collection_manifest,
    validate_collection_reconciliation,
)
from chesslens.ingestion.pgn_sharder import run_sharding
from chesslens.ingestion.run_ingestion import run_ingestion
from chesslens.ingestion.run_sharded_ingestion import run_sharded_ingestion
from chesslens.ingestion.sharding_config import load_sharding_config
from chesslens.warehouse.collection import register_collection_bronze_views

FIXTURE_SHARDING_CONFIG = Path("configs/sharding/fixture.yaml")
FIXTURE_ETL_CONFIG = Path("configs/ingestion/fixture_etl.yaml")
FIXTURE_PATH = Path("data/fixtures/lichess_2013_01_first20.pgn.zst")
FIXTURE_SHA256 = "47581f7f487a8ee84f91fc3608623865e72811dcc95511e8710482738502b3c6"


def _write_orchestrator_config(tmp_path: Path) -> Path:
    body = "\n".join(
        [
            f"input_path: {FIXTURE_PATH.resolve().as_posix()}",
            f"output_root: {(tmp_path / 'processed').as_posix()}",
            "source_month: 2013-01",
            "strict: false",
            "require_complete_games: true",
            "batch_games: 5",
            "max_buffered_records: 400",
            "parquet_compression: zstd",
            "parquet_row_group_size: 128",
            "player_hash_mode: fixture_placeholder",
            "schema_version: 1.1.0",
            "position_normalization_version: fen4_legal_ep_v1",
            "board_encoding_version: board18_abs_v1",
            "action_encoding_version: action8x8x73_v1",
            f"shard_root: {(tmp_path / 'raw_shards').as_posix()}",
            f"collection_root: {(tmp_path / 'processed' / 'collections').as_posix()}",
            "games_per_shard: 7",
            "expected_raw_games: 20",
        ]
    )
    path = tmp_path / "fixture_sharded.yaml"
    path.write_text(body + "\n", encoding="utf-8")
    return path


def _shard_fixture(tmp_path: Path) -> None:
    config = load_sharding_config(
        FIXTURE_SHARDING_CONFIG,
        overrides={"shard_output_root": str(tmp_path / "raw_shards")},
    )
    result = run_sharding(config, resume=False)
    assert result.status == "complete"
    assert result.total_shard_count == 3


def _direct_games(tmp_path: Path) -> dict[int, tuple[str, str | None, str | None]]:
    result = run_ingestion(
        config_path=FIXTURE_ETL_CONFIG,
        overrides={
            "input_path": str(FIXTURE_PATH),
            "output_root": str(tmp_path / "direct"),
            "expected_source_sha256": FIXTURE_SHA256,
            "max_games": None,
            "source_month": "2013-01",
        },
    )
    glob = (result.dataset_path / "games" / "source_month=*" / "part-*.parquet").as_posix()
    connection = duckdb.connect(":memory:")
    try:
        rows = connection.execute(
            f"SELECT source_game_index, game_id, white_player_hash, black_player_hash "
            f"FROM read_parquet('{glob}')"
        ).fetchall()
    finally:
        connection.close()
    return {int(r[0]): (r[1], r[2], r[3]) for r in rows}


def test_sharded_ingestion_matches_direct_identity_and_collection(tmp_path: Path) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)

    first = run_sharded_ingestion(config_path, max_new_shards=1, resume=True)
    assert first["new_shards_processed"] == 1
    assert first["status"] == "incomplete"
    assert first["completed_shard_count"] == 1

    final = run_sharded_ingestion(config_path, resume=True)
    assert final["status"] == "complete"
    assert final["completed_shard_count"] == 3
    assert final["accepted_games"] == 20
    assert final["portfolio_10m_satisfied"] is False

    rerun = run_sharded_ingestion(config_path, resume=True)
    assert rerun["new_shards_processed"] == 0
    assert rerun["accepted_games"] == 20

    collection_dir = Path(final["collection_dir"])
    manifest = load_collection_manifest(collection_dir / "_collection_manifest.json")
    validate_collection_reconciliation(manifest, require_complete=True)

    direct = _direct_games(tmp_path)
    assert len(direct) == 20

    register_collection_bronze_views(
        collection_root=collection_dir,
        duckdb_path=tmp_path / "wh.duckdb",
        memory_limit="512MB",
        temp_directory=tmp_path / "duckdb_tmp",
    )
    connection = duckdb.connect(str(tmp_path / "wh.duckdb"))
    try:
        collection_rows = connection.execute(
            "SELECT source_game_index, game_id, white_player_hash, black_player_hash "
            "FROM main.bronze_games"
        ).fetchall()
        dup_games = connection.execute(
            "SELECT COUNT(*) FROM (SELECT game_id FROM main.bronze_games "
            "GROUP BY game_id HAVING COUNT(*) > 1)"
        ).fetchone()
        dup_moves = connection.execute(
            "SELECT COUNT(*) FROM (SELECT game_id, ply FROM main.bronze_moves "
            "GROUP BY game_id, ply HAVING COUNT(*) > 1)"
        ).fetchone()
        distinct_months = connection.execute(
            "SELECT COUNT(DISTINCT source_month) FROM main.bronze_games"
        ).fetchone()
    finally:
        connection.close()

    assert dup_games is not None and dup_games[0] == 0
    assert dup_moves is not None and dup_moves[0] == 0
    assert distinct_months is not None and distinct_months[0] == 1

    collection = {int(r[0]): (r[1], r[2], r[3]) for r in collection_rows}
    assert collection == direct


def test_nonzero_offset_produces_global_indices(tmp_path: Path) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)
    run_sharded_ingestion(config_path, resume=True)

    collection_dir = Path(
        run_sharded_ingestion(config_path, resume=True)["collection_dir"]
    )
    manifest = load_collection_manifest(collection_dir / "_collection_manifest.json")
    second_shard = next(e for e in manifest["shards"] if e["shard_index"] == 1)
    member_root = collection_dir / second_shard["dataset_relpath"]
    glob = (member_root / "games" / "source_month=*" / "part-*.parquet").as_posix()
    connection = duckdb.connect(":memory:")
    try:
        bounds = connection.execute(
            f"SELECT MIN(source_game_index), MAX(source_game_index) "
            f"FROM read_parquet('{glob}')"
        ).fetchone()
    finally:
        connection.close()
    assert bounds is not None
    assert bounds[0] == 7
    assert bounds[1] == 13
