from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

import chesslens.ingestion.run_sharded_ingestion as sharded_module
from chesslens.ingestion.collection_manifest import (
    load_collection_manifest,
    validate_collection_reconciliation,
)
from chesslens.ingestion.pgn_sharder import run_sharding
from chesslens.ingestion.run_ingestion import run_ingestion
from chesslens.ingestion.run_sharded_ingestion import (
    ShardedIngestionError,
    WorkerTestBehavior,
    run_sharded_ingestion,
)
from chesslens.ingestion.sharding_config import load_sharding_config
from chesslens.warehouse.collection import register_collection_bronze_views

FIXTURE_SHARDING_CONFIG = Path("configs/sharding/fixture.yaml")
FIXTURE_ETL_CONFIG = Path("configs/ingestion/fixture_etl.yaml")
FIXTURE_PATH = Path("data/fixtures/lichess_2013_01_first20.pgn.zst")
FIXTURE_SHA256 = "47581f7f487a8ee84f91fc3608623865e72811dcc95511e8710482738502b3c6"


def _write_orchestrator_config(
    tmp_path: Path,
    *,
    hmac_mode: bool = False,
) -> Path:
    body = [
        f"input_path: {FIXTURE_PATH.resolve().as_posix()}",
        f"output_root: {(tmp_path / 'processed').as_posix()}",
        "source_month: 2013-01",
        "strict: false",
        "require_complete_games: true",
        "batch_games: 5",
        "max_buffered_records: 400",
        "parquet_compression: zstd",
        "parquet_row_group_size: 128",
        "schema_version: 1.1.0",
        "position_normalization_version: fen4_legal_ep_v1",
        "board_encoding_version: board18_abs_v1",
        "action_encoding_version: action8x8x73_v1",
        f"shard_root: {(tmp_path / 'raw_shards').as_posix()}",
        f"collection_root: {(tmp_path / 'processed' / 'collections').as_posix()}",
        "games_per_shard: 7",
        "expected_raw_games: 20",
    ]
    if hmac_mode:
        body.extend(
            [
                "player_hash_mode: hmac_sha256",
                "player_hmac_key_env: CHESSLENS_TEST_HMAC_SECRET",
                "player_hmac_key_id_env: CHESSLENS_TEST_HMAC_KEY_ID",
            ]
        )
    else:
        body.append("player_hash_mode: fixture_placeholder")

    path = tmp_path / "fixture_sharded.yaml"
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
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


def _collection_games(
    collection_dir: Path,
    scratch_dir: Path,
) -> dict[int, tuple[str, str | None, str | None]]:
    db_path = scratch_dir / "collection.duckdb"
    register_collection_bronze_views(
        collection_root=collection_dir,
        duckdb_path=db_path,
        memory_limit="512MB",
        temp_directory=scratch_dir / "duckdb_tmp",
    )
    connection = duckdb.connect(str(db_path))
    try:
        rows = connection.execute(
            "SELECT source_game_index, game_id, white_player_hash, black_player_hash "
            "FROM main.bronze_games"
        ).fetchall()
    finally:
        connection.close()
    return {int(r[0]): (r[1], r[2], r[3]) for r in rows}


def test_sharded_ingestion_matches_direct_identity_and_collection(tmp_path: Path) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)

    first = run_sharded_ingestion(
        config_path,
        max_new_shards=1,
        resume=True,
        workers=1,
    )
    assert first["new_shards_processed"] == 1
    assert first["status"] == "incomplete"
    assert first["completed_shard_count"] == 1

    final = run_sharded_ingestion(config_path, resume=True, workers=1)
    assert final["status"] == "complete"
    assert final["completed_shard_count"] == 3
    assert final["accepted_games"] == 20
    assert final["portfolio_10m_satisfied"] is False

    rerun = run_sharded_ingestion(config_path, resume=True, workers=1)
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


def test_parallel_workers_match_sequential_identity_and_hashes(tmp_path: Path) -> None:
    seq_root = tmp_path / "seq"
    par_root = tmp_path / "par"

    _shard_fixture(seq_root)
    _shard_fixture(par_root)

    seq_config = _write_orchestrator_config(seq_root)
    par_config = _write_orchestrator_config(par_root)

    seq_result = run_sharded_ingestion(seq_config, resume=True, workers=1)
    par_result = run_sharded_ingestion(par_config, resume=True, workers=2)

    seq_collection = _collection_games(Path(seq_result["collection_dir"]), seq_root / "warehouse")
    par_collection = _collection_games(Path(par_result["collection_dir"]), par_root / "warehouse")

    assert seq_result["accepted_games"] == par_result["accepted_games"] == 20
    assert seq_result["emitted_moves"] == par_result["emitted_moves"]
    assert seq_collection == par_collection


def test_parallel_out_of_order_completion_keeps_manifest_contiguous(tmp_path: Path) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)

    result = run_sharded_ingestion(
        config_path,
        resume=True,
        workers=2,
        max_new_shards=3,
        test_worker_behavior=WorkerTestBehavior(
            sleep_seconds_by_shard_index={0: 0.25, 1: 0.01},
        ),
    )

    manifest = load_collection_manifest(Path(result["collection_manifest"]))
    assert [entry["shard_index"] for entry in manifest["shards"]] == [0, 1, 2]
    assert result["failed_shard_indices"] == []
    assert result["new_shards_processed"] == 3


def test_worker_failure_then_resume_completes(tmp_path: Path) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)

    with pytest.raises(ShardedIngestionError, match="selected shards failed"):
        run_sharded_ingestion(
            config_path,
            resume=True,
            workers=2,
            max_new_shards=3,
            test_worker_behavior=WorkerTestBehavior(
                sleep_seconds_by_shard_index={1: 0.2},
                fail_shard_indices={1},
            ),
        )

    collection_id = run_sharded_ingestion(
        config_path,
        resume=True,
        workers=2,
        max_new_shards=3,
    )["collection_id"]
    manifest_path = (
        tmp_path
        / "processed"
        / "collections"
        / collection_id
        / "_collection_manifest.json"
    )
    manifest = load_collection_manifest(manifest_path)
    assert manifest["status"] == "complete"
    assert manifest["completed_shard_count"] == 3


def test_max_new_shards_is_total_not_per_worker(tmp_path: Path) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)

    first = run_sharded_ingestion(
        config_path,
        resume=True,
        workers=1,
        max_new_shards=1,
    )
    assert first["selected_shard_indices"] == [0]
    assert first["new_shards_processed"] == 1

    second = run_sharded_ingestion(
        config_path,
        resume=True,
        workers=3,
        max_new_shards=2,
    )
    assert second["selected_shard_indices"] == [1, 2]
    assert second["new_shards_processed"] == 2


def test_workers_greater_than_selected_shards_is_safe(tmp_path: Path) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)

    result = run_sharded_ingestion(
        config_path,
        resume=True,
        workers=8,
        max_new_shards=1,
    )
    assert result["new_shards_processed"] == 1
    assert result["selected_shard_indices"] == [0]


def test_invalid_worker_count_rejected(tmp_path: Path) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)

    with pytest.raises(ShardedIngestionError, match="workers"):
        run_sharded_ingestion(
            config_path,
            resume=True,
            workers=0,
            max_new_shards=1,
        )


def test_run_metrics_include_process_tree_peak_rss(tmp_path: Path) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)

    result = run_sharded_ingestion(
        config_path,
        resume=True,
        workers=2,
    )

    assert result["worker_count"] == 2
    assert result["peak_process_tree_rss_bytes"] > 0
    assert result["max_worker_peak_rss_bytes"] > 0
    assert result["per_worker_peak_rss_bytes"]
    assert result["wall_clock_duration_seconds"] > 0


def test_manifest_write_failure_preserves_previous_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)

    first = run_sharded_ingestion(
        config_path,
        resume=True,
        workers=1,
        max_new_shards=1,
    )
    manifest_path = Path(first["collection_manifest"])
    saved = manifest_path.read_text(encoding="utf-8")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(sharded_module, "atomic_write_json", boom)
    with pytest.raises(OSError, match="simulated manifest write failure"):
        run_sharded_ingestion(
            config_path,
            resume=True,
            workers=1,
            max_new_shards=1,
        )

    assert manifest_path.read_text(encoding="utf-8") == saved
    load_collection_manifest(manifest_path)


def test_collection_lock_blocks_second_mutator(tmp_path: Path) -> None:
    lock_path = tmp_path / "lockfile"

    first = sharded_module._CollectionMutationLock(lock_path)
    first.acquire()
    second = sharded_module._CollectionMutationLock(lock_path)
    try:
        with pytest.raises(ShardedIngestionError, match="active"):
            second.acquire()
    finally:
        first.release()
        lock_path.unlink(missing_ok=True)


def test_stale_lock_is_recovered(tmp_path: Path) -> None:
    lock_path = tmp_path / "lockfile"

    lock_path.write_text(
        json.dumps(
            {
                "pid": 999999,
                "process_create_time": 0.0,
                "hostname": "test-host",
                "acquired_at_utc": "2020-01-01T00:00:00+00:00",
                "token": "stale",
            }
        ),
        encoding="utf-8",
    )

    lock = sharded_module._CollectionMutationLock(lock_path)
    lock.acquire()
    assert lock_path.exists()
    lock.release()
    assert not lock_path.exists()


def test_hmac_secret_never_appears_in_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path, hmac_mode=True)
    secret = "phase12d-super-secret"
    monkeypatch.setenv("CHESSLENS_TEST_HMAC_SECRET", secret)
    monkeypatch.setenv("CHESSLENS_TEST_HMAC_KEY_ID", "phase12d-test-key")

    result = run_sharded_ingestion(config_path, resume=True, workers=2)

    collection_dir = Path(result["collection_dir"])
    collection_manifest_path = collection_dir / "_collection_manifest.json"
    collection_manifest_text = collection_manifest_path.read_text(encoding="utf-8")
    assert secret not in collection_manifest_text
    assert secret not in json.dumps(result, sort_keys=True)

    manifest = load_collection_manifest(collection_manifest_path)
    for shard in manifest["shards"]:
        dataset_manifest = collection_dir / shard["dataset_relpath"] / "_manifest.json"
        assert secret not in dataset_manifest.read_text(encoding="utf-8")


def test_nonzero_offset_produces_global_indices(tmp_path: Path) -> None:
    _shard_fixture(tmp_path)
    config_path = _write_orchestrator_config(tmp_path)
    run_sharded_ingestion(config_path, resume=True, workers=2)

    collection_dir = Path(
        run_sharded_ingestion(config_path, resume=True, workers=2)["collection_dir"]
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
