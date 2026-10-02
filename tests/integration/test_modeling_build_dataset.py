from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import chess
import duckdb
import pytest

from chesslens.features.position_encoding import normalize_fen, position_id_from_fen
from chesslens.modeling.build_dataset import run_modeling_dataset_build
from chesslens.modeling.provenance import (
    WAREHOUSE_PROVENANCE_VERSION,
    canonical_warehouse_provenance_sha256,
)


def _parse_single_json_document(text: str) -> dict[str, object]:
    stripped = text.lstrip()
    assert stripped, "stdout is unexpectedly empty"

    decoder = json.JSONDecoder()
    payload, end_index = decoder.raw_decode(stripped)
    assert stripped[end_index:].strip() == "", "stdout contains trailing non-JSON content"
    assert isinstance(payload, dict), "CLI JSON root is not an object"
    return cast(dict[str, object], payload)


def _run_modeling_cli(
    config_path: Path,
    *,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        "-m",
        "chesslens.modeling.build_dataset",
        "--config",
        str(config_path),
    ]
    if extra_args:
        args.extend(extra_args)

    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        check=False,
    )


def _write_warehouse_provenance(path: Path, payload_without_hash: dict[str, Any]) -> None:
    payload = dict(payload_without_hash)
    payload["warehouse_provenance_sha256"] = canonical_warehouse_provenance_sha256(
        payload_without_hash
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fixture_warehouse_provenance_payload(
    *,
    collection_id: str,
    games_relation: str,
    move_context_relation: str,
    snapshot_games: int,
    snapshot_moves: int,
    warehouse_kind: str = "fixture",
    sampling: dict[str, Any] | None = None,
    transform_identity_value: str = "fixture-dbt-manifest-sha256",
) -> dict[str, Any]:
    return {
        "provenance_version": WAREHOUSE_PROVENANCE_VERSION,
        "warehouse_kind": warehouse_kind,
        "collection_id": collection_id,
        "relations": {
            "games_relation": games_relation,
            "move_context_relation": move_context_relation,
        },
        "snapshot_counts": {
            "games": snapshot_games,
            "moves": snapshot_moves,
        },
        "transformation_identity": {
            "identity_kind": "dbt_manifest_sha256",
            "identity_value": transform_identity_value,
        },
        "sampling": sampling,
    }


def _iter_manifest_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        for item in value:
            out.extend(_iter_manifest_strings(item))
        return out
    if isinstance(value, dict):
        for item in value.values():
            out.extend(_iter_manifest_strings(item))
        return out
    return out


def _is_machine_absolute_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if text.startswith("/"):
        return True
    if re.match(r"^[A-Za-z]:[/\\]", text):
        return True
    return False


def _rewrite_provenance_with_fresh_hash(path: Path, payload: dict[str, Any]) -> None:
    payload_without_hash = {
        key: value for key, value in payload.items() if key != "warehouse_provenance_sha256"
    }
    _write_warehouse_provenance(path, payload_without_hash)


def _write_collection_manifest(collection_root: Path) -> None:
    collection_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "collection_id": "collection-fixture-001",
        "status": "complete",
        "counts": {
            "accepted_games": 3,
            "emitted_moves": 5,
        },
    }
    (collection_root / "_collection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _move_rows_for_game(
    *,
    game_id: str,
    source_month: str,
    white_rating: int,
    black_rating: int,
    time_control_raw: str,
    eco: str,
    opening: str,
    result: str,
    termination: str,
    moves: list[str],
) -> list[tuple[object, ...]]:
    board = chess.Board()
    rows: list[tuple[object, ...]] = []
    for ply, move_uci in enumerate(moves):
        pre_move_fen = board.fen(en_passant="legal")
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise AssertionError(f"Illegal move in test fixture: {move_uci}")

        rows.append(
            (
                game_id,
                ply,
                source_month,
                position_id_from_fen(pre_move_fen),
                pre_move_fen,
                normalize_fen(pre_move_fen),
                "w" if board.turn == chess.WHITE else "b",
                move_uci,
                white_rating,
                black_rating,
                time_control_raw,
                eco,
                opening,
                result,
                termination,
            )
        )
        board.push(move)

    return rows


def _create_modeling_source_tables(db_path: Path, *, bad_move: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(
            """
            CREATE TABLE main.stg_games (
                game_id VARCHAR,
                source_month VARCHAR,
                played_date VARCHAR,
                result VARCHAR,
                white_player_hash VARCHAR,
                black_player_hash VARCHAR,
                white_rating INTEGER,
                black_rating INTEGER,
                time_control_raw VARCHAR,
                eco VARCHAR,
                opening VARCHAR,
                ply_count INTEGER
            )
            """
        )

        games = [
            (
                "g_train",
                "2013-01",
                "2013.01.05",
                "1-0",
                "white_a",
                "black_a",
                1500,
                1400,
                "300+0",
                "C20",
                "King Pawn Game",
                2,
            ),
            (
                "g_validation",
                "2013-01",
                "2013.01.24",
                "0-1",
                "white_b",
                "black_b",
                1700,
                1600,
                "180+2",
                "D00",
                "Queen Pawn Game",
                1,
            ),
            (
                "g_test",
                "2013-01",
                "2013.01.29",
                "1/2-1/2",
                "white_c",
                "black_c",
                1300,
                1250,
                "600+0",
                "A10",
                "English Opening",
                2,
            ),
        ]
        connection.executemany(
            """
            INSERT INTO main.stg_games (
                game_id, source_month, played_date, result,
                white_player_hash, black_player_hash,
                white_rating, black_rating,
                time_control_raw, eco, opening, ply_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            games,
        )

        move_rows = []
        move_rows.extend(
            _move_rows_for_game(
                game_id="g_train",
                source_month="2013-01",
                white_rating=1500,
                black_rating=1400,
                time_control_raw="300+0",
                eco="C20",
                opening="King Pawn Game",
                result="1-0",
                termination="Normal",
                moves=["e2e4", "e7e5"],
            )
        )
        move_rows.extend(
            _move_rows_for_game(
                game_id="g_validation",
                source_month="2013-01",
                white_rating=1700,
                black_rating=1600,
                time_control_raw="180+2",
                eco="D00",
                opening="Queen Pawn Game",
                result="0-1",
                termination="Normal",
                moves=["d2d4"],
            )
        )
        move_rows.extend(
            _move_rows_for_game(
                game_id="g_test",
                source_month="2013-01",
                white_rating=1300,
                black_rating=1250,
                time_control_raw="600+0",
                eco="A10",
                opening="English Opening",
                result="1/2-1/2",
                termination="Normal",
                moves=["c2c4", "e7e5"],
            )
        )

        if bad_move:
            first = move_rows[0]
            move_rows[0] = first[:7] + ("e7e5",) + first[8:]

        connection.execute(
            """
            CREATE TABLE main.int_move_context (
                game_id VARCHAR,
                ply INTEGER,
                source_month VARCHAR,
                position_id VARCHAR,
                pre_move_fen VARCHAR,
                normalized_pre_move_fen VARCHAR,
                side_to_move VARCHAR,
                played_move_uci VARCHAR,
                white_rating INTEGER,
                black_rating INTEGER,
                time_control_raw VARCHAR,
                eco VARCHAR,
                opening VARCHAR,
                result VARCHAR,
                termination VARCHAR
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO main.int_move_context (
                game_id, ply, source_month, position_id, pre_move_fen,
                normalized_pre_move_fen, side_to_move, played_move_uci,
                white_rating, black_rating, time_control_raw,
                eco, opening, result, termination
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            move_rows,
        )
    finally:
        connection.close()


def _execute_duckdb_sql(db_path: Path, sql: str) -> None:
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(sql)
    finally:
        connection.close()


def _write_modeling_config(
    path: Path,
    *,
    collection_root: Path,
    duckdb_path: Path,
    provenance_path: Path,
    output_root: Path,
) -> None:
    path.write_text(
        "\n".join(
            [
                "input:",
                f"  collection_root: {collection_root.as_posix()}",
                f"  duckdb_path: {duckdb_path.as_posix()}",
                f"  warehouse_provenance_path: {provenance_path.as_posix()}",
                "  expected_collection_id: collection-fixture-001",
                "  move_context_relation: main.int_move_context",
                "  games_relation: main.stg_games",
                "",
                "versions:",
                "  modeling_pipeline_version: modeling_dataset_v1",
                "  split_definition_version: temporal_game_split_v1",
                "  feature_schema_version: policy_value_features_v1",
                "  label_definition_version: policy_value_labels_v1",
                "  schema_version: 1.1.0",
                "  position_normalization_version: fen4_legal_ep_v1",
                "  board_encoding_version: board18_abs_v1",
                "  action_encoding_version: action8x8x73_v1",
                "",
                "sampling:",
                "  rule_version: sha256_mod_v1",
                "  seed: integration-seed",
                "  hash_modulus: 10000",
                "  hash_threshold: 10000",
                "  requested_rate_percent: 100.0",
                "  max_games: null",
                "",
                "splits:",
                "  train:",
                "    start_date: 2013-01-01",
                "    end_date: 2013-01-20",
                "  validation:",
                "    start_date: 2013-01-21",
                "    end_date: 2013-01-25",
                "  test:",
                "    start_date: 2013-01-26",
                "    end_date: 2013-01-31",
                "  missing_or_invalid_date_policy: assign_train",
                "",
                "player_holdout:",
                "  rule_version: sha256_mod_v1",
                "  seed: holdout-seed",
                "  hash_modulus: 10000",
                "  hash_threshold: 0",
                "  requested_rate_percent: 0.0",
                "  missing_player_hash_policy: exclude_from_player_disjoint_training",
                "",
                "output:",
                f"  output_root: {output_root.as_posix()}",
                "  batch_rows: 2",
                "  max_examples: null",
                "  parquet_compression: zstd",
                "  parquet_row_group_size: 2",
                "",
                "behavior:",
                "  strict: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _setup_fixture_environment(
    tmp_path: Path,
    *,
    bad_move: bool = False,
) -> tuple[Path, Path, Path]:
    collection_root = tmp_path / "collection"
    duckdb_path = tmp_path / "warehouse.duckdb"
    provenance_path = tmp_path / "warehouse_provenance.json"
    output_root = tmp_path / "modeling"
    config_path = tmp_path / "modeling.yaml"

    _write_collection_manifest(collection_root)
    _create_modeling_source_tables(duckdb_path, bad_move=bad_move)
    _write_warehouse_provenance(
        provenance_path,
        _fixture_warehouse_provenance_payload(
            collection_id="collection-fixture-001",
            games_relation="main.stg_games",
            move_context_relation="main.int_move_context",
            snapshot_games=3,
            snapshot_moves=5,
        ),
    )
    _write_modeling_config(
        config_path,
        collection_root=collection_root,
        duckdb_path=duckdb_path,
        provenance_path=provenance_path,
        output_root=output_root,
    )
    return config_path, output_root, provenance_path


def test_modeling_build_idempotent_reuse(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)

    first = run_modeling_dataset_build(config_path=config_path)
    second = run_modeling_dataset_build(config_path=config_path)

    assert first.reused_existing is False
    assert second.reused_existing is True
    assert first.modeling_dataset_id == second.modeling_dataset_id
    assert first.collection_id == second.collection_id == "collection-fixture-001"
    assert first.selected_games == second.selected_games == 3
    assert first.selected_examples == second.selected_examples == 5
    assert first.split_counts_games == second.split_counts_games
    assert first.split_counts_examples == second.split_counts_examples
    assert first.novel_position_test_rows == second.novel_position_test_rows == 1

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["counts"]["selected_games"] == 3
    assert manifest["counts"]["selected_policy_examples"] == 5
    assert manifest["counts"]["novel_position_test_rows"] == 1
    assert (first.dataset_path / "_SUCCESS").exists()


def test_modeling_builder_opens_duckdb_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)
    duckdb_path = (tmp_path / "warehouse.duckdb").resolve()

    original_connect = duckdb.connect
    read_only_flags: list[bool] = []

    def _recording_connect(
        database: str | Path = ":memory:",
        *,
        read_only: bool = False,
        config: dict[str, str | bool | int | float | list[str]] | None = None,
    ) -> duckdb.DuckDBPyConnection:
        db_path = Path(database).resolve()
        if db_path == duckdb_path:
            read_only_flags.append(read_only)
        if config is None:
            return original_connect(database, read_only)
        return original_connect(database, read_only, config)

    monkeypatch.setattr(duckdb, "connect", _recording_connect)

    first = run_modeling_dataset_build(config_path=config_path)
    assert first.reused_existing is False

    validate_only = run_modeling_dataset_build(config_path=config_path, validate_only=True)
    assert validate_only.validate_only is True

    reused = run_modeling_dataset_build(config_path=config_path)
    assert reused.reused_existing is True

    assert len(read_only_flags) >= 3
    assert all(read_only_flags)


def test_modeling_reuse_fails_if_duckdb_deleted(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)
    duckdb_path = tmp_path / "warehouse.duckdb"

    first = run_modeling_dataset_build(config_path=config_path)
    assert first.reused_existing is False

    duckdb_path.unlink()

    with pytest.raises(FileNotFoundError, match="duckdb_path does not exist"):
        run_modeling_dataset_build(config_path=config_path)


def test_modeling_reuse_fails_if_games_relation_changes(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)
    duckdb_path = tmp_path / "warehouse.duckdb"

    first = run_modeling_dataset_build(config_path=config_path)
    assert first.reused_existing is False

    _execute_duckdb_sql(
        duckdb_path,
        """
        INSERT INTO main.stg_games (
            game_id, source_month, played_date, result,
            white_player_hash, black_player_hash,
            white_rating, black_rating,
            time_control_raw, eco, opening, ply_count
        )
        VALUES (
            'g_extra', '2013-01', '2013.01.11', '1-0',
            'white_x', 'black_x',
            1500, 1400,
            '300+0', 'C20', 'King Pawn Game', 1
        )
        """,
    )

    with pytest.raises(RuntimeError, match="snapshot game count mismatch"):
        run_modeling_dataset_build(config_path=config_path)


def test_modeling_reuse_fails_if_move_relation_changes(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)
    duckdb_path = tmp_path / "warehouse.duckdb"

    first = run_modeling_dataset_build(config_path=config_path)
    assert first.reused_existing is False

    _execute_duckdb_sql(
        duckdb_path,
        "INSERT INTO main.int_move_context SELECT * FROM main.int_move_context LIMIT 1",
    )

    with pytest.raises(RuntimeError, match="snapshot move count mismatch"):
        run_modeling_dataset_build(config_path=config_path)


def test_validate_only_fails_if_current_snapshot_changes(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)
    duckdb_path = tmp_path / "warehouse.duckdb"

    first = run_modeling_dataset_build(config_path=config_path)
    assert first.reused_existing is False

    _execute_duckdb_sql(
        duckdb_path,
        "INSERT INTO main.int_move_context SELECT * FROM main.int_move_context LIMIT 1",
    )

    with pytest.raises(RuntimeError, match="snapshot move count mismatch"):
        run_modeling_dataset_build(config_path=config_path, validate_only=True)


def test_modeling_cli_emits_single_json_and_reuses_with_same_counts(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)

    first = _run_modeling_cli(config_path)
    assert first.returncode == 0, first.stderr
    first_payload = _parse_single_json_document(first.stdout)
    assert first_payload["reused_existing"] is False
    assert "RuntimeWarning" not in first.stderr
    assert "RuntimeWarning" not in first.stdout

    second = _run_modeling_cli(config_path)
    assert second.returncode == 0, second.stderr
    second_payload = _parse_single_json_document(second.stdout)
    assert second_payload["reused_existing"] is True
    assert "RuntimeWarning" not in second.stderr
    assert "RuntimeWarning" not in second.stdout

    assert first_payload["modeling_dataset_id"] == second_payload["modeling_dataset_id"]
    assert first_payload["selected_games"] == second_payload["selected_games"]
    assert first_payload["selected_examples"] == second_payload["selected_examples"]
    assert first_payload["split_counts_games"] == second_payload["split_counts_games"]
    assert first_payload["split_counts_examples"] == second_payload["split_counts_examples"]
    assert first_payload["novel_position_test_rows"] == second_payload["novel_position_test_rows"]
    assert first_payload["collection_id"] == second_payload["collection_id"]

    manifest_path = Path(cast(str, first_payload["manifest_path"]))
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    counts = cast(dict[str, Any], manifest["counts"])

    assert first_payload["selected_games"] == counts["selected_games"]
    assert first_payload["selected_examples"] == counts["selected_policy_examples"]
    assert first_payload["split_counts_games"] == counts["game_assignments_by_split"]
    assert first_payload["split_counts_examples"] == counts["policy_examples_by_split"]
    assert first_payload["novel_position_test_rows"] == counts["novel_position_test_rows"]


def test_modeling_cli_tampered_manifest_still_fails_reuse(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)

    first = _run_modeling_cli(config_path)
    assert first.returncode == 0, first.stderr
    first_payload = _parse_single_json_document(first.stdout)

    manifest_path = Path(cast(str, first_payload["manifest_path"]))
    manifest_payload = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest_payload["identity_payload_sha256"] = "tampered"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    second = _run_modeling_cli(config_path)
    assert second.returncode != 0
    assert second.stdout.strip() == ""
    assert "identity mismatch" in second.stderr.lower()


def test_modeling_cli_dry_run_emits_single_json(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)

    result = _run_modeling_cli(config_path, extra_args=["--dry-run"])
    assert result.returncode == 0, result.stderr

    payload = _parse_single_json_document(result.stdout)
    assert payload["dry_run"] is True
    assert payload["validate_only"] is False
    assert payload["reused_existing"] is False
    assert payload["selected_games"] == 0
    assert payload["selected_examples"] == 0
    assert payload["split_counts_games"] == {}
    assert payload["split_counts_examples"] == {}
    assert "RuntimeWarning" not in result.stderr
    assert "RuntimeWarning" not in result.stdout


def test_modeling_cli_validate_only_success_emits_single_json(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)

    build = _run_modeling_cli(config_path)
    assert build.returncode == 0, build.stderr

    result = _run_modeling_cli(config_path, extra_args=["--validate-only"])
    assert result.returncode == 0, result.stderr

    payload = _parse_single_json_document(result.stdout)
    assert payload["dry_run"] is False
    assert payload["validate_only"] is True
    assert payload["reused_existing"] is True
    assert payload["selected_games"] == 3
    assert payload["selected_examples"] == 5
    assert payload["split_counts_games"] == {"test": 1, "train": 1, "validation": 1}
    assert payload["split_counts_examples"] == {"test": 2, "train": 2, "validation": 1}
    assert payload["novel_position_test_rows"] == 1
    assert "RuntimeWarning" not in result.stderr
    assert "RuntimeWarning" not in result.stdout


def test_modeling_dataset_id_stable_with_identical_provenance(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)

    first = run_modeling_dataset_build(config_path=config_path)
    second = run_modeling_dataset_build(config_path=config_path)

    assert first.modeling_dataset_id == second.modeling_dataset_id
    assert second.reused_existing is True


def test_modeling_dataset_id_changes_when_provenance_changes(tmp_path: Path) -> None:
    config_path, _, provenance_path = _setup_fixture_environment(tmp_path)

    first = run_modeling_dataset_build(config_path=config_path)
    payload = cast(dict[str, Any], json.loads(provenance_path.read_text(encoding="utf-8")))
    transform = cast(dict[str, Any], payload["transformation_identity"])
    transform["identity_value"] = "fixture-dbt-manifest-sha256-v2"
    _rewrite_provenance_with_fresh_hash(provenance_path, payload)

    second = run_modeling_dataset_build(config_path=config_path)

    assert first.modeling_dataset_id != second.modeling_dataset_id
    assert second.reused_existing is False


def test_modeling_build_fails_when_provenance_missing(tmp_path: Path) -> None:
    config_path, _, provenance_path = _setup_fixture_environment(tmp_path)
    provenance_path.unlink()

    with pytest.raises(RuntimeError, match="Invalid warehouse provenance JSON"):
        run_modeling_dataset_build(config_path=config_path)


def test_modeling_build_fails_for_malformed_provenance(tmp_path: Path) -> None:
    config_path, _, provenance_path = _setup_fixture_environment(tmp_path)
    provenance_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid warehouse provenance JSON"):
        run_modeling_dataset_build(config_path=config_path)


def test_modeling_build_fails_for_provenance_collection_mismatch(tmp_path: Path) -> None:
    config_path, _, provenance_path = _setup_fixture_environment(tmp_path)
    payload = cast(dict[str, Any], json.loads(provenance_path.read_text(encoding="utf-8")))
    payload["collection_id"] = "other-collection-id"
    _rewrite_provenance_with_fresh_hash(provenance_path, payload)

    with pytest.raises(RuntimeError, match="collection_id does not match collection manifest"):
        run_modeling_dataset_build(config_path=config_path)


def test_modeling_build_fails_for_provenance_row_count_mismatch(tmp_path: Path) -> None:
    config_path, _, provenance_path = _setup_fixture_environment(tmp_path)
    payload = cast(dict[str, Any], json.loads(provenance_path.read_text(encoding="utf-8")))
    snapshot_counts = cast(dict[str, Any], payload["snapshot_counts"])
    snapshot_counts["games"] = 999
    _rewrite_provenance_with_fresh_hash(provenance_path, payload)

    with pytest.raises(RuntimeError, match="snapshot game count mismatch"):
        run_modeling_dataset_build(config_path=config_path)


def test_sampled_warehouse_cannot_claim_full_input(tmp_path: Path) -> None:
    config_path, _, provenance_path = _setup_fixture_environment(tmp_path)

    collection_manifest_path = tmp_path / "collection" / "_collection_manifest.json"
    collection_manifest = cast(
        dict[str, Any], json.loads(collection_manifest_path.read_text(encoding="utf-8"))
    )
    counts = cast(dict[str, Any], collection_manifest["counts"])
    counts["accepted_games"] = 30
    counts["emitted_moves"] = 50
    collection_manifest_path.write_text(
        json.dumps(collection_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    payload = cast(dict[str, Any], json.loads(provenance_path.read_text(encoding="utf-8")))
    payload["warehouse_kind"] = "full"
    payload["sampling"] = None
    _rewrite_provenance_with_fresh_hash(provenance_path, payload)

    with pytest.raises(RuntimeError, match="warehouse_kind='full'"):
        run_modeling_dataset_build(config_path=config_path)


def test_modeling_build_fails_for_tampered_provenance_hash(tmp_path: Path) -> None:
    config_path, _, provenance_path = _setup_fixture_environment(tmp_path)
    payload = cast(dict[str, Any], json.loads(provenance_path.read_text(encoding="utf-8")))
    snapshot_counts = cast(dict[str, Any], payload["snapshot_counts"])
    snapshot_counts["moves"] = 999
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="warehouse_provenance_sha256 mismatch"):
        run_modeling_dataset_build(config_path=config_path)


def test_modeling_manifest_has_no_secrets_or_machine_absolute_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)
    secret_value = "super-secret-phase2-1b"
    monkeypatch.setenv("CHESSLENS_PLAYER_HMAC_KEY", secret_value)

    result = run_modeling_dataset_build(config_path=config_path)
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    manifest = cast(dict[str, Any], json.loads(manifest_text))

    assert secret_value not in manifest_text
    assert "CHESSLENS_PLAYER_HMAC_KEY" not in manifest_text

    for text in _iter_manifest_strings(manifest):
        assert not _is_machine_absolute_path(text), text


def test_modeling_build_rejects_tampered_manifest(tmp_path: Path) -> None:
    config_path, _, _ = _setup_fixture_environment(tmp_path)

    first = run_modeling_dataset_build(config_path=config_path)

    manifest_payload = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest_payload["identity_payload_sha256"] = "tampered"
    first.manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="identity mismatch"):
        run_modeling_dataset_build(config_path=config_path)


def test_modeling_build_cleans_staging_on_failure(tmp_path: Path) -> None:
    config_path, output_root, _ = _setup_fixture_environment(tmp_path, bad_move=True)

    with pytest.raises(RuntimeError, match="Policy label validation failed"):
        run_modeling_dataset_build(config_path=config_path)

    staging_root = output_root / "staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []
