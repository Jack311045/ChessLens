from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import pytest
import zstandard

from chesslens.domain.records import INGESTION_PIPELINE_VERSION
from chesslens.ingestion.pgn_reader import compute_sha256
from chesslens.ingestion.run_ingestion import StrictModeIngestionError, run_ingestion
from chesslens.validation.schemas import (
    GAME_ARROW_SCHEMA,
    INGESTION_ERROR_ARROW_SCHEMA,
    MOVE_ARROW_SCHEMA,
)
from tests.conftest import write_zst_text

FIXTURE_CONFIG = Path("configs/ingestion/fixture_etl.yaml")
FIXTURE_PATH = Path("data/fixtures/lichess_2013_01_first20.pgn.zst")
FIXTURE_SHA256 = "47581f7f487a8ee84f91fc3608623865e72811dcc95511e8710482738502b3c6"


def _base_overrides(tmp_path: Path) -> dict[str, Any]:
    return {
        "input_path": str(FIXTURE_PATH),
        "output_root": str(tmp_path / "processed"),
        "expected_source_sha256": FIXTURE_SHA256,
        "max_games": None,
        "strict": True,
        "require_complete_games": True,
        "batch_games": 5,
        "max_buffered_records": 400,
        "parquet_compression": "zstd",
        "parquet_row_group_size": 128,
        "player_hash_mode": "fixture_placeholder",
        "source_month": "2013-01",
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], payload)


def _set_nested(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: dict[str, Any] = mapping
    for key in path[:-1]:
        child = current[key]
        if not isinstance(child, dict):
            raise TypeError(f"Expected nested dict at {key!r}")
        current = child
    current[path[-1]] = value


def _delete_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> None:
    current: dict[str, Any] = mapping
    for key in path[:-1]:
        child = current[key]
        if not isinstance(child, dict):
            raise TypeError(f"Expected nested dict at {key!r}")
        current = child
    del current[path[-1]]


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def test_fixture_ingestion_writes_partitioned_parquet_and_manifest(tmp_path: Path) -> None:
    result = run_ingestion(config_path=FIXTURE_CONFIG, overrides=_base_overrides(tmp_path))
    manifest = _load_manifest(result.manifest_path)

    assert result.reused_existing is False
    assert manifest["status"] == "complete"
    assert manifest["dataset_id"] == result.dataset_id
    assert manifest["counts"]["accepted_games"] == 20
    assert manifest["counts"]["rejected_games"] == 0
    assert manifest["counts"]["error_records"] == 0
    assert manifest["source"]["archive_sha256"] == FIXTURE_SHA256
    assert manifest["versions"]["ingestion_pipeline_version"] == INGESTION_PIPELINE_VERSION

    games_files = list(
        (result.dataset_path / "games" / "source_month=2013-01").glob("part-*.parquet")
    )
    moves_files = list(
        (result.dataset_path / "moves" / "source_month=2013-01").glob("part-*.parquet")
    )
    errors_files = list(
        (result.dataset_path / "ingestion_errors" / "source_month=2013-01").glob("part-*.parquet")
    )

    assert games_files
    assert moves_files
    assert errors_files


def test_checksum_mismatch_fails_before_publication(tmp_path: Path) -> None:
    overrides = _base_overrides(tmp_path)
    overrides["expected_source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Source SHA-256 mismatch"):
        run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)

    assert not (tmp_path / "processed" / "datasets").exists()


def test_tiny_batch_limits_produce_multiple_parts(tmp_path: Path) -> None:
    overrides = _base_overrides(tmp_path)
    overrides["max_games"] = 5
    overrides["batch_games"] = 1

    result = run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)
    manifest = _load_manifest(result.manifest_path)

    assert manifest["counts"]["accepted_games"] == 5
    assert manifest["datasets"]["games"]["part_count"] == 5
    assert manifest["datasets"]["moves"]["part_count"] == 5
    assert manifest["datasets"]["ingestion_errors"]["part_count"] >= 1


def test_parquet_round_trip_schemas_match_contracts(tmp_path: Path) -> None:
    result = run_ingestion(config_path=FIXTURE_CONFIG, overrides=_base_overrides(tmp_path))

    games_file = sorted(
        (result.dataset_path / "games" / "source_month=2013-01").glob("part-*.parquet")
    )[0]
    moves_file = sorted(
        (result.dataset_path / "moves" / "source_month=2013-01").glob("part-*.parquet")
    )[0]
    errors_file = sorted(
        (result.dataset_path / "ingestion_errors" / "source_month=2013-01").glob("part-*.parquet")
    )[0]

    assert pq.read_schema(games_file).equals(GAME_ARROW_SCHEMA, check_metadata=False)
    assert pq.read_schema(moves_file).equals(MOVE_ARROW_SCHEMA, check_metadata=False)
    assert pq.read_schema(errors_file).equals(INGESTION_ERROR_ARROW_SCHEMA, check_metadata=False)


def test_strict_mode_rejects_incomplete_game_and_skips_publication(tmp_path: Path) -> None:
    pgn = (
        '[Event "Complete"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n'
        '[Event "Incomplete"]\n[Result "*"]\n\n1. d4 d5 *\n'
    )
    archive_path = write_zst_text(tmp_path, pgn, filename="strict-incomplete.pgn.zst")

    overrides = _base_overrides(tmp_path)
    overrides.update(
        {
            "input_path": str(archive_path),
            "expected_source_sha256": compute_sha256(archive_path),
            "source_month": "2026-01",
            "strict": True,
            "batch_games": 1,
        }
    )

    with pytest.raises(StrictModeIngestionError):
        run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)

    assert not (tmp_path / "processed" / "datasets").exists()


def test_tolerant_mode_records_incomplete_game_error(tmp_path: Path) -> None:
    pgn = (
        '[Event "Complete"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n'
        '[Event "Incomplete"]\n[Result "*"]\n\n1. d4 d5 *\n'
    )
    archive_path = write_zst_text(tmp_path, pgn, filename="tolerant-incomplete.pgn.zst")

    overrides = _base_overrides(tmp_path)
    overrides.update(
        {
            "input_path": str(archive_path),
            "expected_source_sha256": compute_sha256(archive_path),
            "source_month": "2026-01",
            "strict": False,
            "batch_games": 1,
        }
    )

    result = run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)
    manifest = _load_manifest(result.manifest_path)

    assert manifest["counts"]["accepted_games"] == 1
    assert manifest["counts"]["rejected_games"] == 1
    assert manifest["counts"]["error_records"] == 1


def test_repeated_identical_run_reuses_existing_dataset(tmp_path: Path) -> None:
    overrides = _base_overrides(tmp_path)
    first = run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)
    second = run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)

    assert first.dataset_id == second.dataset_id
    assert second.reused_existing is True
    assert first.dataset_path == second.dataset_path


@pytest.mark.parametrize(
    ("field_path", "tampered_value"),
    [
        (("dataset_id",), "tampered-dataset-id"),
        (("configuration_hash",), "tampered-configuration-hash"),
        (("source", "archive_sha256"), "f" * 64),
        (("source", "source_month"), "2099-01"),
        (("versions", "ingestion_pipeline_version"), "parquet_etl_old"),
        (("versions", "schema_version"), "9.9.9"),
        (("versions", "position_normalization_version"), "bad_position_version"),
        (("versions", "board_encoding_version"), "bad_board_version"),
        (("versions", "action_encoding_version"), "bad_action_version"),
        (("versions", "player_hmac_key_id"), "different-key-id"),
        (("status",), "failed"),
    ],
)
def test_reuse_rejected_when_manifest_identity_field_tampered(
    tmp_path: Path,
    field_path: tuple[str, ...],
    tampered_value: Any,
) -> None:
    overrides = _base_overrides(tmp_path)
    first = run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)

    manifest = _load_manifest(first.manifest_path)
    _set_nested(manifest, field_path, tampered_value)
    _save_manifest(first.manifest_path, manifest)

    with pytest.raises(RuntimeError, match="identity"):
        run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)


@pytest.mark.parametrize(
    "field_path",
    [
        ("dataset_id",),
        ("configuration_hash",),
        ("source", "archive_sha256"),
        ("source", "source_month"),
        ("versions", "ingestion_pipeline_version"),
        ("versions", "schema_version"),
        ("versions", "position_normalization_version"),
        ("versions", "board_encoding_version"),
        ("versions", "action_encoding_version"),
        ("versions", "player_hmac_key_id"),
        ("status",),
    ],
)
def test_reuse_rejected_when_manifest_identity_field_missing(
    tmp_path: Path,
    field_path: tuple[str, ...],
) -> None:
    overrides = _base_overrides(tmp_path)
    first = run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)

    manifest = _load_manifest(first.manifest_path)
    _delete_nested(manifest, field_path)
    _save_manifest(first.manifest_path, manifest)

    with pytest.raises(RuntimeError, match="missing identity field"):
        run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)


def test_output_affecting_config_change_updates_dataset_id(tmp_path: Path) -> None:
    overrides_a = _base_overrides(tmp_path)
    overrides_b = _base_overrides(tmp_path)
    overrides_a["max_games"] = 5
    overrides_b["max_games"] = 6

    run_a = run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides_a)
    run_b = run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides_b)

    assert run_a.dataset_id != run_b.dataset_id


def test_simulated_failure_after_batch_leaves_no_completed_dataset(tmp_path: Path) -> None:
    overrides = _base_overrides(tmp_path)
    overrides["max_games"] = 5
    overrides["batch_games"] = 1

    with pytest.raises(RuntimeError, match="Simulated failure"):
        run_ingestion(
            config_path=FIXTURE_CONFIG,
            overrides=overrides,
            simulate_failure_after_batches=1,
        )

    dataset_root = tmp_path / "processed" / "datasets"
    assert not dataset_root.exists() or not any(dataset_root.iterdir())


def test_failed_run_does_not_damage_existing_completed_dataset(tmp_path: Path) -> None:
    stable_overrides = _base_overrides(tmp_path)
    stable_overrides["max_games"] = 5
    stable_run = run_ingestion(config_path=FIXTURE_CONFIG, overrides=stable_overrides)

    stable_manifest_before = stable_run.manifest_path.read_text(encoding="utf-8")
    stable_files_before = sorted(
        path.as_posix() for path in stable_run.dataset_path.rglob("*.parquet")
    )

    failing_overrides = _base_overrides(tmp_path)
    failing_overrides["max_games"] = 8
    failing_overrides["batch_games"] = 1
    with pytest.raises(RuntimeError, match="Simulated failure"):
        run_ingestion(
            config_path=FIXTURE_CONFIG,
            overrides=failing_overrides,
            simulate_failure_after_batches=1,
        )

    stable_manifest_after = stable_run.manifest_path.read_text(encoding="utf-8")
    stable_files_after = sorted(
        path.as_posix() for path in stable_run.dataset_path.rglob("*.parquet")
    )

    assert stable_manifest_after == stable_manifest_before
    assert stable_files_after == stable_files_before


def test_hmac_mode_requires_environment_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overrides = _base_overrides(tmp_path)
    overrides["player_hash_mode"] = "hmac_sha256"
    overrides["player_hmac_key_env"] = "CHESSLENS_TEST_HMAC_SECRET"
    overrides["player_hmac_key_id_env"] = "CHESSLENS_TEST_HMAC_KEY_ID"

    monkeypatch.delenv("CHESSLENS_TEST_HMAC_SECRET", raising=False)
    monkeypatch.delenv("CHESSLENS_TEST_HMAC_KEY_ID", raising=False)

    with pytest.raises(ValueError, match="CHESSLENS_TEST_HMAC_SECRET"):
        run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)


def test_manifest_excludes_hmac_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret_value = "super-secret-hmac-value"
    key_id = "dev-key-1"

    monkeypatch.setenv("CHESSLENS_TEST_HMAC_SECRET", secret_value)
    monkeypatch.setenv("CHESSLENS_TEST_HMAC_KEY_ID", key_id)

    overrides = _base_overrides(tmp_path)
    overrides["player_hash_mode"] = "hmac_sha256"
    overrides["player_hmac_key_env"] = "CHESSLENS_TEST_HMAC_SECRET"
    overrides["player_hmac_key_id_env"] = "CHESSLENS_TEST_HMAC_KEY_ID"

    result = run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)
    manifest_text = result.manifest_path.read_text(encoding="utf-8")

    assert secret_value not in manifest_text
    assert secret_value not in result.dataset_path.as_posix()
    assert key_id in manifest_text


def test_empty_input_with_zero_limit(tmp_path: Path) -> None:
    archive_path = write_zst_text(tmp_path, "", filename="empty.pgn.zst")

    overrides = _base_overrides(tmp_path)
    overrides.update(
        {
            "input_path": str(archive_path),
            "expected_source_sha256": compute_sha256(archive_path),
            "source_month": "2026-02",
            "max_games": 0,
            "strict": True,
        }
    )

    result = run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)
    manifest = _load_manifest(result.manifest_path)

    assert manifest["counts"]["scanned_games"] == 0
    assert manifest["counts"]["accepted_games"] == 0
    assert manifest["counts"]["rejected_games"] == 0
    assert manifest["counts"]["emitted_moves"] == 0


def test_manifest_records_honest_duration_fields(tmp_path: Path) -> None:
    result = run_ingestion(config_path=FIXTURE_CONFIG, overrides=_base_overrides(tmp_path))
    manifest = _load_manifest(result.manifest_path)
    performance = manifest["performance"]

    checksum_duration = float(performance["checksum_duration_seconds"])
    processing_duration = float(performance["processing_and_validation_duration_seconds"])
    total_duration = float(performance["total_duration_seconds"])

    assert checksum_duration >= 0.0
    assert processing_duration >= 0.0
    assert total_duration >= 0.0
    assert total_duration >= checksum_duration
    assert total_duration >= processing_duration


def test_truncated_zstd_is_fatal_archive_error(tmp_path: Path) -> None:
    archive_path = tmp_path / "broken.pgn.zst"
    archive_path.write_bytes(b"not-zstd")

    overrides = _base_overrides(tmp_path)
    overrides.update(
        {
            "input_path": str(archive_path),
            "expected_source_sha256": compute_sha256(archive_path),
            "source_month": "2026-03",
            "strict": False,
            "max_games": None,
        }
    )

    with pytest.raises(zstandard.ZstdError):
        run_ingestion(config_path=FIXTURE_CONFIG, overrides=overrides)
