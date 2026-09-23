from __future__ import annotations

from pathlib import Path

import pytest

from chesslens.ingestion.config import load_ingestion_config


def test_load_config_from_workspace_file() -> None:
    fixture_path = Path("data/fixtures/lichess_2013_01_first20.pgn.zst")
    config = load_ingestion_config(
        "configs/ingestion/smoke.yaml",
        overrides={"input_path": str(fixture_path)},
    )
    assert config.max_games == 100
    assert config.strict is False
    assert config.input_path == Path.cwd() / fixture_path
    assert config.input_path.exists()
    assert config.schema_version == "1.1.0"
    assert config.batch_games > 0
    assert config.max_buffered_records > 0
    assert config.player_hash_mode == "fixture_placeholder"


def test_nonzero_limit_requires_existing_input(tmp_path: Path) -> None:
    missing_input_path = (tmp_path / "missing_file.pgn.zst").as_posix()
    config_path = tmp_path / "missing.yaml"
    config_path.write_text(
        f"input_path: {missing_input_path}\nmax_games: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Input archive not found"):
        load_ingestion_config(config_path)


def test_negative_limit_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("input_path: data/raw/file.pgn.zst\nmax_games: -3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        load_ingestion_config(config_path)


def test_zero_limit_allowed_without_existing_input(tmp_path: Path) -> None:
    config_path = tmp_path / "zero.yaml"
    config_path.write_text("input_path: does/not/exist.pgn.zst\nmax_games: 0\n", encoding="utf-8")
    config = load_ingestion_config(config_path)
    assert config.max_games == 0


def test_null_max_games_means_read_to_eof(tmp_path: Path) -> None:
    archive_path = tmp_path / "fixture.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "null-max.yaml"
    config_path.write_text(
        f"input_path: {archive_path.as_posix()}\nmax_games: null\n",
        encoding="utf-8",
    )
    config = load_ingestion_config(config_path)
    assert config.max_games is None


def test_string_boolean_parsing(tmp_path: Path) -> None:
    config_path = tmp_path / "bool.yaml"
    config_path.write_text(
        "input_path: does/not/exist.pgn.zst\nmax_games: 0\nstrict: 'true'\n",
        encoding="utf-8",
    )
    config = load_ingestion_config(config_path)
    assert config.strict is True


def test_hmac_mode_requires_env_variable_names(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "hmac.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "max_games: 1",
                "player_hash_mode: hmac_sha256",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="player_hmac_key_env"):
        load_ingestion_config(config_path)


def test_expected_source_sha256_must_be_hex(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "sha.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "max_games: 1",
                "expected_source_sha256: not-a-valid-sha",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_source_sha256"):
        load_ingestion_config(config_path)


def test_batch_limits_must_be_positive(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "max_games: 1",
                "batch_games: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="batch_games"):
        load_ingestion_config(config_path)


def test_source_month_matches_inferable_filename(tmp_path: Path) -> None:
    archive_path = tmp_path / "lichess_db_standard_rated_2013-01.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "month-match.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "source_month: 2013-01",
                "max_games: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_ingestion_config(config_path)
    assert config.source_month == "2013-01"


def test_source_month_mismatch_with_filename_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "lichess_db_standard_rated_2013-01.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "month-mismatch.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "source_month: 2014-01",
                "max_games: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match archive filename month"):
        load_ingestion_config(config_path)


def test_source_month_must_be_real_calendar_month(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample_archive.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "invalid-month.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "source_month: 2013-99",
                "max_games: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="real calendar month"):
        load_ingestion_config(config_path)


def test_inferable_filename_month_must_be_real_calendar_month(tmp_path: Path) -> None:
    archive_path = tmp_path / "lichess_db_standard_rated_2013-99.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "invalid-month-name.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "max_games: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source month inferred from archive filename"):
        load_ingestion_config(config_path)


def test_non_inferable_filename_allowed_with_explicit_source_month(tmp_path: Path) -> None:
    archive_path = tmp_path / "custom_source_archive.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "non-inferable-name.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "source_month: 2013-01",
                "max_games: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_ingestion_config(config_path)
    assert config.source_month == "2013-01"


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "bad-schema-version.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "schema_version: 9.9.9",
                "max_games: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        load_ingestion_config(config_path)


def test_unsupported_position_normalization_version_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "bad-position-version.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "position_normalization_version: fen4_v999",
                "max_games: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported position_normalization_version"):
        load_ingestion_config(config_path)


def test_unsupported_board_encoding_version_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "bad-board-version.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "board_encoding_version: board_legacy",
                "max_games: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported board_encoding_version"):
        load_ingestion_config(config_path)


def test_unsupported_action_encoding_version_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.pgn.zst"
    archive_path.write_bytes(b"placeholder")
    config_path = tmp_path / "bad-action-version.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {archive_path.as_posix()}",
                "action_encoding_version: action_legacy",
                "max_games: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported action_encoding_version"):
        load_ingestion_config(config_path)