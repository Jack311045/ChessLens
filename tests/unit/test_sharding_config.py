from __future__ import annotations

from pathlib import Path

import pytest

from chesslens.ingestion.sharding_config import ShardingConfigError, load_sharding_config


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sharding.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _archive(tmp_path: Path) -> Path:
    archive = tmp_path / "lichess_db_standard_rated_2017-01.pgn.zst"
    archive.write_bytes(b"placeholder")
    return archive


def test_valid_config(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    config_path = _write(
        tmp_path,
        "\n".join(
            [
                f"input_path: {archive.as_posix()}",
                "source_month: 2017-01",
                "games_per_shard: 250000",
                "expected_total_games: 10680708",
            ]
        ),
    )
    config = load_sharding_config(config_path)
    assert config.games_per_shard == 250000
    assert config.source_month == "2017-01"
    assert config.compression_codec == "zstd"


def test_rejects_non_positive_shard_size(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    config_path = _write(
        tmp_path,
        f"input_path: {archive.as_posix()}\nsource_month: 2017-01\ngames_per_shard: 0\n",
    )
    with pytest.raises(ShardingConfigError, match="games_per_shard"):
        load_sharding_config(config_path)


def test_rejects_invalid_month(tmp_path: Path) -> None:
    archive = tmp_path / "archive.pgn.zst"
    archive.write_bytes(b"x")
    config_path = _write(
        tmp_path,
        f"input_path: {archive.as_posix()}\nsource_month: 2017-13\ngames_per_shard: 10\n",
    )
    with pytest.raises(ShardingConfigError, match="real calendar month"):
        load_sharding_config(config_path)


def test_rejects_month_filename_mismatch(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    config_path = _write(
        tmp_path,
        f"input_path: {archive.as_posix()}\nsource_month: 2018-01\ngames_per_shard: 10\n",
    )
    with pytest.raises(ShardingConfigError, match="does not match archive filename month"):
        load_sharding_config(config_path)


def test_rejects_bad_sha256(tmp_path: Path) -> None:
    archive = tmp_path / "archive.pgn.zst"
    archive.write_bytes(b"x")
    config_path = _write(
        tmp_path,
        "\n".join(
            [
                f"input_path: {archive.as_posix()}",
                "source_month: 2017-01",
                "games_per_shard: 10",
                "expected_source_sha256: not-hex",
            ]
        ),
    )
    with pytest.raises(ShardingConfigError, match="expected_source_sha256"):
        load_sharding_config(config_path)


def test_rejects_secret_like_keys(tmp_path: Path) -> None:
    archive = tmp_path / "archive.pgn.zst"
    archive.write_bytes(b"x")
    config_path = _write(
        tmp_path,
        "\n".join(
            [
                f"input_path: {archive.as_posix()}",
                "source_month: 2017-01",
                "games_per_shard: 10",
                "hmac_key: supersecret",
            ]
        ),
    )
    with pytest.raises(ShardingConfigError, match="secret-like key"):
        load_sharding_config(config_path)


def test_rejects_invalid_compression_level(tmp_path: Path) -> None:
    archive = tmp_path / "archive.pgn.zst"
    archive.write_bytes(b"x")
    config_path = _write(
        tmp_path,
        "\n".join(
            [
                f"input_path: {archive.as_posix()}",
                "source_month: 2017-01",
                "games_per_shard: 10",
                "compression_level: 99",
            ]
        ),
    )
    with pytest.raises(ShardingConfigError, match="compression_level"):
        load_sharding_config(config_path)
