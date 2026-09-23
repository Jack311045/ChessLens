from __future__ import annotations

from pathlib import Path

import pytest

import chesslens.ingestion.pgn_sharder as sharder_module
from chesslens.ingestion.pgn_boundaries import count_games_in_zst, iter_game_bytes_from_zst
from chesslens.ingestion.pgn_reader import compute_sha256
from chesslens.ingestion.pgn_sharder import run_sharding
from chesslens.ingestion.shard_manifest import (
    ShardManifestError,
    load_shard_manifest,
    parse_shard_entries,
    validate_shard_ranges,
)
from chesslens.ingestion.sharding_config import ShardingConfig, load_sharding_config

FIXTURE_SHARDING_CONFIG = Path("configs/sharding/fixture.yaml")
FIXTURE_PATH = Path("data/fixtures/lichess_2013_01_first20.pgn.zst")


def _config(tmp_path: Path, **overrides: object) -> ShardingConfig:
    merged: dict[str, object] = {"shard_output_root": str(tmp_path / "raw_shards")}
    merged.update(overrides)
    return load_sharding_config(FIXTURE_SHARDING_CONFIG, overrides=merged)


def _shard_files(shard_dir: Path) -> list[Path]:
    return sorted(shard_dir.glob("shard-*.pgn.zst"))


def test_fixture_splits_into_expected_shards(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_sharding(config, resume=False)

    assert result.status == "complete"
    assert result.total_emitted_games == 20
    assert result.total_shard_count == 3
    assert result.reached_end_of_archive is True

    entries = parse_shard_entries(load_shard_manifest(result.manifest_path))
    assert [entry.game_count for entry in entries] == [7, 7, 6]
    validate_shard_ranges(entries, total_emitted_games=20)

    for entry in entries:
        shard_path = result.shard_dir / entry.filename
        assert count_games_in_zst(shard_path) == entry.game_count
        assert compute_sha256(shard_path) == entry.sha256


def test_concatenated_shards_equal_parent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_sharding(config, resume=False)

    parent_games = list(iter_game_bytes_from_zst(FIXTURE_PATH))
    shard_games: list[bytes] = []
    for shard_path in _shard_files(result.shard_dir):
        shard_games.extend(iter_game_bytes_from_zst(shard_path))

    assert len(shard_games) == len(parent_games)
    assert shard_games == parent_games


def test_failure_mid_shard_then_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    real_count = count_games_in_zst
    calls = {"n": 0}

    def flaky_count(path: Path, **kwargs: object) -> int:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ShardManifestError("injected failure during second shard verification")
        return real_count(path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sharder_module, "count_games_in_zst", flaky_count)
    with pytest.raises(ShardManifestError, match="injected failure"):
        run_sharding(config, resume=False)
    monkeypatch.undo()

    shard_dir = config.shard_output_root / config.source_month
    assert (shard_dir / "shard-00000.pgn.zst").exists()
    assert not (shard_dir / "shard-00001.pgn.zst").exists()

    result = run_sharding(config, resume=True)
    assert result.status == "complete"
    entries = parse_shard_entries(load_shard_manifest(result.manifest_path))
    assert [entry.game_count for entry in entries] == [7, 7, 6]


def test_completed_rerun_is_idempotent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = run_sharding(config, resume=False)
    files = _shard_files(first.shard_dir)
    before = {p.name: (compute_sha256(p), p.stat().st_mtime_ns) for p in files}

    second = run_sharding(config, resume=True)
    assert second.status == "complete"
    assert second.new_shards_written == 0
    after = {
        p.name: (compute_sha256(p), p.stat().st_mtime_ns)
        for p in _shard_files(first.shard_dir)
    }
    assert before == after


def test_tampered_shard_fails_loudly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_sharding(config, resume=False)
    tampered = result.shard_dir / "shard-00001.pgn.zst"
    tampered.write_bytes(tampered.read_bytes() + b"\x00")

    with pytest.raises(ShardManifestError, match="checksum mismatch"):
        run_sharding(config, resume=True)


def test_missing_shard_fails_loudly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_sharding(config, resume=False)
    (result.shard_dir / "shard-00002.pgn.zst").unlink()

    with pytest.raises(ShardManifestError, match="missing"):
        run_sharding(config, resume=True)


def test_changed_shard_size_fails_loudly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_sharding(config, resume=False)
    changed = _config(tmp_path, games_per_shard=5)
    with pytest.raises(ShardManifestError, match="identity mismatch"):
        run_sharding(changed, resume=True)


def test_checksum_mismatch_before_publication(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_source_sha256="0" * 64)
    with pytest.raises(ShardManifestError, match="SHA-256 mismatch"):
        run_sharding(config, resume=False)
    shard_dir = config.shard_output_root / config.source_month
    if shard_dir.exists():
        assert not list(shard_dir.glob("shard-*.pgn.zst"))


def test_expected_total_mismatch_prevents_complete(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_total_games=19)
    with pytest.raises(ShardManifestError, match="does not match expected_total_games"):
        run_sharding(config, resume=False)
    manifest_path = (
        config.shard_output_root / config.source_month / "_shard_manifest.json"
    )
    manifest = load_shard_manifest(manifest_path)
    assert manifest["status"] != "complete"


def test_stale_partial_only_affects_exact_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_sharding(config, max_new_shards=1, resume=False)
    shard_dir = config.shard_output_root / config.source_month
    (shard_dir / "shard-00001.pgn.zst.partial").write_bytes(b"garbage-not-zstd")
    unrelated = shard_dir / "shard-00007.pgn.zst.partial"
    unrelated.write_bytes(b"unrelated")

    result = run_sharding(config, resume=True)
    assert result.status == "complete"
    assert not (shard_dir / "shard-00001.pgn.zst.partial").exists()
    assert unrelated.exists()


def test_manifest_write_failure_preserves_previous_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    run_sharding(config, max_new_shards=1, resume=False)
    manifest_path = config.shard_output_root / config.source_month / "_shard_manifest.json"
    saved = manifest_path.read_text(encoding="utf-8")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(sharder_module, "atomic_write_json", boom)
    with pytest.raises(OSError, match="simulated manifest write failure"):
        run_sharding(config, max_new_shards=1, resume=True)
    monkeypatch.undo()

    assert manifest_path.read_text(encoding="utf-8") == saved
    load_shard_manifest(manifest_path)  # still valid JSON

    final = run_sharding(config, resume=True)
    assert final.status == "complete"