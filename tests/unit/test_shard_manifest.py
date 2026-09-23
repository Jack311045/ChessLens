from __future__ import annotations

import json
from pathlib import Path

import pytest

from chesslens.ingestion.shard_manifest import (
    ShardEntry,
    ShardManifestError,
    atomic_write_json,
    build_shard_manifest_dict,
    load_shard_manifest,
    parse_shard_entries,
    shard_manifest_identity_hash,
    validate_shard_ranges,
)


def _entry(index: int, start: int, end: int, sha: str = "a" * 64) -> ShardEntry:
    return ShardEntry(
        shard_index=index,
        filename=f"shard-{index:05d}.pgn.zst",
        start_global_index=start,
        end_global_index=end,
        game_count=end - start,
        compressed_bytes=100,
        sha256=sha,
    )


def test_entry_round_trip() -> None:
    entry = _entry(0, 0, 7)
    assert ShardEntry.from_dict(entry.to_dict()) == entry


def test_validate_ranges_contiguous_ok() -> None:
    entries = [_entry(0, 0, 7), _entry(1, 7, 14), _entry(2, 14, 20)]
    validate_shard_ranges(entries, total_emitted_games=20)


def test_validate_ranges_rejects_gap() -> None:
    entries = [_entry(0, 0, 7), _entry(1, 8, 14)]
    with pytest.raises(ShardManifestError, match="does not equal expected"):
        validate_shard_ranges(entries)


def test_validate_ranges_rejects_overlap() -> None:
    entries = [_entry(0, 0, 7), _entry(1, 6, 14)]
    with pytest.raises(ShardManifestError, match="does not equal expected"):
        validate_shard_ranges(entries)


def test_validate_ranges_rejects_bad_index() -> None:
    entries = [_entry(0, 0, 7), _entry(2, 7, 14)]
    with pytest.raises(ShardManifestError, match="Non-contiguous shard index"):
        validate_shard_ranges(entries)


def test_validate_ranges_rejects_total_mismatch() -> None:
    entries = [_entry(0, 0, 7), _entry(1, 7, 14)]
    with pytest.raises(ShardManifestError, match="does not equal total emitted"):
        validate_shard_ranges(entries, total_emitted_games=20)


def test_atomic_write_json_replaces(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    atomic_write_json(path, {"a": 1})
    atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    assert not list(tmp_path.glob("*.tmp-*"))


def test_identity_hash_is_path_free_and_size_independent() -> None:
    entries = [_entry(0, 0, 7), _entry(1, 7, 14), _entry(2, 14, 20)]

    def _manifest(peak_rss_bytes: int, sharding_active_seconds: float) -> dict:
        return build_shard_manifest_dict(
            status="complete",
            parent_archive_filename="parent.pgn.zst",
            parent_archive_sha256="b" * 64,
            parent_archive_size_bytes=1234,
            source_month="2017-01",
            expected_total_games=20,
            games_per_shard=7,
            parquet_compression="zstd",
            compression_level=10,
            max_game_bytes=1024,
            started_at_utc="t0",
            finished_at_utc="t1",
            total_emitted_games=20,
            entries=entries,
            peak_rss_bytes=peak_rss_bytes,
            sharding_active_seconds=sharding_active_seconds,
        )

    manifest_a = _manifest(1, 1.0)
    manifest_b = _manifest(999999, 88.0)
    assert shard_manifest_identity_hash(manifest_a) == shard_manifest_identity_hash(manifest_b)


def test_load_and_parse_round_trip(tmp_path: Path) -> None:
    manifest = build_shard_manifest_dict(
        status="complete",
        parent_archive_filename="parent.pgn.zst",
        parent_archive_sha256="b" * 64,
        parent_archive_size_bytes=1234,
        source_month="2017-01",
        expected_total_games=14,
        games_per_shard=7,
        parquet_compression="zstd",
        compression_level=10,
        max_game_bytes=1024,
        started_at_utc="t0",
        finished_at_utc="t1",
        total_emitted_games=14,
        entries=[_entry(0, 0, 7), _entry(1, 7, 14)],
        peak_rss_bytes=1,
        sharding_active_seconds=1.0,
    )
    path = tmp_path / "_shard_manifest.json"
    atomic_write_json(path, manifest)
    loaded = load_shard_manifest(path)
    assert len(parse_shard_entries(loaded)) == 2
