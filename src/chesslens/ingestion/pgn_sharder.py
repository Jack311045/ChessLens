"""Streaming, resumable, PGN-game-boundary-aware sharder (Phase 1.2c).

Reads a parent ``.pgn.zst`` archive one complete game at a time and writes groups
of complete games into independent ``.pgn.zst`` shards. Each shard is written to a
``.partial`` file first and only atomically renamed to its final name after it is
closed, independently decompressed, checksummed, and counted. Progress is recorded
in an atomically-written shard manifest so interrupted runs resume safely.

Resume note: zstd frames cannot be randomly seeked to a game boundary, so resuming
re-opens the parent stream and *boundary-scans* past already-emitted games. This is
byte-boundary scanning, not chess-board reconstruction.
"""

from __future__ import annotations

import itertools
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import zstandard

from chesslens.ingestion.pgn_boundaries import count_games_in_zst, iter_game_bytes_from_zst
from chesslens.ingestion.pgn_reader import compute_sha256
from chesslens.ingestion.rss import PeakRssSampler
from chesslens.ingestion.shard_manifest import (
    SHARD_MANIFEST_FILENAME,
    ShardEntry,
    ShardManifestError,
    atomic_write_json,
    build_shard_manifest_dict,
    load_shard_manifest,
    parse_shard_entries,
    validate_shard_ranges,
)
from chesslens.ingestion.sharding_config import ShardingConfig


@dataclass(frozen=True)
class ShardingResult:
    shard_dir: Path
    manifest_path: Path
    status: str
    parent_archive_sha256: str
    total_emitted_games: int
    total_shard_count: int
    new_shards_written: int
    reached_end_of_archive: bool
    peak_rss_bytes: int
    sharding_active_seconds: float


def _shard_partial_path(shard_dir: Path, index: int) -> Path:
    return shard_dir / f"shard-{index:05d}.pgn.zst.partial"


def _shard_final_path(shard_dir: Path, index: int) -> Path:
    return shard_dir / f"shard-{index:05d}.pgn.zst"


def _verify_identity(manifest: dict[str, object], config: ShardingConfig, parent_sha: str) -> None:
    compression = manifest.get("compression")
    compression = compression if isinstance(compression, dict) else {}
    mismatches: list[str] = []
    if manifest.get("parent_archive_sha256") != parent_sha:
        mismatches.append("parent_archive_sha256")
    if manifest.get("games_per_shard") != config.games_per_shard:
        mismatches.append("games_per_shard")
    if manifest.get("source_month") != config.source_month:
        mismatches.append("source_month")
    if compression.get("codec") != config.compression_codec:
        mismatches.append("compression.codec")
    if compression.get("level") != config.compression_level:
        mismatches.append("compression.level")
    if compression.get("max_game_bytes") != config.max_game_bytes:
        mismatches.append("compression.max_game_bytes")
    if mismatches:
        raise ShardManifestError(
            "Existing shard manifest identity mismatch: " + ", ".join(mismatches)
        )


def _verify_completed_shard_files(shard_dir: Path, entries: list[ShardEntry]) -> None:
    for entry in entries:
        shard_path = shard_dir / entry.filename
        if not shard_path.exists():
            raise ShardManifestError(f"Completed shard is missing: {shard_path.as_posix()}")
        actual_sha = compute_sha256(shard_path)
        if actual_sha != entry.sha256:
            raise ShardManifestError(
                f"Completed shard checksum mismatch (tampered?): {shard_path.as_posix()}"
            )


def _write_one_shard(
    *,
    game_iter: Iterator[bytes],
    shard_dir: Path,
    index: int,
    start_global: int,
    games_per_shard: int,
    compression_level: int,
    max_game_bytes: int,
) -> tuple[ShardEntry | None, bool]:
    """Write one shard. Returns (entry_or_none, reached_end_of_archive)."""

    final_path = _shard_final_path(shard_dir, index)
    if final_path.exists():
        raise ShardManifestError(
            f"Refusing to overwrite existing shard: {final_path.as_posix()}"
        )
    partial_path = _shard_partial_path(shard_dir, index)
    if partial_path.exists():
        partial_path.unlink()

    count = 0
    reached_eof = False
    compressor = zstandard.ZstdCompressor(level=compression_level)
    with partial_path.open("wb") as raw:
        with compressor.stream_writer(raw) as writer:
            while count < games_per_shard:
                try:
                    game_bytes = next(game_iter)
                except StopIteration:
                    reached_eof = True
                    break
                writer.write(game_bytes)
                count += 1

    if count == 0:
        partial_path.unlink(missing_ok=True)
        return None, True

    verified_games = count_games_in_zst(partial_path, max_game_bytes=max_game_bytes)
    if verified_games != count:
        partial_path.unlink(missing_ok=True)
        raise ShardManifestError(
            f"Independent decode of shard {index} found {verified_games} games, "
            f"expected {count}"
        )

    sha256 = compute_sha256(partial_path)
    compressed_bytes = partial_path.stat().st_size
    os.replace(partial_path, final_path)

    entry = ShardEntry(
        shard_index=index,
        filename=final_path.name,
        start_global_index=start_global,
        end_global_index=start_global + count,
        game_count=count,
        compressed_bytes=compressed_bytes,
        sha256=sha256,
        status="complete",
    )
    return entry, reached_eof


def run_sharding(
    config: ShardingConfig,
    *,
    max_new_shards: int | None = None,
    resume: bool = False,
) -> ShardingResult:
    if not config.input_path.exists():
        raise ShardManifestError(f"Parent archive not found: {config.input_path}")

    parent_sha = compute_sha256(config.input_path)
    if (
        config.expected_source_sha256 is not None
        and parent_sha != config.expected_source_sha256
    ):
        raise ShardManifestError(
            "Parent archive SHA-256 mismatch: "
            f"expected {config.expected_source_sha256}, got {parent_sha}"
        )

    shard_dir = config.shard_output_root / config.source_month
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_dir / SHARD_MANIFEST_FILENAME

    started_at = _iso_now()
    entries: list[ShardEntry] = []
    resume_from = 0
    next_index = 0

    if manifest_path.exists():
        if not resume:
            raise ShardManifestError(
                "Existing shard manifest found; pass resume=True to continue safely"
            )
        manifest = load_shard_manifest(manifest_path)
        _verify_identity(manifest, config, parent_sha)
        entries = parse_shard_entries(manifest)
        validate_shard_ranges(entries)
        _verify_completed_shard_files(shard_dir, entries)
        started_at = str(manifest.get("started_at_utc", started_at))

        if manifest.get("status") == "complete":
            return ShardingResult(
                shard_dir=shard_dir,
                manifest_path=manifest_path,
                status="complete",
                parent_archive_sha256=parent_sha,
                total_emitted_games=int(manifest.get("total_emitted_games", 0)),
                total_shard_count=len(entries),
                new_shards_written=0,
                reached_end_of_archive=True,
                peak_rss_bytes=0,
                sharding_active_seconds=0.0,
            )

        resume_from = entries[-1].end_global_index if entries else 0
        next_index = len(entries)

    stale_partial = _shard_partial_path(shard_dir, next_index)
    if stale_partial.exists():
        stale_partial.unlink()

    # An orphan final shard at next_index (published before a manifest write
    # completed) is not referenced by the manifest source of truth, so it is safe
    # to remove and rewrite. Completed shards recorded in the manifest are never
    # touched here.
    orphan_final = _shard_final_path(shard_dir, next_index)
    if orphan_final.exists():
        orphan_final.unlink()

    sampler = PeakRssSampler()
    sampler.start()
    started_timer = time.perf_counter()
    new_written = 0
    reached_eof = False
    index = next_index
    start = resume_from

    def persist(status: str, finished_at: str | None) -> None:
        payload = build_shard_manifest_dict(
            status=status,
            parent_archive_filename=config.input_path.name,
            parent_archive_sha256=parent_sha,
            parent_archive_size_bytes=config.input_path.stat().st_size,
            source_month=config.source_month,
            expected_total_games=config.expected_total_games,
            games_per_shard=config.games_per_shard,
            parquet_compression=config.compression_codec,
            compression_level=config.compression_level,
            max_game_bytes=config.max_game_bytes,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            total_emitted_games=start,
            entries=entries,
            peak_rss_bytes=peak_rss,
            sharding_active_seconds=active_seconds,
        )
        atomic_write_json(manifest_path, payload)

    peak_rss = 0
    active_seconds = 0.0
    try:
        game_iter = iter_game_bytes_from_zst(
            config.input_path,
            chunk_size=config.chunk_size,
            max_game_bytes=config.max_game_bytes,
        )
        for _ in itertools.islice(game_iter, resume_from):
            pass

        while True:
            if max_new_shards is not None and new_written >= max_new_shards:
                break
            entry, eof = _write_one_shard(
                game_iter=game_iter,
                shard_dir=shard_dir,
                index=index,
                start_global=start,
                games_per_shard=config.games_per_shard,
                compression_level=config.compression_level,
                max_game_bytes=config.max_game_bytes,
            )
            if entry is None:
                reached_eof = True
                break
            entries.append(entry)
            new_written += 1
            index += 1
            start = entry.end_global_index
            peak_rss = sampler.peak_rss_bytes
            active_seconds = max(time.perf_counter() - started_timer, 0.0)
            persist("incomplete", None)
            if eof:
                reached_eof = True
                break
    finally:
        peak_rss = sampler.stop()
        active_seconds = max(time.perf_counter() - started_timer, 0.0)

    if reached_eof:
        if (
            config.expected_total_games is not None
            and start != config.expected_total_games
        ):
            persist("incomplete", _iso_now())
            raise ShardManifestError(
                "Emitted game count "
                f"{start} does not match expected_total_games "
                f"{config.expected_total_games}; refusing to mark complete"
            )
        validate_shard_ranges(entries, total_emitted_games=start)
        persist("complete", _iso_now())
        status = "complete"
    else:
        persist("incomplete", None)
        status = "incomplete"

    return ShardingResult(
        shard_dir=shard_dir,
        manifest_path=manifest_path,
        status=status,
        parent_archive_sha256=parent_sha,
        total_emitted_games=start,
        total_shard_count=len(entries),
        new_shards_written=new_written,
        reached_end_of_archive=reached_eof,
        peak_rss_bytes=peak_rss,
        sharding_active_seconds=active_seconds,
    )


def _iso_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")
