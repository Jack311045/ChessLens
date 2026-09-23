"""Typed, atomically-written shard manifest for resumable PGN sharding.

The shard manifest is the source of truth for a sharding plan: it records the
parent archive identity, the sharding configuration, and one entry per completed
shard (index, filename, half-open global game range, game count, byte size, and
SHA-256). It is written atomically via a temporary file and ``os.replace`` so an
interrupted write can never corrupt a previously valid manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from chesslens.domain.records import SHARD_MANIFEST_VERSION, SHARDING_PIPELINE_VERSION

SHARD_MANIFEST_FILENAME = "_shard_manifest.json"


class ShardManifestError(RuntimeError):
    """Raised when a shard manifest is missing, invalid, or inconsistent."""


@dataclass(frozen=True)
class ShardEntry:
    """One completed, independently decompressible shard.

    The global game range is half-open: ``[start_global_index, end_global_index)``.
    """

    shard_index: int
    filename: str
    start_global_index: int
    end_global_index: int
    game_count: int
    compressed_bytes: int
    sha256: str
    status: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "filename": self.filename,
            "start_global_index": self.start_global_index,
            "end_global_index": self.end_global_index,
            "game_count": self.game_count,
            "compressed_bytes": self.compressed_bytes,
            "sha256": self.sha256,
            "status": self.status,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> ShardEntry:
        try:
            return ShardEntry(
                shard_index=int(payload["shard_index"]),
                filename=str(payload["filename"]),
                start_global_index=int(payload["start_global_index"]),
                end_global_index=int(payload["end_global_index"]),
                game_count=int(payload["game_count"]),
                compressed_bytes=int(payload["compressed_bytes"]),
                sha256=str(payload["sha256"]),
                status=str(payload.get("status", "complete")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ShardManifestError(f"Invalid shard entry: {payload!r}") from exc


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically: temp file in the same directory, then os.replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def build_shard_manifest_dict(
    *,
    status: str,
    parent_archive_filename: str,
    parent_archive_sha256: str,
    parent_archive_size_bytes: int,
    source_month: str,
    expected_total_games: int | None,
    games_per_shard: int,
    parquet_compression: str,
    compression_level: int,
    max_game_bytes: int,
    started_at_utc: str,
    finished_at_utc: str | None,
    total_emitted_games: int,
    entries: list[ShardEntry],
    peak_rss_bytes: int,
    sharding_active_seconds: float,
) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda entry: entry.shard_index)
    return {
        "manifest_version": SHARD_MANIFEST_VERSION,
        "splitter_pipeline_version": SHARDING_PIPELINE_VERSION,
        "status": status,
        "parent_archive_filename": parent_archive_filename,
        "parent_archive_sha256": parent_archive_sha256,
        "parent_archive_size_bytes": parent_archive_size_bytes,
        "source_month": source_month,
        "expected_total_games": expected_total_games,
        "games_per_shard": games_per_shard,
        "compression": {
            "codec": parquet_compression,
            "level": compression_level,
            "max_game_bytes": max_game_bytes,
        },
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "total_emitted_games": total_emitted_games,
        "total_shard_count": len(ordered),
        "performance": {
            "peak_rss_bytes": peak_rss_bytes,
            "sharding_active_seconds": round(sharding_active_seconds, 6),
            "sharding_games_per_second": round(
                total_emitted_games / sharding_active_seconds, 4
            )
            if sharding_active_seconds > 0
            else 0.0,
        },
        "shards": [entry.to_dict() for entry in ordered],
    }


def load_shard_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ShardManifestError(f"Shard manifest not found: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ShardManifestError("Shard manifest must be a JSON object")
    return cast(dict[str, Any], payload)


def parse_shard_entries(manifest: dict[str, Any]) -> list[ShardEntry]:
    raw = manifest.get("shards", [])
    if not isinstance(raw, list):
        raise ShardManifestError("Shard manifest 'shards' must be a list")
    return [ShardEntry.from_dict(item) for item in raw]


def validate_shard_ranges(
    entries: list[ShardEntry],
    *,
    total_emitted_games: int | None = None,
) -> None:
    """Validate that completed shard ranges are contiguous and half-open.

    Requires: indices 0..n-1 in order, first start == 0, each next start equals
    the previous end, no gaps or overlaps, and (when provided) the final end
    equals ``total_emitted_games``.
    """

    ordered = sorted(entries, key=lambda entry: entry.shard_index)
    expected_index = 0
    expected_start = 0
    for entry in ordered:
        if entry.shard_index != expected_index:
            raise ShardManifestError(
                f"Non-contiguous shard index: expected {expected_index}, "
                f"got {entry.shard_index}"
            )
        if entry.start_global_index != expected_start:
            raise ShardManifestError(
                f"Shard {entry.shard_index} start {entry.start_global_index} "
                f"does not equal expected {expected_start}"
            )
        if entry.end_global_index < entry.start_global_index:
            raise ShardManifestError(
                f"Shard {entry.shard_index} has end before start"
            )
        if entry.game_count != entry.end_global_index - entry.start_global_index:
            raise ShardManifestError(
                f"Shard {entry.shard_index} game_count does not match its range"
            )
        expected_index += 1
        expected_start = entry.end_global_index

    if total_emitted_games is not None and expected_start != total_emitted_games:
        raise ShardManifestError(
            f"Final shard end {expected_start} does not equal total emitted "
            f"games {total_emitted_games}"
        )


def shard_manifest_identity_hash(manifest: dict[str, Any]) -> str:
    """Path-free logical identity of a completed sharding plan.

    Excludes timestamps, byte sizes, and any machine-specific path so the value
    is stable across machines that produce the same logical shards.
    """

    entries = parse_shard_entries(manifest)
    ordered = sorted(entries, key=lambda entry: entry.shard_index)
    payload = {
        "manifest_version": manifest.get("manifest_version"),
        "splitter_pipeline_version": manifest.get("splitter_pipeline_version"),
        "parent_archive_filename": manifest.get("parent_archive_filename"),
        "parent_archive_sha256": manifest.get("parent_archive_sha256"),
        "source_month": manifest.get("source_month"),
        "games_per_shard": manifest.get("games_per_shard"),
        "total_emitted_games": manifest.get("total_emitted_games"),
        "shards": [
            {
                "shard_index": entry.shard_index,
                "start_global_index": entry.start_global_index,
                "end_global_index": entry.end_global_index,
                "game_count": entry.game_count,
                "sha256": entry.sha256,
            }
            for entry in ordered
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
