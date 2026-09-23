"""Typed, atomically-written collection manifest for sharded ingestion.

A *collection* is the set of published per-shard datasets produced from one shard
plan. The collection manifest records deterministic collection identity, aggregate
counts and active-compute timing (never multi-day wall clock), and one entry per
processed shard so validation can reconcile the whole month without double-counting
reused shards.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from chesslens.domain.records import (
    ACTION_ENCODING_VERSION,
    BOARD_ENCODING_VERSION,
    COLLECTION_MANIFEST_VERSION,
    INGESTION_PIPELINE_VERSION,
    POSITION_NORMALIZATION_VERSION,
    SCHEMA_VERSION,
    SHARDING_PIPELINE_VERSION,
)

COLLECTION_MANIFEST_FILENAME = "_collection_manifest.json"


class CollectionManifestError(RuntimeError):
    """Raised when a collection manifest is missing, invalid, or inconsistent."""


@dataclass(frozen=True)
class CollectionShardEntry:
    """One processed shard dataset inside a collection."""

    shard_index: int
    dataset_id: str
    dataset_relpath: str
    start_global_index: int
    end_global_index: int
    scanned_games: int
    accepted_games: int
    rejected_games: int
    emitted_moves: int
    error_records: int
    output_bytes: int
    processing_active_seconds: float
    peak_rss_bytes: int
    status: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "dataset_id": self.dataset_id,
            "dataset_relpath": self.dataset_relpath,
            "start_global_index": self.start_global_index,
            "end_global_index": self.end_global_index,
            "scanned_games": self.scanned_games,
            "accepted_games": self.accepted_games,
            "rejected_games": self.rejected_games,
            "emitted_moves": self.emitted_moves,
            "error_records": self.error_records,
            "output_bytes": self.output_bytes,
            "processing_active_seconds": round(self.processing_active_seconds, 6),
            "peak_rss_bytes": self.peak_rss_bytes,
            "status": self.status,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> CollectionShardEntry:
        try:
            return CollectionShardEntry(
                shard_index=int(payload["shard_index"]),
                dataset_id=str(payload["dataset_id"]),
                dataset_relpath=str(payload["dataset_relpath"]),
                start_global_index=int(payload["start_global_index"]),
                end_global_index=int(payload["end_global_index"]),
                scanned_games=int(payload["scanned_games"]),
                accepted_games=int(payload["accepted_games"]),
                rejected_games=int(payload["rejected_games"]),
                emitted_moves=int(payload["emitted_moves"]),
                error_records=int(payload["error_records"]),
                output_bytes=int(payload["output_bytes"]),
                processing_active_seconds=float(payload["processing_active_seconds"]),
                peak_rss_bytes=int(payload["peak_rss_bytes"]),
                status=str(payload.get("status", "complete")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CollectionManifestError(f"Invalid collection shard entry: {payload!r}") from exc


def derive_collection_id(
    *,
    parent_archive_sha256: str,
    source_month: str,
    shard_manifest_identity_hash: str,
    games_per_shard: int,
    player_hmac_key_id: str | None,
) -> str:
    """Deterministic collection identity from logical inputs only.

    Excludes timestamps, secrets, absolute paths, and per-run ids.
    """

    payload = {
        "parent_archive_sha256": parent_archive_sha256,
        "source_month": source_month,
        "shard_manifest_identity_hash": shard_manifest_identity_hash,
        "games_per_shard": games_per_shard,
        "sharding_pipeline_version": SHARDING_PIPELINE_VERSION,
        "ingestion_pipeline_version": INGESTION_PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "position_normalization_version": POSITION_NORMALIZATION_VERSION,
        "board_encoding_version": BOARD_ENCODING_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "player_hmac_key_id": player_hmac_key_id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_collection_manifest_dict(
    *,
    collection_id: str,
    status: str,
    parent_archive_filename: str,
    parent_archive_sha256: str,
    source_month: str,
    shard_manifest_identity_hash: str,
    games_per_shard: int,
    player_hmac_key_id: str | None,
    expected_raw_games: int | None,
    selected_shard_count: int,
    completed_shard_count: int,
    started_at_utc: str,
    updated_at_utc: str,
    entries: list[CollectionShardEntry],
) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda entry: entry.shard_index)
    scanned = sum(entry.scanned_games for entry in ordered)
    accepted = sum(entry.accepted_games for entry in ordered)
    rejected = sum(entry.rejected_games for entry in ordered)
    emitted = sum(entry.emitted_moves for entry in ordered)
    errors = sum(entry.error_records for entry in ordered)
    output_bytes = sum(entry.output_bytes for entry in ordered)
    ingestion_active_seconds = sum(entry.processing_active_seconds for entry in ordered)
    peak_rss = max((entry.peak_rss_bytes for entry in ordered), default=0)
    return {
        "manifest_version": COLLECTION_MANIFEST_VERSION,
        "collection_id": collection_id,
        "status": status,
        "parent_archive_filename": parent_archive_filename,
        "parent_archive_sha256": parent_archive_sha256,
        "source_month": source_month,
        "shard_manifest_identity_hash": shard_manifest_identity_hash,
        "games_per_shard": games_per_shard,
        "versions": {
            "sharding_pipeline_version": SHARDING_PIPELINE_VERSION,
            "ingestion_pipeline_version": INGESTION_PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "position_normalization_version": POSITION_NORMALIZATION_VERSION,
            "board_encoding_version": BOARD_ENCODING_VERSION,
            "action_encoding_version": ACTION_ENCODING_VERSION,
            "player_hmac_key_id": player_hmac_key_id,
        },
        "expected_raw_games": expected_raw_games,
        "selected_shard_count": selected_shard_count,
        "completed_shard_count": completed_shard_count,
        "counts": {
            "scanned_games": scanned,
            "accepted_games": accepted,
            "rejected_games": rejected,
            "emitted_moves": emitted,
            "error_records": errors,
        },
        "aggregate_output_bytes": output_bytes,
        "timing": {
            "ingestion_active_seconds": round(ingestion_active_seconds, 6),
            "aggregate_games_per_second": round(scanned / ingestion_active_seconds, 4)
            if ingestion_active_seconds > 0
            else 0.0,
            "aggregate_moves_per_second": round(emitted / ingestion_active_seconds, 4)
            if ingestion_active_seconds > 0
            else 0.0,
            "peak_rss_bytes": peak_rss,
            "started_at_utc": started_at_utc,
            "updated_at_utc": updated_at_utc,
        },
        "shards": [entry.to_dict() for entry in ordered],
    }


def load_collection_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CollectionManifestError(f"Collection manifest not found: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CollectionManifestError("Collection manifest must be a JSON object")
    return cast(dict[str, Any], payload)


def parse_collection_entries(manifest: dict[str, Any]) -> list[CollectionShardEntry]:
    raw = manifest.get("shards", [])
    if not isinstance(raw, list):
        raise CollectionManifestError("Collection manifest 'shards' must be a list")
    return [CollectionShardEntry.from_dict(item) for item in raw]


def validate_collection_reconciliation(
    manifest: dict[str, Any],
    *,
    require_complete: bool = False,
) -> None:
    """Validate structural reconciliation of a collection manifest.

    Checks shard-index contiguity, half-open global range contiguity, unique
    dataset ids, and that aggregate counts equal the sum of per-shard counts.
    Deep row-level uniqueness is enforced separately by the dbt collection build.
    """

    entries = parse_collection_entries(manifest)
    ordered = sorted(entries, key=lambda entry: entry.shard_index)

    expected_index = 0
    expected_start = 0
    seen_dataset_ids: set[str] = set()
    for entry in ordered:
        if entry.shard_index != expected_index:
            raise CollectionManifestError(
                f"Non-contiguous shard index: expected {expected_index}, got {entry.shard_index}"
            )
        if entry.start_global_index != expected_start:
            raise CollectionManifestError(
                f"Shard {entry.shard_index} global range has a gap/overlap: "
                f"expected start {expected_start}, got {entry.start_global_index}"
            )
        if entry.dataset_id in seen_dataset_ids:
            raise CollectionManifestError(
                f"Duplicate dataset id registered in collection: {entry.dataset_id}"
            )
        seen_dataset_ids.add(entry.dataset_id)
        expected_index += 1
        expected_start = entry.end_global_index

    counts = manifest.get("counts", {})
    if not isinstance(counts, dict):
        raise CollectionManifestError("Collection manifest missing counts object")

    sum_accepted = sum(entry.accepted_games for entry in ordered)
    sum_emitted = sum(entry.emitted_moves for entry in ordered)
    sum_errors = sum(entry.error_records for entry in ordered)
    if int(counts.get("accepted_games", -1)) != sum_accepted:
        raise CollectionManifestError("accepted_games does not reconcile with shard entries")
    if int(counts.get("emitted_moves", -1)) != sum_emitted:
        raise CollectionManifestError("emitted_moves does not reconcile with shard entries")
    if int(counts.get("error_records", -1)) != sum_errors:
        raise CollectionManifestError("error_records does not reconcile with shard entries")

    if require_complete and manifest.get("status") != "complete":
        raise CollectionManifestError("Collection manifest status is not 'complete'")
