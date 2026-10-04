"""Resumable header-only UTCDate enrichment sidecar for shard plans.

This pipeline reads existing shard files and writes a per-game sidecar keyed by
stable game identity so downstream temporal splitting can consume canonical dates
without rewriting existing bronze datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from chesslens.ingestion.date_resolution import (
    CANONICAL_PLAYED_DATE_RESOLVER_VERSION,
    resolve_canonical_played_date,
)
from chesslens.ingestion.pgn_reader import compute_sha256, iter_raw_games, stable_game_id
from chesslens.ingestion.rss import PeakRssSampler
from chesslens.ingestion.shard_manifest import (
    SHARD_MANIFEST_FILENAME,
    ShardEntry,
    atomic_write_json,
    load_shard_manifest,
    parse_shard_entries,
    shard_manifest_identity_hash,
    validate_shard_ranges,
)

DATE_ENRICHMENT_MANIFEST_FILENAME = "_date_enrichment_manifest.json"
DATE_ENRICHMENT_MANIFEST_VERSION = "1.0.0"
DATE_ENRICHMENT_PIPELINE_VERSION = "played_date_sidecar_v1"

DATE_ENRICHMENT_ARROW_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("source_month", pa.string(), nullable=True),
        pa.field("source_game_index", pa.int64(), nullable=False),
        pa.field("canonical_played_date_iso", pa.string(), nullable=True),
        pa.field("canonical_date_source", pa.string(), nullable=False),
        pa.field("date_header_raw", pa.string(), nullable=True),
        pa.field("utc_date_header_raw", pa.string(), nullable=True),
        pa.field("date_header_valid", pa.bool_(), nullable=False),
        pa.field("utc_date_header_valid", pa.bool_(), nullable=False),
        pa.field("resolver_version", pa.string(), nullable=False),
    ]
)


class DateEnrichmentError(RuntimeError):
    """Raised when date enrichment cannot proceed safely."""


@dataclass(frozen=True)
class DateEnrichmentShardEntry:
    shard_index: int
    shard_filename: str
    output_relpath: str
    start_global_index: int
    end_global_index: int
    game_count: int
    output_bytes: int
    output_sha256: str
    canonical_from_utc_date: int
    canonical_from_date: int
    canonical_missing_or_invalid: int
    invalid_date_but_valid_utc_date: int
    status: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "shard_filename": self.shard_filename,
            "output_relpath": self.output_relpath,
            "start_global_index": self.start_global_index,
            "end_global_index": self.end_global_index,
            "game_count": self.game_count,
            "output_bytes": self.output_bytes,
            "output_sha256": self.output_sha256,
            "canonical_from_utc_date": self.canonical_from_utc_date,
            "canonical_from_date": self.canonical_from_date,
            "canonical_missing_or_invalid": self.canonical_missing_or_invalid,
            "invalid_date_but_valid_utc_date": self.invalid_date_but_valid_utc_date,
            "status": self.status,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> DateEnrichmentShardEntry:
        try:
            return DateEnrichmentShardEntry(
                shard_index=int(payload["shard_index"]),
                shard_filename=str(payload["shard_filename"]),
                output_relpath=str(payload["output_relpath"]),
                start_global_index=int(payload["start_global_index"]),
                end_global_index=int(payload["end_global_index"]),
                game_count=int(payload["game_count"]),
                output_bytes=int(payload["output_bytes"]),
                output_sha256=str(payload["output_sha256"]),
                canonical_from_utc_date=int(payload["canonical_from_utc_date"]),
                canonical_from_date=int(payload["canonical_from_date"]),
                canonical_missing_or_invalid=int(payload["canonical_missing_or_invalid"]),
                invalid_date_but_valid_utc_date=int(
                    payload["invalid_date_but_valid_utc_date"]
                ),
                status=str(payload.get("status", "complete")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DateEnrichmentError(f"Invalid date enrichment shard entry: {payload!r}") from exc


@dataclass(frozen=True)
class DateEnrichmentResult:
    date_enrichment_id: str
    dataset_root: Path
    manifest_path: Path
    status: str
    reused_existing: bool
    selected_shard_count: int
    completed_shard_count: int
    enriched_games: int
    new_shards_written: int
    peak_rss_bytes: int
    active_seconds: float


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_date_enrichment_id(
    *,
    parent_archive_sha256: str,
    source_month: str,
    shard_manifest_identity: str,
    resolver_version: str,
) -> str:
    payload = {
        "parent_archive_sha256": parent_archive_sha256,
        "source_month": source_month,
        "shard_manifest_identity_hash": shard_manifest_identity,
        "pipeline_version": DATE_ENRICHMENT_PIPELINE_VERSION,
        "resolver_version": resolver_version,
    }
    return _sha256_text(_canonical_json(payload))


def _manifest_path(dataset_root: Path) -> Path:
    return dataset_root / DATE_ENRICHMENT_MANIFEST_FILENAME


def _parse_manifest_entries(manifest: dict[str, Any]) -> list[DateEnrichmentShardEntry]:
    raw = manifest.get("shards", [])
    if not isinstance(raw, list):
        raise DateEnrichmentError("Date enrichment manifest 'shards' must be a list")
    return [DateEnrichmentShardEntry.from_dict(item) for item in raw]


def _validate_entry_ranges(entries: list[DateEnrichmentShardEntry]) -> None:
    ordered = sorted(entries, key=lambda item: item.shard_index)
    expected_index = 0
    expected_start = 0
    for entry in ordered:
        if entry.shard_index != expected_index:
            raise DateEnrichmentError(
                f"Non-contiguous enrichment shard index: expected {expected_index}, "
                f"got {entry.shard_index}"
            )
        if entry.start_global_index != expected_start:
            raise DateEnrichmentError(
                "Date enrichment shard range has gap/overlap: "
                f"expected {expected_start}, got {entry.start_global_index}"
            )
        if entry.end_global_index < entry.start_global_index:
            raise DateEnrichmentError(
                f"Date enrichment shard {entry.shard_index} end is before start"
            )
        if entry.game_count != entry.end_global_index - entry.start_global_index:
            raise DateEnrichmentError(
                f"Date enrichment shard {entry.shard_index} game_count mismatch"
            )
        expected_index += 1
        expected_start = entry.end_global_index


def _verify_entry_artifacts(dataset_root: Path, entries: list[DateEnrichmentShardEntry]) -> None:
    for entry in entries:
        rel = Path(entry.output_relpath)
        if rel.is_absolute() or ".." in rel.parts:
            raise DateEnrichmentError(
                f"Invalid output_relpath in manifest: {entry.output_relpath!r}"
            )
        artifact_path = dataset_root / rel
        if not artifact_path.exists():
            raise DateEnrichmentError(
                f"Date enrichment artifact missing: {artifact_path.as_posix()}"
            )
        actual_sha = compute_sha256(artifact_path)
        if actual_sha != entry.output_sha256:
            raise DateEnrichmentError(
                "Date enrichment artifact checksum mismatch (tampered?): "
                f"{artifact_path.as_posix()}"
            )


def _sum_counts(entries: list[DateEnrichmentShardEntry]) -> dict[str, int]:
    return {
        "enriched_games": sum(item.game_count for item in entries),
        "canonical_from_utc_date": sum(item.canonical_from_utc_date for item in entries),
        "canonical_from_date": sum(item.canonical_from_date for item in entries),
        "canonical_missing_or_invalid": sum(
            item.canonical_missing_or_invalid for item in entries
        ),
        "invalid_date_but_valid_utc_date": sum(
            item.invalid_date_but_valid_utc_date for item in entries
        ),
    }


def _build_manifest(
    *,
    date_enrichment_id: str,
    status: str,
    shard_manifest: dict[str, Any],
    shard_manifest_identity: str,
    selected_shard_count: int,
    completed_shard_count: int,
    started_at_utc: str,
    updated_at_utc: str,
    entries: list[DateEnrichmentShardEntry],
    peak_rss_bytes: int,
    active_seconds: float,
) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda item: item.shard_index)
    counts = _sum_counts(ordered)
    return {
        "manifest_version": DATE_ENRICHMENT_MANIFEST_VERSION,
        "pipeline_version": DATE_ENRICHMENT_PIPELINE_VERSION,
        "resolver_version": CANONICAL_PLAYED_DATE_RESOLVER_VERSION,
        "status": status,
        "date_enrichment_id": date_enrichment_id,
        "parent_archive_filename": shard_manifest.get("parent_archive_filename"),
        "parent_archive_sha256": shard_manifest.get("parent_archive_sha256"),
        "source_month": shard_manifest.get("source_month"),
        "shard_manifest_identity_hash": shard_manifest_identity,
        "games_per_shard": shard_manifest.get("games_per_shard"),
        "selected_shard_count": selected_shard_count,
        "completed_shard_count": completed_shard_count,
        "counts": counts,
        "timing": {
            "started_at_utc": started_at_utc,
            "updated_at_utc": updated_at_utc,
            "active_seconds": round(active_seconds, 6),
            "games_per_second": round(
                counts["enriched_games"] / active_seconds, 4
            )
            if active_seconds > 0
            else 0.0,
            "peak_rss_bytes": int(peak_rss_bytes),
        },
        "shards": [entry.to_dict() for entry in ordered],
    }


def _load_and_validate_existing_manifest(
    *,
    manifest_path: Path,
    expected_date_enrichment_id: str,
    expected_parent_archive_sha256: str,
    expected_source_month: str,
    expected_shard_manifest_identity: str,
) -> tuple[dict[str, Any], list[DateEnrichmentShardEntry]]:
    if not manifest_path.exists():
        raise DateEnrichmentError(f"Date enrichment manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DateEnrichmentError(
            f"Date enrichment manifest root must be object: {manifest_path.as_posix()}"
        )

    checks = {
        "date_enrichment_id": expected_date_enrichment_id,
        "parent_archive_sha256": expected_parent_archive_sha256,
        "source_month": expected_source_month,
        "resolver_version": CANONICAL_PLAYED_DATE_RESOLVER_VERSION,
        "shard_manifest_identity_hash": expected_shard_manifest_identity,
    }
    mismatches: list[str] = []
    for key, expected in checks.items():
        if payload.get(key) != expected:
            mismatches.append(
                f"{key}: expected {expected!r}, got {payload.get(key)!r}"
            )
    if mismatches:
        raise DateEnrichmentError(
            "Date enrichment manifest identity mismatch: " + "; ".join(mismatches)
        )

    entries = _parse_manifest_entries(payload)
    _validate_entry_ranges(entries)
    return payload, entries


def _write_sidecar_for_shard(
    *,
    shard_path: Path,
    shard_entry: ShardEntry,
    parent_archive_sha256: str,
    source_month: str,
    output_path: Path,
    row_group_size: int,
) -> DateEnrichmentShardEntry:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp-{os.getpid()}")
    if temp_path.exists():
        temp_path.unlink()

    columns = [field.name for field in DATE_ENRICHMENT_ARROW_SCHEMA]
    buffered: dict[str, list[Any]] = {name: [] for name in columns}
    writer: pq.ParquetWriter | None = None
    row_count = 0

    canonical_from_utc_date = 0
    canonical_from_date = 0
    canonical_missing_or_invalid = 0
    invalid_date_but_valid_utc_date = 0

    def flush_buffer() -> None:
        nonlocal writer
        if not buffered["game_id"]:
            return
        table = pa.table(buffered, schema=DATE_ENRICHMENT_ARROW_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(
                str(temp_path),
                DATE_ENRICHMENT_ARROW_SCHEMA,
                compression="zstd",
            )
        writer.write_table(table, row_group_size=row_group_size)
        for key in columns:
            buffered[key].clear()

    try:
        for local_index, game in iter_raw_games(shard_path, max_games=None):
            global_index = shard_entry.start_global_index + int(local_index)
            game_id = stable_game_id(parent_archive_sha256, global_index)
            resolved = resolve_canonical_played_date(
                date_header=game.headers.get("Date"),
                utc_date_header=game.headers.get("UTCDate"),
            )

            buffered["game_id"].append(game_id)
            buffered["source_month"].append(source_month)
            buffered["source_game_index"].append(global_index)
            buffered["canonical_played_date_iso"].append(resolved.canonical_played_date_iso)
            buffered["canonical_date_source"].append(resolved.canonical_date_source)
            buffered["date_header_raw"].append(resolved.date_header_raw)
            buffered["utc_date_header_raw"].append(resolved.utc_date_header_raw)
            buffered["date_header_valid"].append(resolved.date_header_valid)
            buffered["utc_date_header_valid"].append(resolved.utc_date_header_valid)
            buffered["resolver_version"].append(resolved.resolver_version)

            if resolved.canonical_date_source == "utc_date":
                canonical_from_utc_date += 1
            elif resolved.canonical_date_source == "date":
                canonical_from_date += 1
            else:
                canonical_missing_or_invalid += 1

            if (not resolved.date_header_valid) and resolved.utc_date_header_valid:
                invalid_date_but_valid_utc_date += 1

            row_count += 1
            if len(buffered["game_id"]) >= row_group_size:
                flush_buffer()

        flush_buffer()
    finally:
        if writer is not None:
            writer.close()

    expected = shard_entry.game_count
    if row_count != expected:
        if temp_path.exists():
            temp_path.unlink()
        raise DateEnrichmentError(
            "Shard game count mismatch while building date enrichment sidecar for "
            f"{shard_entry.filename}: expected {expected}, got {row_count}"
        )

    if not temp_path.exists():
        raise DateEnrichmentError(
            f"Date enrichment writer did not produce an output file: {temp_path.as_posix()}"
        )

    output_sha = compute_sha256(temp_path)
    output_bytes = temp_path.stat().st_size
    os.replace(temp_path, output_path)

    return DateEnrichmentShardEntry(
        shard_index=shard_entry.shard_index,
        shard_filename=shard_entry.filename,
        output_relpath=output_path.name,
        start_global_index=shard_entry.start_global_index,
        end_global_index=shard_entry.end_global_index,
        game_count=row_count,
        output_bytes=output_bytes,
        output_sha256=output_sha,
        canonical_from_utc_date=canonical_from_utc_date,
        canonical_from_date=canonical_from_date,
        canonical_missing_or_invalid=canonical_missing_or_invalid,
        invalid_date_but_valid_utc_date=invalid_date_but_valid_utc_date,
    )


def run_date_enrichment(
    *,
    shard_root: Path,
    source_month: str,
    output_root: Path,
    resume: bool = False,
    max_new_shards: int | None = None,
    row_group_size: int = 50000,
) -> DateEnrichmentResult:
    shard_dir = shard_root / source_month
    shard_manifest_path = shard_dir / SHARD_MANIFEST_FILENAME
    if not shard_manifest_path.exists():
        raise DateEnrichmentError(
            "Shard manifest not found; run sharding first: "
            f"{shard_manifest_path.as_posix()}"
        )

    shard_manifest = load_shard_manifest(shard_manifest_path)
    shard_entries = parse_shard_entries(shard_manifest)
    validate_shard_ranges(shard_entries)

    parent_archive_sha256 = str(shard_manifest.get("parent_archive_sha256", "")).strip()
    if not parent_archive_sha256:
        raise DateEnrichmentError("Shard manifest missing parent_archive_sha256")

    shard_identity = shard_manifest_identity_hash(shard_manifest)
    date_enrichment_id = derive_date_enrichment_id(
        parent_archive_sha256=parent_archive_sha256,
        source_month=source_month,
        shard_manifest_identity=shard_identity,
        resolver_version=CANONICAL_PLAYED_DATE_RESOLVER_VERSION,
    )

    dataset_root = output_root / date_enrichment_id
    manifest_path = _manifest_path(dataset_root)

    completed_entries: list[DateEnrichmentShardEntry] = []
    started_at_utc = _utc_now_iso()

    if manifest_path.exists():
        existing_manifest, existing_entries = _load_and_validate_existing_manifest(
            manifest_path=manifest_path,
            expected_date_enrichment_id=date_enrichment_id,
            expected_parent_archive_sha256=parent_archive_sha256,
            expected_source_month=source_month,
            expected_shard_manifest_identity=shard_identity,
        )
        _verify_entry_artifacts(dataset_root, existing_entries)

        selected_count = int(
            existing_manifest.get("selected_shard_count", len(shard_entries))
        )
        completed_count = int(
            existing_manifest.get("completed_shard_count", len(existing_entries))
        )
        status = str(existing_manifest.get("status", "incomplete"))

        if status == "complete":
            counts = existing_manifest.get("counts", {})
            enriched_games = int(counts.get("enriched_games", 0)) if isinstance(counts, dict) else 0
            timing = existing_manifest.get("timing", {})
            active_seconds = (
                float(timing.get("active_seconds", 0.0)) if isinstance(timing, dict) else 0.0
            )
            peak_rss_bytes = (
                int(timing.get("peak_rss_bytes", 0)) if isinstance(timing, dict) else 0
            )
            return DateEnrichmentResult(
                date_enrichment_id=date_enrichment_id,
                dataset_root=dataset_root,
                manifest_path=manifest_path,
                status="complete",
                reused_existing=True,
                selected_shard_count=selected_count,
                completed_shard_count=completed_count,
                enriched_games=enriched_games,
                new_shards_written=0,
                peak_rss_bytes=peak_rss_bytes,
                active_seconds=active_seconds,
            )

        if not resume:
            raise DateEnrichmentError(
                "Existing incomplete date enrichment manifest found; pass resume=True to continue"
            )

        completed_entries = existing_entries
        timing = existing_manifest.get("timing", {})
        if isinstance(timing, dict):
            existing_started = str(timing.get("started_at_utc", "")).strip()
            if existing_started:
                started_at_utc = existing_started

    dataset_root.mkdir(parents=True, exist_ok=True)
    shards_output_dir = dataset_root / "shards"
    shards_output_dir.mkdir(parents=True, exist_ok=True)

    plan_size = len(shard_entries)
    if len(completed_entries) > plan_size:
        raise DateEnrichmentError("Completed entries exceed shard plan size")

    for completed in sorted(completed_entries, key=lambda item: item.shard_index):
        expected = shard_entries[completed.shard_index]
        if completed.shard_index != expected.shard_index:
            raise DateEnrichmentError(
                "Date enrichment resume mismatch at shard index: "
                f"{completed.shard_index} != {expected.shard_index}"
            )
        if completed.start_global_index != expected.start_global_index:
            raise DateEnrichmentError(
                "Date enrichment resume mismatch for start_global_index at shard "
                f"{expected.shard_index}"
            )
        if completed.end_global_index != expected.end_global_index:
            raise DateEnrichmentError(
                "Date enrichment resume mismatch for end_global_index at shard "
                f"{expected.shard_index}"
            )

    pending_shards = shard_entries[len(completed_entries) :]
    if max_new_shards is not None:
        if max_new_shards < 0:
            raise DateEnrichmentError("max_new_shards must be non-negative")
        pending_shards = pending_shards[:max_new_shards]

    entries: list[DateEnrichmentShardEntry] = list(completed_entries)
    new_shards_written = 0

    sampler = PeakRssSampler(interval_seconds=0.05)
    sampler.start()
    started = time.perf_counter()

    def persist(status: str) -> None:
        active_seconds = max(time.perf_counter() - started, 0.0)
        payload = _build_manifest(
            date_enrichment_id=date_enrichment_id,
            status=status,
            shard_manifest=shard_manifest,
            shard_manifest_identity=shard_identity,
            selected_shard_count=plan_size,
            completed_shard_count=len(entries),
            started_at_utc=started_at_utc,
            updated_at_utc=_utc_now_iso(),
            entries=entries,
            peak_rss_bytes=sampler.peak_rss_bytes,
            active_seconds=active_seconds,
        )
        atomic_write_json(manifest_path, payload)

    try:
        persist("incomplete")

        for shard_entry in pending_shards:
            shard_path = shard_dir / shard_entry.filename
            if not shard_path.exists():
                raise DateEnrichmentError(
                    f"Shard file missing during enrichment: {shard_path.as_posix()}"
                )
            actual_shard_sha = compute_sha256(shard_path)
            if actual_shard_sha != shard_entry.sha256:
                raise DateEnrichmentError(
                    "Shard checksum mismatch before enrichment (tampered?): "
                    f"{shard_path.as_posix()}"
                )

            relpath = Path("shards") / f"shard-{shard_entry.shard_index:05d}.parquet"
            output_path = dataset_root / relpath

            output_entry = _write_sidecar_for_shard(
                shard_path=shard_path,
                shard_entry=shard_entry,
                parent_archive_sha256=parent_archive_sha256,
                source_month=source_month,
                output_path=output_path,
                row_group_size=row_group_size,
            )

            entry_with_relpath = DateEnrichmentShardEntry(
                shard_index=output_entry.shard_index,
                shard_filename=output_entry.shard_filename,
                output_relpath=relpath.as_posix(),
                start_global_index=output_entry.start_global_index,
                end_global_index=output_entry.end_global_index,
                game_count=output_entry.game_count,
                output_bytes=output_entry.output_bytes,
                output_sha256=output_entry.output_sha256,
                canonical_from_utc_date=output_entry.canonical_from_utc_date,
                canonical_from_date=output_entry.canonical_from_date,
                canonical_missing_or_invalid=output_entry.canonical_missing_or_invalid,
                invalid_date_but_valid_utc_date=output_entry.invalid_date_but_valid_utc_date,
                status=output_entry.status,
            )
            entries.append(entry_with_relpath)
            _validate_entry_ranges(entries)
            new_shards_written += 1
            persist("incomplete")

        is_complete = len(entries) == plan_size and str(shard_manifest.get("status")) == "complete"
        persist("complete" if is_complete else "incomplete")
    finally:
        peak_rss_bytes = sampler.stop()

    active_seconds = max(time.perf_counter() - started, 0.0)
    counts = _sum_counts(entries)
    status = (
        "complete"
        if (len(entries) == plan_size and str(shard_manifest.get("status")) == "complete")
        else "incomplete"
    )
    return DateEnrichmentResult(
        date_enrichment_id=date_enrichment_id,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        status=status,
        reused_existing=False,
        selected_shard_count=plan_size,
        completed_shard_count=len(entries),
        enriched_games=counts["enriched_games"],
        new_shards_written=new_shards_written,
        peak_rss_bytes=peak_rss_bytes,
        active_seconds=active_seconds,
    )


def date_enrichment_result_to_json(result: DateEnrichmentResult) -> dict[str, Any]:
    return {
        "date_enrichment_id": result.date_enrichment_id,
        "dataset_root": result.dataset_root.as_posix(),
        "manifest_path": result.manifest_path.as_posix(),
        "status": result.status,
        "reused_existing": result.reused_existing,
        "selected_shard_count": result.selected_shard_count,
        "completed_shard_count": result.completed_shard_count,
        "enriched_games": result.enriched_games,
        "new_shards_written": result.new_shards_written,
        "peak_rss_bytes": result.peak_rss_bytes,
        "active_seconds": round(result.active_seconds, 6),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build resumable UTCDate-first header enrichment sidecar"
    )
    parser.add_argument("--shard-root", default="data/raw_shards")
    parser.add_argument("--source-month", required=True)
    parser.add_argument("--output-root", default="data/processed/date_enrichment")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-shards", type=int, default=None)
    parser.add_argument("--row-group-size", type=int, default=50000)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = run_date_enrichment(
        shard_root=Path(args.shard_root),
        source_month=str(args.source_month),
        output_root=Path(args.output_root),
        resume=bool(args.resume),
        max_new_shards=args.max_new_shards,
        row_group_size=int(args.row_group_size),
    )
    print(json.dumps(date_enrichment_result_to_json(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()