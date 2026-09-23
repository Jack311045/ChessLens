"""Phase 1.2a batch-based Parquet ingestion runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import psutil

from chesslens.domain.records import INGESTION_PIPELINE_VERSION, IngestionErrorRecord
from chesslens.ingestion.config import IngestionConfig, load_ingestion_config
from chesslens.ingestion.parquet_writer import PartitionedParquetWriter
from chesslens.ingestion.pgn_reader import (
    GameParseError,
    compute_sha256,
    infer_source_month,
    is_complete_game,
    iter_raw_games,
    parse_game_to_records,
)
from chesslens.validation.dataset_validation import (
    DatasetValidationResult,
    validate_staged_dataset,
)

_PACKAGE_VERSION_NAMES = (
    "python-chess",
    "zstandard",
    "pyarrow",
    "duckdb",
    "polars",
    "psutil",
    "PyYAML",
)

_SOURCE_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


class StrictModeIngestionError(RuntimeError):
    """Raised when strict mode hits a recoverable per-game rejection."""

    def __init__(self, error_record: IngestionErrorRecord) -> None:
        super().__init__(
            "Strict mode rejected game "
            f"{error_record.source_game_index}: {error_record.error_code}"
        )
        self.error_record = error_record


@dataclass(frozen=True)
class IngestionRunResult:
    run_id: str
    dataset_id: str
    dataset_path: Path
    manifest_path: Path
    reused_existing: bool
    scanned_games: int
    accepted_games: int
    rejected_games: int
    emitted_moves: int
    error_records: int
    checksum_duration_seconds: float
    processing_and_validation_duration_seconds: float
    total_duration_seconds: float
    peak_rss_bytes: int
    part_counts: dict[str, int]
    duckdb_validation_result: str


class PeakRssSampler:
    """Cross-platform periodic RSS sampler."""

    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0

    def start(self) -> None:
        process = psutil.Process(os.getpid())

        def sampler() -> None:
            while not self._stop.is_set():
                try:
                    rss = int(process.memory_info().rss)
                    if rss > self._peak_rss_bytes:
                        self._peak_rss_bytes = rss
                finally:
                    self._stop.wait(self._interval_seconds)

        self._thread = threading.Thread(target=sampler, name="peak-rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self._peak_rss_bytes


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = output.strip()
    return commit if commit else None


def _runtime_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _PACKAGE_VERSION_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _resolve_source_month(config: IngestionConfig) -> str:
    if config.source_month is not None:
        return config.source_month
    inferred = infer_source_month(config.input_path.name)
    if inferred is None:
        raise ValueError(
            "Unable to infer source_month from archive filename; set source_month in config"
        )
    if not _SOURCE_MONTH_RE.fullmatch(inferred):
        raise ValueError(
            "Unable to parse inferred source_month from archive filename in YYYY-MM format; "
            "set source_month in config"
        )
    try:
        datetime.strptime(inferred, "%Y-%m")
    except ValueError as exc:
        raise ValueError(
            "Inferred source_month from archive filename is not a real calendar month: "
            f"{inferred!r}"
        ) from exc
    return inferred


def _resolve_player_hash_credentials(config: IngestionConfig) -> tuple[str | None, str | None]:
    if config.player_hash_mode == "fixture_placeholder":
        return None, None

    if config.player_hmac_key_env is None or config.player_hmac_key_id_env is None:
        raise ValueError("HMAC mode requires both player_hmac_key_env and player_hmac_key_id_env")

    secret = os.environ.get(config.player_hmac_key_env)
    if secret is None or not secret.strip():
        raise ValueError(
            f"Missing required environment variable for HMAC secret: {config.player_hmac_key_env}"
        )

    key_id = os.environ.get(config.player_hmac_key_id_env)
    if key_id is None or not key_id.strip():
        raise ValueError(
            "Missing required environment variable for HMAC key id: "
            f"{config.player_hmac_key_id_env}"
        )

    return secret, key_id.strip()


def _effective_config_payload(
    *,
    config: IngestionConfig,
    source_month: str,
    player_hmac_key_id: str | None,
) -> dict[str, Any]:
    return {
        "input_archive": config.input_path.name,
        "expected_source_sha256": config.expected_source_sha256,
        "max_games": config.max_games,
        "strict": config.strict,
        "require_complete_games": config.require_complete_games,
        "batch_games": config.batch_games,
        "max_buffered_records": config.max_buffered_records,
        "parquet_compression": config.parquet_compression,
        "parquet_row_group_size": config.parquet_row_group_size,
        "player_hash_mode": config.player_hash_mode,
        "player_hmac_key_id": player_hmac_key_id,
        "source_month": source_month,
        "schema_version": config.schema_version,
        "position_normalization_version": config.position_normalization_version,
        "board_encoding_version": config.board_encoding_version,
        "action_encoding_version": config.action_encoding_version,
        "ingestion_pipeline_version": INGESTION_PIPELINE_VERSION,
    }


def _build_dataset_id(
    *,
    source_archive_sha256: str,
    configuration_hash: str,
    schema_version: str,
    position_normalization_version: str,
    board_encoding_version: str,
    action_encoding_version: str,
    ingestion_pipeline_version: str,
    player_hmac_key_id: str | None,
) -> str:
    payload = {
        "source_archive_sha256": source_archive_sha256,
        "configuration_hash": configuration_hash,
        "schema_version": schema_version,
        "position_normalization_version": position_normalization_version,
        "board_encoding_version": board_encoding_version,
        "action_encoding_version": action_encoding_version,
        "ingestion_pipeline_version": ingestion_pipeline_version,
        "player_hmac_key_id": player_hmac_key_id,
    }
    return _sha256_text(_canonical_json(payload))


def _read_manifest_identity_field(
    manifest: dict[str, Any],
    path: tuple[str, ...],
) -> tuple[bool, Any]:
    current: Any = manifest
    for key in path:
        if not isinstance(current, dict):
            return False, None
        if key not in current:
            return False, None
        current = current[key]
    return True, current


def _validate_existing_manifest_identity(
    *,
    manifest: dict[str, Any],
    expected: dict[tuple[str, ...], Any],
) -> None:
    issues: list[str] = []
    for path, expected_value in expected.items():
        present, actual_value = _read_manifest_identity_field(manifest, path)
        dotted = ".".join(path)
        if not present:
            issues.append(f"missing identity field {dotted}")
            continue
        if actual_value != expected_value:
            issues.append(
                f"identity mismatch for {dotted}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )

    if issues:
        raise RuntimeError("Existing dataset manifest identity mismatch: " + "; ".join(issues))


def _load_existing_manifest(dataset_path: Path) -> dict[str, Any]:
    manifest_path = dataset_path / "_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Existing dataset is missing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid manifest JSON object in {manifest_path}")
    return cast(dict[str, Any], payload)


def _validate_existing_dataset(
    dataset_path: Path,
    manifest: dict[str, Any],
) -> DatasetValidationResult:
    counts = manifest.get("counts", {})
    source = manifest.get("source", {})
    return validate_staged_dataset(
        dataset_root=dataset_path,
        expected_source_month=str(source.get("source_month", "")),
        expected_games=int(counts.get("accepted_games", 0)),
        expected_moves=int(counts.get("emitted_moves", 0)),
        expected_errors=int(counts.get("error_records", 0)),
    )


def _incomplete_game_error(
    *,
    source_archive: str,
    source_game_index: int,
    run_id: str,
    result: str | None,
    ply_count: int,
) -> IngestionErrorRecord:
    return IngestionErrorRecord(
        source_archive=source_archive,
        source_game_index=source_game_index,
        error_code="incomplete_game",
        error_message=(
            "Game does not satisfy complete-game policy "
            f"(result={result!r}, ply_count={ply_count})"
        ),
        handling_decision="rejected",
        recoverability="recoverable",
        run_id=run_id,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded-memory Parquet ingestion")
    parser.add_argument("--config", default="configs/ingestion/fixture_etl.yaml")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--tolerant", action="store_true")
    return parser


def run_ingestion(
    *,
    config_path: Path,
    overrides: dict[str, Any] | None = None,
    simulate_failure_after_batches: int | None = None,
) -> IngestionRunResult:
    config = load_ingestion_config(config_path, overrides=overrides)

    started_at = _utc_now_iso()
    run_id = f"ingestion-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    source_month = _resolve_source_month(config)
    player_hmac_secret, player_hmac_key_id = _resolve_player_hash_credentials(config)

    total_started = time.perf_counter()
    checksum_started = time.perf_counter()
    source_archive_sha256 = compute_sha256(config.input_path)
    checksum_duration_seconds = max(time.perf_counter() - checksum_started, 0.0)
    if (
        config.expected_source_sha256 is not None
        and source_archive_sha256 != config.expected_source_sha256
    ):
        raise ValueError(
            "Source SHA-256 mismatch: "
            f"expected {config.expected_source_sha256}, got {source_archive_sha256}"
        )

    effective_config = _effective_config_payload(
        config=config,
        source_month=source_month,
        player_hmac_key_id=player_hmac_key_id,
    )
    configuration_hash = _sha256_text(_canonical_json(effective_config))
    dataset_id = _build_dataset_id(
        source_archive_sha256=source_archive_sha256,
        configuration_hash=configuration_hash,
        schema_version=config.schema_version,
        position_normalization_version=config.position_normalization_version,
        board_encoding_version=config.board_encoding_version,
        action_encoding_version=config.action_encoding_version,
        ingestion_pipeline_version=INGESTION_PIPELINE_VERSION,
        player_hmac_key_id=player_hmac_key_id,
    )

    output_root = config.output_root
    final_dataset_path = output_root / "datasets" / dataset_id
    if final_dataset_path.exists():
        existing_manifest = _load_existing_manifest(final_dataset_path)
        _validate_existing_manifest_identity(
            manifest=existing_manifest,
            expected={
                ("status",): "complete",
                ("dataset_id",): dataset_id,
                ("configuration_hash",): configuration_hash,
                ("source", "archive_sha256"): source_archive_sha256,
                ("source", "source_month"): source_month,
                ("versions", "schema_version"): config.schema_version,
                (
                    "versions",
                    "position_normalization_version",
                ): config.position_normalization_version,
                ("versions", "board_encoding_version"): config.board_encoding_version,
                ("versions", "action_encoding_version"): config.action_encoding_version,
                ("versions", "ingestion_pipeline_version"): INGESTION_PIPELINE_VERSION,
                ("versions", "player_hmac_key_id"): player_hmac_key_id,
            },
        )

        validation_result = _validate_existing_dataset(final_dataset_path, existing_manifest)
        counts = existing_manifest["counts"]
        performance = existing_manifest.get("performance", {})
        return IngestionRunResult(
            run_id=str(existing_manifest.get("run_id", run_id)),
            dataset_id=dataset_id,
            dataset_path=final_dataset_path,
            manifest_path=final_dataset_path / "_manifest.json",
            reused_existing=True,
            scanned_games=int(counts.get("scanned_games", 0)),
            accepted_games=int(counts.get("accepted_games", 0)),
            rejected_games=int(counts.get("rejected_games", 0)),
            emitted_moves=int(counts.get("emitted_moves", 0)),
            error_records=int(counts.get("error_records", 0)),
            checksum_duration_seconds=float(performance.get("checksum_duration_seconds", 0.0)),
            processing_and_validation_duration_seconds=float(
                performance.get(
                    "processing_and_validation_duration_seconds",
                    performance.get("duration_seconds", 0.0),
                )
            ),
            total_duration_seconds=float(
                performance.get("total_duration_seconds", performance.get("duration_seconds", 0.0))
            ),
            peak_rss_bytes=int(performance.get("peak_rss_bytes", 0)),
            part_counts={
                name: int(details.get("part_count", 0))
                for name, details in existing_manifest.get("datasets", {}).items()
            },
            duckdb_validation_result=(
                f"reused-existing-valid (games={validation_result.games_rows}, "
                f"moves={validation_result.moves_rows})"
            ),
        )

    staging_dataset_path = output_root / "staging" / f"{dataset_id}__{run_id}"
    if staging_dataset_path.exists():
        raise RuntimeError(f"Staging path already exists: {staging_dataset_path}")

    staging_dataset_path.mkdir(parents=True, exist_ok=False)

    memory_sampler = PeakRssSampler()
    memory_sampler.start()

    processing_started = time.perf_counter()
    scanned_games = 0
    accepted_games = 0
    rejected_games = 0
    emitted_moves = 0
    batches_written = 0

    game_buffer: list[Any] = []
    move_buffer: list[Any] = []
    error_buffer: list[Any] = []

    writer = PartitionedParquetWriter(
        dataset_root=staging_dataset_path,
        source_month=source_month,
        compression=config.parquet_compression,
        row_group_size=config.parquet_row_group_size,
    )

    def buffered_record_count() -> int:
        return len(game_buffer) + len(move_buffer) + len(error_buffer)

    def flush_buffers() -> None:
        nonlocal batches_written
        if not game_buffer and not move_buffer and not error_buffer:
            return
        writer.write_records("games", game_buffer)
        writer.write_records("moves", move_buffer)
        writer.write_records("ingestion_errors", error_buffer)
        game_buffer.clear()
        move_buffer.clear()
        error_buffer.clear()
        batches_written += 1
        if (
            simulate_failure_after_batches is not None
            and batches_written >= simulate_failure_after_batches
        ):
            raise RuntimeError("Simulated failure after batch write")

    try:
        for source_game_index, raw_game in iter_raw_games(
            config.input_path,
            max_games=config.max_games,
        ):
            scanned_games += 1
            try:
                parsed = parse_game_to_records(
                    game=raw_game,
                    source_archive=config.input_path.name,
                    source_archive_sha256=source_archive_sha256,
                    source_game_index=source_game_index,
                    schema_version=config.schema_version,
                    run_id=run_id,
                    player_hash_mode=config.player_hash_mode,
                    player_hash_secret=player_hmac_secret,
                )
            except GameParseError as exc:
                rejected_games += 1
                error_buffer.append(exc.error_record)
                if config.strict:
                    raise StrictModeIngestionError(exc.error_record) from exc
                if (
                    len(game_buffer) >= config.batch_games
                    or buffered_record_count() >= config.max_buffered_records
                ):
                    flush_buffers()
                continue

            game_record = parsed.game_record
            if game_record.source_month != source_month:
                game_record = replace(game_record, source_month=source_month)

            if config.require_complete_games and not is_complete_game(game_record):
                error_record = _incomplete_game_error(
                    source_archive=config.input_path.name,
                    source_game_index=source_game_index,
                    run_id=run_id,
                    result=game_record.result,
                    ply_count=game_record.ply_count,
                )
                rejected_games += 1
                error_buffer.append(error_record)
                if config.strict:
                    raise StrictModeIngestionError(error_record)
                if (
                    len(game_buffer) >= config.batch_games
                    or buffered_record_count() >= config.max_buffered_records
                ):
                    flush_buffers()
                continue

            accepted_games += 1
            emitted_moves += len(parsed.move_records)
            game_buffer.append(game_record)
            move_buffer.extend(parsed.move_records)

            if (
                len(game_buffer) >= config.batch_games
                or buffered_record_count() >= config.max_buffered_records
            ):
                flush_buffers()

        flush_buffers()

        writer.ensure_schema_safe_empty_parts()
        validation_result = validate_staged_dataset(
            dataset_root=staging_dataset_path,
            expected_source_month=source_month,
            expected_games=accepted_games,
            expected_moves=emitted_moves,
            expected_errors=rejected_games,
        )

        peak_rss_bytes = memory_sampler.stop()
        processing_and_validation_duration_seconds = max(
            time.perf_counter() - processing_started,
            0.0,
        )

        stats = writer.stats()
        total_duration_seconds = max(time.perf_counter() - total_started, 0.0)
        manifest = {
            "manifest_version": "1.0.0",
            "dataset_id": dataset_id,
            "run_id": run_id,
            "status": "complete",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now_iso(),
            "git_commit": _git_commit(),
            "source": {
                "archive_filename": config.input_path.name,
                "archive_size_bytes": config.input_path.stat().st_size,
                "archive_sha256": source_archive_sha256,
                "source_month": source_month,
            },
            "configuration_hash": configuration_hash,
            "effective_config": effective_config,
            "versions": {
                "schema_version": config.schema_version,
                "position_normalization_version": config.position_normalization_version,
                "board_encoding_version": config.board_encoding_version,
                "action_encoding_version": config.action_encoding_version,
                "ingestion_pipeline_version": INGESTION_PIPELINE_VERSION,
                "player_hmac_key_id": player_hmac_key_id,
            },
            "counts": {
                "requested_raw_game_limit": config.max_games,
                "scanned_games": scanned_games,
                "accepted_games": accepted_games,
                "rejected_games": rejected_games,
                "emitted_moves": emitted_moves,
                "error_records": rejected_games,
            },
            "datasets": {
                dataset_name: {
                    "part_count": dataset_stats.part_count,
                    "row_count": dataset_stats.row_count,
                    "total_bytes": dataset_stats.total_bytes,
                    "files": dataset_stats.relative_files,
                }
                for dataset_name, dataset_stats in stats.items()
            },
            "total_output_bytes": writer.total_output_bytes,
            "performance": {
                "checksum_duration_seconds": round(checksum_duration_seconds, 6),
                "processing_and_validation_duration_seconds": round(
                    processing_and_validation_duration_seconds,
                    6,
                ),
                "total_duration_seconds": round(total_duration_seconds, 6),
                "processing_games_per_second": round(
                    scanned_games / processing_and_validation_duration_seconds,
                    4,
                )
                if processing_and_validation_duration_seconds > 0
                else 0.0,
                "processing_moves_per_second": round(
                    emitted_moves / processing_and_validation_duration_seconds,
                    4,
                )
                if processing_and_validation_duration_seconds > 0
                else 0.0,
                "total_games_per_second": round(scanned_games / total_duration_seconds, 4)
                if total_duration_seconds > 0
                else 0.0,
                "total_moves_per_second": round(emitted_moves / total_duration_seconds, 4)
                if total_duration_seconds > 0
                else 0.0,
                "peak_rss_bytes": peak_rss_bytes,
            },
            "runtime": {
                "python_version": platform.python_version(),
                "package_versions": _runtime_package_versions(),
            },
            "duckdb_validation": {
                "status": "passed",
                "checks": validation_result.checks,
            },
        }

        manifest_path = staging_dataset_path / "_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        final_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if final_dataset_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing dataset path: {final_dataset_path}"
            )
        staging_dataset_path.rename(final_dataset_path)

        return IngestionRunResult(
            run_id=run_id,
            dataset_id=dataset_id,
            dataset_path=final_dataset_path,
            manifest_path=final_dataset_path / "_manifest.json",
            reused_existing=False,
            scanned_games=scanned_games,
            accepted_games=accepted_games,
            rejected_games=rejected_games,
            emitted_moves=emitted_moves,
            error_records=rejected_games,
            checksum_duration_seconds=checksum_duration_seconds,
            processing_and_validation_duration_seconds=processing_and_validation_duration_seconds,
            total_duration_seconds=total_duration_seconds,
            peak_rss_bytes=peak_rss_bytes,
            part_counts={name: dataset_stats.part_count for name, dataset_stats in stats.items()},
            duckdb_validation_result="passed",
        )
    except Exception as exc:
        peak_rss_bytes = memory_sampler.stop()
        failed_total_duration_seconds = max(time.perf_counter() - total_started, 0.0)
        failed_manifest = {
            "manifest_version": "1.0.0",
            "dataset_id": dataset_id,
            "run_id": run_id,
            "status": "failed",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now_iso(),
            "error": str(exc),
            "source": {
                "archive_filename": config.input_path.name,
                "archive_sha256": source_archive_sha256,
                "source_month": source_month,
            },
            "counts": {
                "requested_raw_game_limit": config.max_games,
                "scanned_games": scanned_games,
                "accepted_games": accepted_games,
                "rejected_games": rejected_games,
                "emitted_moves": emitted_moves,
            },
            "performance": {
                "checksum_duration_seconds": round(checksum_duration_seconds, 6),
                "processing_and_validation_duration_seconds": round(
                    max(time.perf_counter() - processing_started, 0.0),
                    6,
                ),
                "total_duration_seconds": round(failed_total_duration_seconds, 6),
                "peak_rss_bytes": peak_rss_bytes,
            },
        }
        (staging_dataset_path / "_manifest.json").write_text(
            json.dumps(failed_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.strict and args.tolerant:
        raise ValueError("Choose either --strict or --tolerant, not both")

    overrides: dict[str, Any] = {}
    if args.input_path is not None:
        overrides["input_path"] = args.input_path
    if args.output_root is not None:
        overrides["output_root"] = args.output_root
    if args.max_games is not None:
        overrides["max_games"] = args.max_games
    if args.strict:
        overrides["strict"] = True
    if args.tolerant:
        overrides["strict"] = False

    result = run_ingestion(
        config_path=Path(args.config),
        overrides=overrides if overrides else None,
    )
    print(
        "\n".join(
            [
                f"dataset_id={result.dataset_id}",
                f"dataset_path={result.dataset_path.as_posix()}",
                f"reused_existing={result.reused_existing}",
                f"scanned_games={result.scanned_games}",
                f"accepted_games={result.accepted_games}",
                f"rejected_games={result.rejected_games}",
                f"emitted_moves={result.emitted_moves}",
                f"error_records={result.error_records}",
                f"part_counts={result.part_counts}",
                f"checksum_duration_seconds={result.checksum_duration_seconds:.4f}",
                (
                    "processing_and_validation_duration_seconds="
                    f"{result.processing_and_validation_duration_seconds:.4f}"
                ),
                f"total_duration_seconds={result.total_duration_seconds:.4f}",
                f"peak_rss_bytes={result.peak_rss_bytes}",
                f"duckdb_validation={result.duckdb_validation_result}",
                f"manifest={result.manifest_path.as_posix()}",
            ]
        )
    )


if __name__ == "__main__":
    main()
