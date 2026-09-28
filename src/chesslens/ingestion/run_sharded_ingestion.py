"""Resumable, lock-safe sharded ingestion orchestrator (Phase 1.2d).

This coordinator supports shard-level multiprocessing while preserving the existing
identity and reproducibility contracts:

- workers process disjoint shard indices only;
- each worker reuses the single-shard run_ingestion path with SourceLineage;
- only the parent process writes the collection manifest;
- collection entries are committed in shard-index order (contiguous prefix only).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import yaml

from chesslens.ingestion.collection_manifest import (
    COLLECTION_MANIFEST_FILENAME,
    CollectionShardEntry,
    build_collection_manifest_dict,
    derive_collection_id,
    load_collection_manifest,
    parse_collection_entries,
    validate_collection_reconciliation,
)
from chesslens.ingestion.pgn_reader import compute_sha256
from chesslens.ingestion.run_ingestion import run_ingestion
from chesslens.ingestion.shard_estimates import estimate_sizes
from chesslens.ingestion.shard_manifest import (
    SHARD_MANIFEST_FILENAME,
    ShardEntry,
    atomic_write_json,
    load_shard_manifest,
    parse_shard_entries,
    shard_manifest_identity_hash,
    validate_shard_ranges,
)
from chesslens.ingestion.source_lineage import SourceLineage
from chesslens.warehouse.preflight import validate_published_dataset_root

_TEN_MILLION = 10_000_000
_COLLECTION_LOCK_FILENAME = ".collection_ingestion.lock"
_HEARTBEAT_INTERVAL_SECONDS = 10.0


class ShardedIngestionError(RuntimeError):
    """Raised when sharded ingestion cannot proceed safely."""


@dataclass(frozen=True)
class OrchestratorConfig:
    ingestion_config_path: Path
    source_month: str
    shard_root: Path
    collection_root: Path
    games_per_shard: int
    expected_raw_games: int | None
    player_hash_mode: str
    player_hmac_key_env: str | None
    player_hmac_key_id_env: str | None


@dataclass(frozen=True)
class WorkerTestBehavior:
    """Optional test hooks for deterministic worker ordering/failure tests."""

    sleep_seconds_by_shard_index: dict[int, float] = field(default_factory=dict)
    fail_shard_indices: set[int] = field(default_factory=set)

    def sleep_for(self, shard_index: int) -> float:
        value = self.sleep_seconds_by_shard_index.get(shard_index, 0.0)
        return max(float(value), 0.0)

    def should_fail(self, shard_index: int) -> bool:
        return shard_index in self.fail_shard_indices


@dataclass(frozen=True)
class _Plan:
    shard_dir: Path
    manifest: dict[str, Any]
    entries: list[ShardEntry]
    parent_filename: str
    parent_sha256: str


@dataclass(frozen=True)
class _WorkerTask:
    ingestion_config_path: str
    source_month: str
    shard_path: str
    shard_sha256: str
    shard_index: int
    shard_filename: str
    parent_filename: str
    parent_sha256: str
    start_global_index: int
    end_global_index: int
    collection_dir: str
    shard_output_root: str
    sleep_seconds: float
    fail_before_run: bool


@dataclass(frozen=True)
class _WorkerResult:
    shard_index: int
    worker_pid: int
    success: bool
    error_message: str | None
    dataset_id: str | None
    dataset_relpath: str | None
    scanned_games: int
    accepted_games: int
    rejected_games: int
    emitted_moves: int
    error_records: int
    output_bytes: int
    processing_active_seconds: float
    peak_rss_bytes: int
    reused_existing: bool


class _ProcessTreeRssSampler:
    """Samples current process + child process RSS and stores the peak."""

    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0

    def _sample_once(self) -> None:
        root = psutil.Process(os.getpid())
        total_rss = int(root.memory_info().rss)
        for child in root.children(recursive=True):
            try:
                total_rss += int(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if total_rss > self._peak_rss_bytes:
            self._peak_rss_bytes = total_rss

    def start(self) -> None:
        def sampler() -> None:
            while not self._stop.is_set():
                try:
                    self._sample_once()
                finally:
                    self._stop.wait(self._interval_seconds)

        self._thread = threading.Thread(target=sampler, name="process-tree-rss", daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self._peak_rss_bytes


class _CollectionMutationLock:
    """Single-writer lock for collection mutation operations."""

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._token = uuid.uuid4().hex
        self._acquired = False

    def __enter__(self) -> _CollectionMutationLock:
        self.acquire()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: Any,
    ) -> None:
        self.release()

    def acquire(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "process_create_time": psutil.Process(os.getpid()).create_time(),
            "hostname": socket.gethostname(),
            "acquired_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "token": self._token,
        }

        for _ in range(3):
            try:
                fd = os.open(
                    str(self._lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError as exc:
                if self._recover_stale_lock():
                    continue
                holder = self._describe_current_holder()
                raise ShardedIngestionError(
                    "Another mutating sharded-ingestion coordinator is active for this "
                    f"collection ({holder})."
                ) from exc

            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
            self._acquired = True
            return

        raise ShardedIngestionError(
            f"Unable to acquire collection lock: {self._lock_path.as_posix()}"
        )

    def release(self) -> None:
        if not self._acquired:
            return
        current = self._read_lock_payload()
        if current is not None and current.get("token") != self._token:
            self._acquired = False
            return
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass
        self._acquired = False

    def _read_lock_payload(self) -> dict[str, Any] | None:
        if not self._lock_path.exists():
            return None
        try:
            payload = json.loads(self._lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _recover_stale_lock(self) -> bool:
        payload = self._read_lock_payload()
        if payload is not None and not self._is_stale(payload):
            return False
        try:
            self._lock_path.unlink()
            _log(
                "Recovered stale collection lock: "
                f"{self._lock_path.as_posix()}"
            )
        except FileNotFoundError:
            pass
        return True

    def _is_stale(self, payload: dict[str, Any]) -> bool:
        raw_pid = payload.get("pid")
        raw_create_time = payload.get("process_create_time")
        if raw_pid is None:
            return True
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            return True
        if pid <= 0:
            return True
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return True
        except psutil.AccessDenied:
            return False

        if raw_create_time is None:
            return False

        try:
            expected_create_time = float(raw_create_time)
        except (TypeError, ValueError):
            return True
        try:
            actual_create_time = float(process.create_time())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return True
        return abs(actual_create_time - expected_create_time) > 1.0

    def _describe_current_holder(self) -> str:
        payload = self._read_lock_payload()
        if payload is None:
            return "lock present"
        pid = payload.get("pid")
        host = payload.get("hostname")
        acquired = payload.get("acquired_at_utc")
        return f"pid={pid}, host={host}, acquired_at={acquired}"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load_orchestrator_config(config_path: Path) -> OrchestratorConfig:
    if not config_path.exists():
        raise ShardedIngestionError(f"Orchestrator config not found: {config_path}")
    raw_loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw: dict[str, Any] = dict(raw_loaded) if isinstance(raw_loaded, dict) else {}

    source_month = str(raw.get("source_month", "")).strip()
    if not source_month:
        raise ShardedIngestionError("Orchestrator config must include source_month")

    return OrchestratorConfig(
        ingestion_config_path=config_path,
        source_month=source_month,
        shard_root=_resolve_path(str(raw.get("shard_root", "data/raw_shards"))),
        collection_root=_resolve_path(
            str(raw.get("collection_root", "data/processed/collections"))
        ),
        games_per_shard=int(raw.get("games_per_shard", 250000)),
        expected_raw_games=(
            int(raw["expected_raw_games"]) if raw.get("expected_raw_games") is not None else None
        ),
        player_hash_mode=str(raw.get("player_hash_mode", "fixture_placeholder")).strip(),
        player_hmac_key_env=raw.get("player_hmac_key_env"),
        player_hmac_key_id_env=raw.get("player_hmac_key_id_env"),
    )


def _resolve_player_key_id(
    config: OrchestratorConfig,
    *,
    require_secret: bool,
) -> str | None:
    if config.player_hash_mode == "fixture_placeholder":
        return None

    if not config.player_hmac_key_env or not config.player_hmac_key_id_env:
        raise ShardedIngestionError(
            "hmac_sha256 mode requires player_hmac_key_env and player_hmac_key_id_env"
        )

    if require_secret:
        secret = os.environ.get(config.player_hmac_key_env)
        if not secret or not secret.strip():
            raise ShardedIngestionError(
                f"Missing HMAC secret environment variable: {config.player_hmac_key_env}"
            )

    key_id = os.environ.get(config.player_hmac_key_id_env)
    if not key_id or not key_id.strip():
        raise ShardedIngestionError(
            f"Missing HMAC key-id environment variable: {config.player_hmac_key_id_env}"
        )
    return key_id.strip()


def _load_plan(config: OrchestratorConfig) -> _Plan:
    shard_dir = config.shard_root / config.source_month
    manifest_path = shard_dir / SHARD_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ShardedIngestionError(
            f"Shard plan not found; run run_sharding first: {manifest_path.as_posix()}"
        )
    manifest = load_shard_manifest(manifest_path)
    entries = parse_shard_entries(manifest)
    validate_shard_ranges(entries)
    return _Plan(
        shard_dir=shard_dir,
        manifest=manifest,
        entries=entries,
        parent_filename=str(manifest.get("parent_archive_filename", "")),
        parent_sha256=str(manifest.get("parent_archive_sha256", "")),
    )


def _verify_selected_shards(plan: _Plan) -> None:
    for entry in plan.entries:
        shard_path = plan.shard_dir / entry.filename
        if not shard_path.exists():
            raise ShardedIngestionError(f"Selected shard missing: {shard_path.as_posix()}")
        if compute_sha256(shard_path) != entry.sha256:
            raise ShardedIngestionError(
                f"Selected shard checksum mismatch (tampered?): {shard_path.as_posix()}"
            )


def _read_output_bytes(dataset_path: Path) -> int:
    manifest_path = dataset_path / "_manifest.json"
    if not manifest_path.exists():
        return 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return int(payload.get("total_output_bytes", 0))
    return 0


def _collection_paths(config: OrchestratorConfig, collection_id: str) -> tuple[Path, Path, Path]:
    collection_dir = config.collection_root / collection_id
    shards_dir = collection_dir / "shards"
    manifest_path = collection_dir / COLLECTION_MANIFEST_FILENAME
    return collection_dir, shards_dir, manifest_path


def _load_existing_collection(
    manifest_path: Path,
) -> tuple[dict[int, CollectionShardEntry], str | None, dict[str, Any] | None]:
    if not manifest_path.exists():
        return {}, None, None

    manifest = load_collection_manifest(manifest_path)
    validate_collection_reconciliation(manifest)
    started_at = str(manifest.get("timing", {}).get("started_at_utc", "")).strip()
    entries = {entry.shard_index: entry for entry in parse_collection_entries(manifest)}
    return entries, (started_at or None), manifest


def _validate_existing_collection_members(
    collection_dir: Path,
    entries: dict[int, CollectionShardEntry],
) -> None:
    for entry in sorted(entries.values(), key=lambda item: item.shard_index):
        member_root = collection_dir / entry.dataset_relpath
        if not member_root.is_dir():
            raise ShardedIngestionError(
                "Collection member dataset missing: "
                f"{member_root.as_posix()}"
            )
        manifest_path = member_root / "_manifest.json"
        if not manifest_path.exists():
            raise ShardedIngestionError(
                "Collection member manifest missing: "
                f"{manifest_path.as_posix()}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ShardedIngestionError(
                "Collection member manifest must be a JSON object: "
                f"{manifest_path.as_posix()}"
            )
        if str(payload.get("status")) != "complete":
            raise ShardedIngestionError(
                "Collection member is not complete: "
                f"{manifest_path.as_posix()}"
            )
        if str(payload.get("dataset_id")) != entry.dataset_id:
            raise ShardedIngestionError(
                "Collection member dataset_id mismatch: "
                f"{manifest_path.as_posix()}"
            )

        counts = payload.get("counts")
        if not isinstance(counts, dict):
            raise ShardedIngestionError(
                "Collection member counts are missing: "
                f"{manifest_path.as_posix()}"
            )
        if int(counts.get("scanned_games", -1)) != entry.scanned_games:
            raise ShardedIngestionError(
                "Collection member scanned_games mismatch: "
                f"{manifest_path.as_posix()}"
            )
        if int(counts.get("accepted_games", -1)) != entry.accepted_games:
            raise ShardedIngestionError(
                "Collection member accepted_games mismatch: "
                f"{manifest_path.as_posix()}"
            )
        if int(counts.get("rejected_games", -1)) != entry.rejected_games:
            raise ShardedIngestionError(
                "Collection member rejected_games mismatch: "
                f"{manifest_path.as_posix()}"
            )
        if int(counts.get("emitted_moves", -1)) != entry.emitted_moves:
            raise ShardedIngestionError(
                "Collection member emitted_moves mismatch: "
                f"{manifest_path.as_posix()}"
            )
        if int(counts.get("error_records", -1)) != entry.error_records:
            raise ShardedIngestionError(
                "Collection member error_records mismatch: "
                f"{manifest_path.as_posix()}"
            )

        lineage = payload.get("source_lineage")
        if isinstance(lineage, dict):
            if int(lineage.get("shard_index", -1)) != entry.shard_index:
                raise ShardedIngestionError(
                    "Collection member lineage shard_index mismatch: "
                    f"{manifest_path.as_posix()}"
                )
            if int(lineage.get("global_game_index_offset", -1)) != entry.start_global_index:
                raise ShardedIngestionError(
                    "Collection member lineage global index mismatch: "
                    f"{manifest_path.as_posix()}"
                )


def _persist_collection(
    *,
    config: OrchestratorConfig,
    plan: _Plan,
    collection_id: str,
    manifest_path: Path,
    entries: dict[int, CollectionShardEntry],
    key_id: str | None,
    started_at: str,
    run_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed = len(entries)
    total_plan_shards = len(plan.entries)
    plan_complete = plan.manifest.get("status") == "complete"
    status = "complete" if (plan_complete and completed == total_plan_shards) else "incomplete"
    manifest = build_collection_manifest_dict(
        collection_id=collection_id,
        status=status,
        parent_archive_filename=plan.parent_filename,
        parent_archive_sha256=plan.parent_sha256,
        source_month=config.source_month,
        shard_manifest_identity_hash=shard_manifest_identity_hash(plan.manifest),
        games_per_shard=config.games_per_shard,
        player_hmac_key_id=key_id,
        expected_raw_games=config.expected_raw_games,
        selected_shard_count=total_plan_shards,
        completed_shard_count=completed,
        started_at_utc=started_at,
        updated_at_utc=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        entries=list(entries.values()),
    )
    accepted = int(manifest["counts"]["accepted_games"])
    manifest["portfolio_10m_satisfied"] = bool(
        status == "complete" and accepted >= _TEN_MILLION
    )
    if run_metrics is not None:
        manifest["last_run"] = run_metrics
    atomic_write_json(manifest_path, manifest)
    return manifest


def _select_pending_shards(
    *,
    plan_entries: list[ShardEntry],
    completed_shard_count: int,
    max_new_shards: int | None,
) -> list[ShardEntry]:
    pending = plan_entries[completed_shard_count:]
    if max_new_shards is None:
        return pending
    return pending[:max_new_shards]


def _build_worker_task(
    *,
    config: OrchestratorConfig,
    plan: _Plan,
    collection_dir: Path,
    shard: ShardEntry,
    behavior: WorkerTestBehavior,
) -> _WorkerTask:
    return _WorkerTask(
        ingestion_config_path=str(config.ingestion_config_path),
        source_month=config.source_month,
        shard_path=str(plan.shard_dir / shard.filename),
        shard_sha256=shard.sha256,
        shard_index=shard.shard_index,
        shard_filename=shard.filename,
        parent_filename=plan.parent_filename,
        parent_sha256=plan.parent_sha256,
        start_global_index=shard.start_global_index,
        end_global_index=shard.end_global_index,
        collection_dir=str(collection_dir),
        shard_output_root=str(collection_dir / "shards" / f"shard-{shard.shard_index:05d}"),
        sleep_seconds=behavior.sleep_for(shard.shard_index),
        fail_before_run=behavior.should_fail(shard.shard_index),
    )


def _run_single_shard_task(task: _WorkerTask) -> _WorkerResult:
    worker_pid = os.getpid()
    print(
        (
            "[sharded-ingest worker] "
            f"start shard_index={task.shard_index} pid={worker_pid}"
        ),
        file=sys.stderr,
        flush=True,
    )

    try:
        if task.sleep_seconds > 0:
            time.sleep(task.sleep_seconds)
        if task.fail_before_run:
            raise RuntimeError(
                f"Injected worker failure for shard_index={task.shard_index}"
            )

        lineage = SourceLineage(
            identity_archive_name=task.parent_filename,
            identity_archive_sha256=task.parent_sha256,
            global_game_index_offset=task.start_global_index,
            shard_index=task.shard_index,
            shard_filename=task.shard_filename,
            shard_sha256=task.shard_sha256,
        )
        result = run_ingestion(
            config_path=Path(task.ingestion_config_path),
            overrides={
                "input_path": task.shard_path,
                "output_root": task.shard_output_root,
                "expected_source_sha256": task.shard_sha256,
                "source_month": task.source_month,
                "max_games": None,
            },
            source_lineage=lineage,
        )

        dataset_relpath = result.dataset_path.relative_to(Path(task.collection_dir)).as_posix()
        return _WorkerResult(
            shard_index=task.shard_index,
            worker_pid=worker_pid,
            success=True,
            error_message=None,
            dataset_id=result.dataset_id,
            dataset_relpath=dataset_relpath,
            scanned_games=result.scanned_games,
            accepted_games=result.accepted_games,
            rejected_games=result.rejected_games,
            emitted_moves=result.emitted_moves,
            error_records=result.error_records,
            output_bytes=_read_output_bytes(result.dataset_path),
            processing_active_seconds=result.processing_and_validation_duration_seconds,
            peak_rss_bytes=result.peak_rss_bytes,
            reused_existing=result.reused_existing,
        )
    except Exception as exc:
        return _WorkerResult(
            shard_index=task.shard_index,
            worker_pid=worker_pid,
            success=False,
            error_message=f"{type(exc).__name__}: {exc}",
            dataset_id=None,
            dataset_relpath=None,
            scanned_games=0,
            accepted_games=0,
            rejected_games=0,
            emitted_moves=0,
            error_records=0,
            output_bytes=0,
            processing_active_seconds=0.0,
            peak_rss_bytes=0,
            reused_existing=False,
        )


def _entry_from_worker_result(
    *,
    result: _WorkerResult,
    shard: ShardEntry,
) -> CollectionShardEntry:
    if not result.success or result.dataset_id is None or result.dataset_relpath is None:
        raise ShardedIngestionError(
            f"Cannot build collection entry from failed worker result {result.shard_index}"
        )
    return CollectionShardEntry(
        shard_index=shard.shard_index,
        dataset_id=result.dataset_id,
        dataset_relpath=result.dataset_relpath,
        start_global_index=shard.start_global_index,
        end_global_index=shard.end_global_index,
        scanned_games=result.scanned_games,
        accepted_games=result.accepted_games,
        rejected_games=result.rejected_games,
        emitted_moves=result.emitted_moves,
        error_records=result.error_records,
        output_bytes=result.output_bytes,
        processing_active_seconds=result.processing_active_seconds,
        peak_rss_bytes=result.peak_rss_bytes,
    )


def _compute_run_metrics(
    *,
    worker_count: int,
    selected_shard_indices: list[int],
    new_shards_processed: int,
    wall_clock_duration_seconds: float,
    successful_results: list[_WorkerResult],
    committed_entries: list[CollectionShardEntry],
    peak_process_tree_rss_bytes: int,
    failed_shard_indices: list[int],
    successful_uncommitted_shard_indices: list[int],
    baseline_wall_clock_seconds: float | None,
) -> dict[str, Any]:
    per_worker_peak_rss: dict[str, int] = {}
    for result in successful_results:
        key = str(result.worker_pid)
        per_worker_peak_rss[key] = max(per_worker_peak_rss.get(key, 0), result.peak_rss_bytes)

    max_worker_peak_rss = max(per_worker_peak_rss.values(), default=0)
    sum_worker_active = sum(result.processing_active_seconds for result in successful_results)
    committed_games = sum(entry.accepted_games for entry in committed_entries)
    committed_moves = sum(entry.emitted_moves for entry in committed_entries)
    games_per_wall_clock = (
        committed_games / wall_clock_duration_seconds if wall_clock_duration_seconds > 0 else 0.0
    )
    moves_per_wall_clock = (
        committed_moves / wall_clock_duration_seconds if wall_clock_duration_seconds > 0 else 0.0
    )

    speedup: float | None = None
    efficiency: float | None = None
    if (
        baseline_wall_clock_seconds is not None
        and baseline_wall_clock_seconds > 0
        and wall_clock_duration_seconds > 0
    ):
        speedup = baseline_wall_clock_seconds / wall_clock_duration_seconds
        efficiency = speedup / worker_count if worker_count > 0 else None

    return {
        "worker_count": worker_count,
        "selected_shard_count": len(selected_shard_indices),
        "selected_shard_indices": selected_shard_indices,
        "new_shards_processed": new_shards_processed,
        "wall_clock_duration_seconds": round(wall_clock_duration_seconds, 6),
        "sum_worker_active_duration_seconds": round(sum_worker_active, 6),
        "games_per_wall_clock_second": round(games_per_wall_clock, 4),
        "moves_per_wall_clock_second": round(moves_per_wall_clock, 4),
        "per_worker_peak_rss_bytes": per_worker_peak_rss,
        "max_worker_peak_rss_bytes": int(max_worker_peak_rss),
        "peak_process_tree_rss_bytes": int(peak_process_tree_rss_bytes),
        "failed_shard_indices": failed_shard_indices,
        "successful_uncommitted_shard_indices": successful_uncommitted_shard_indices,
        "speedup_vs_workers_1": round(speedup, 6) if speedup is not None else None,
        "parallel_efficiency_vs_workers_1": (
            round(efficiency, 6) if efficiency is not None else None
        ),
    }


def _validate_workers(workers: int) -> None:
    if workers < 1:
        raise ShardedIngestionError("workers must be >= 1")


def _validate_max_new_shards(max_new_shards: int | None) -> None:
    if max_new_shards is not None and max_new_shards < 0:
        raise ShardedIngestionError("max_new_shards must be >= 0 when provided")


def run_sharded_ingestion(
    config_path: Path,
    *,
    max_new_shards: int | None = None,
    resume: bool = False,
    workers: int = 1,
    baseline_wall_clock_seconds: float | None = None,
    test_worker_behavior: WorkerTestBehavior | None = None,
) -> dict[str, Any]:
    _validate_workers(workers)
    _validate_max_new_shards(max_new_shards)

    config = load_orchestrator_config(config_path)
    plan = _load_plan(config)
    if plan.manifest.get("status") != "complete":
        raise ShardedIngestionError(
            "Shard plan is not complete; finish run_sharding before ingesting shards"
        )
    _verify_selected_shards(plan)

    key_id = _resolve_player_key_id(config, require_secret=True)
    collection_id = derive_collection_id(
        parent_archive_sha256=plan.parent_sha256,
        source_month=config.source_month,
        shard_manifest_identity_hash=shard_manifest_identity_hash(plan.manifest),
        games_per_shard=config.games_per_shard,
        player_hmac_key_id=key_id,
    )
    collection_dir, shards_dir, manifest_path = _collection_paths(config, collection_id)
    collection_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    logical_cpu_count = os.cpu_count() or 1
    if workers > logical_cpu_count:
        _log(
            "[sharded-ingest] warning: workers exceeds logical CPU count "
            f"(workers={workers}, logical_cpus={logical_cpu_count})"
        )

    lock_path = collection_dir / _COLLECTION_LOCK_FILENAME
    with _CollectionMutationLock(lock_path):
        existing_entries, existing_started_at, existing_manifest = _load_existing_collection(
            manifest_path
        )
        if existing_manifest is not None and not resume:
            raise ShardedIngestionError(
                "Existing collection manifest found; pass resume=True to continue safely"
            )

        started_at = existing_started_at or datetime.now(tz=UTC).isoformat(timespec="seconds")
        entries: dict[int, CollectionShardEntry] = dict(existing_entries)
        _validate_existing_collection_members(collection_dir, entries)

        selected_shards = _select_pending_shards(
            plan_entries=plan.entries,
            completed_shard_count=len(entries),
            max_new_shards=max_new_shards,
        )
        selected_indices = [entry.shard_index for entry in selected_shards]

        if not selected_shards:
            manifest = existing_manifest
            if manifest is None:
                manifest = _persist_collection(
                    config=config,
                    plan=plan,
                    collection_id=collection_id,
                    manifest_path=manifest_path,
                    entries=entries,
                    key_id=key_id,
                    started_at=started_at,
                )
            validate_collection_reconciliation(manifest)
            return {
                "collection_id": collection_id,
                "collection_dir": collection_dir.as_posix(),
                "collection_manifest": manifest_path.as_posix(),
                "status": manifest["status"],
                "completed_shard_count": manifest["completed_shard_count"],
                "selected_shard_count": manifest["selected_shard_count"],
                "new_shards_processed": 0,
                "accepted_games": manifest["counts"]["accepted_games"],
                "emitted_moves": manifest["counts"]["emitted_moves"],
                "portfolio_10m_satisfied": manifest["portfolio_10m_satisfied"],
                "worker_count": workers,
                "selected_shard_indices": [],
                "wall_clock_duration_seconds": 0.0,
                "sum_worker_active_duration_seconds": 0.0,
                "games_per_wall_clock_second": 0.0,
                "moves_per_wall_clock_second": 0.0,
                "per_worker_peak_rss_bytes": {},
                "max_worker_peak_rss_bytes": 0,
                "peak_process_tree_rss_bytes": 0,
                "speedup_vs_workers_1": None,
                "parallel_efficiency_vs_workers_1": None,
                "failed_shard_indices": [],
            }

        _log(
            "[sharded-ingest] selected_shards="
            f"{selected_indices} workers={workers} max_new_shards={max_new_shards}"
        )

        behavior = test_worker_behavior or WorkerTestBehavior()
        selected_by_index = {entry.shard_index: entry for entry in selected_shards}
        tasks = [
            _build_worker_task(
                config=config,
                plan=plan,
                collection_dir=collection_dir,
                shard=entry,
                behavior=behavior,
            )
            for entry in selected_shards
        ]

        successful_results: list[_WorkerResult] = []
        buffered_success: dict[int, _WorkerResult] = {}
        failed_results: dict[int, str] = {}
        committed_new_indices: list[int] = []
        processed_new = 0
        manifest = existing_manifest

        def commit_contiguous_successes() -> None:
            nonlocal manifest, processed_new
            committed_any = False
            next_index = len(entries)
            while next_index in buffered_success:
                result = buffered_success.pop(next_index)
                shard = selected_by_index[next_index]
                entries[next_index] = _entry_from_worker_result(result=result, shard=shard)
                committed_new_indices.append(next_index)
                processed_new += 1
                committed_any = True
                next_index = len(entries)

            if committed_any:
                manifest = _persist_collection(
                    config=config,
                    plan=plan,
                    collection_id=collection_id,
                    manifest_path=manifest_path,
                    entries=entries,
                    key_id=key_id,
                    started_at=started_at,
                )

        wall_started = time.perf_counter()
        process_tree_sampler = _ProcessTreeRssSampler()
        process_tree_sampler.start()

        try:
            if workers == 1:
                for task in tasks:
                    result = _run_single_shard_task(task)
                    if result.success:
                        successful_results.append(result)
                        buffered_success[result.shard_index] = result
                        _log(
                            "[sharded-ingest] shard complete "
                            f"index={result.shard_index} pid={result.worker_pid}"
                        )
                        commit_contiguous_successes()
                    else:
                        failed_results[result.shard_index] = result.error_message or "unknown"
                        _log(
                            "[sharded-ingest] shard failed "
                            f"index={result.shard_index} pid={result.worker_pid} "
                            f"error={failed_results[result.shard_index]}"
                        )

            else:
                max_workers = min(workers, len(tasks))
                mp_context = mp.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=mp_context,
                ) as executor:
                    future_to_shard: dict[Future[_WorkerResult], int] = {}
                    for task in tasks:
                        future = executor.submit(_run_single_shard_task, task)
                        future_to_shard[future] = task.shard_index

                    pending: set[Future[_WorkerResult]] = set(future_to_shard.keys())
                    while pending:
                        done, pending = wait(
                            pending,
                            timeout=_HEARTBEAT_INTERVAL_SECONDS,
                            return_when=FIRST_COMPLETED,
                        )

                        if not done:
                            completed = len(successful_results) + len(failed_results)
                            running = len(selected_shards) - completed
                            elapsed = max(time.perf_counter() - wall_started, 0.0)
                            _log(
                                "[sharded-ingest] heartbeat "
                                f"elapsed_s={elapsed:.1f} running={running} "
                                f"completed={len(successful_results)} failed={len(failed_results)}"
                            )
                            continue

                        for future in done:
                            shard_index = future_to_shard[future]
                            try:
                                result = future.result()
                            except Exception as exc:
                                failed_results[shard_index] = (
                                    f"{type(exc).__name__}: {exc}"
                                )
                                error_message = failed_results[shard_index]
                                _log(
                                    "[sharded-ingest] shard failed "
                                    f"index={shard_index} pid=unknown "
                                    f"error={error_message}"
                                )
                                continue

                            if result.success:
                                successful_results.append(result)
                                buffered_success[result.shard_index] = result
                                _log(
                                    "[sharded-ingest] shard complete "
                                    f"index={result.shard_index} pid={result.worker_pid}"
                                )
                                commit_contiguous_successes()
                            else:
                                failed_results[result.shard_index] = (
                                    result.error_message or "unknown"
                                )
                                _log(
                                    "[sharded-ingest] shard failed "
                                    f"index={result.shard_index} pid={result.worker_pid} "
                                    f"error={failed_results[result.shard_index]}"
                                )

                        completed = len(successful_results) + len(failed_results)
                        running = len(selected_shards) - completed
                        elapsed = max(time.perf_counter() - wall_started, 0.0)
                        _log(
                            "[sharded-ingest] progress "
                            f"elapsed_s={elapsed:.1f} running={running} "
                            f"completed={len(successful_results)} failed={len(failed_results)}"
                        )
        finally:
            peak_process_tree_rss_bytes = process_tree_sampler.stop()

        wall_clock_duration_seconds = max(time.perf_counter() - wall_started, 0.0)
        commit_contiguous_successes()

        successful_uncommitted = sorted(buffered_success.keys())
        committed_entries = [entries[index] for index in committed_new_indices]
        failed_indices = sorted(failed_results.keys())

        run_metrics = _compute_run_metrics(
            worker_count=workers,
            selected_shard_indices=selected_indices,
            new_shards_processed=processed_new,
            wall_clock_duration_seconds=wall_clock_duration_seconds,
            successful_results=successful_results,
            committed_entries=committed_entries,
            peak_process_tree_rss_bytes=peak_process_tree_rss_bytes,
            failed_shard_indices=failed_indices,
            successful_uncommitted_shard_indices=successful_uncommitted,
            baseline_wall_clock_seconds=baseline_wall_clock_seconds,
        )

        if processed_new > 0:
            manifest = _persist_collection(
                config=config,
                plan=plan,
                collection_id=collection_id,
                manifest_path=manifest_path,
                entries=entries,
                key_id=key_id,
                started_at=started_at,
                run_metrics=run_metrics,
            )
        elif manifest is None:
            manifest = _persist_collection(
                config=config,
                plan=plan,
                collection_id=collection_id,
                manifest_path=manifest_path,
                entries=entries,
                key_id=key_id,
                started_at=started_at,
            )

        if manifest is None:
            raise ShardedIngestionError("Collection manifest could not be resolved")

        validate_collection_reconciliation(manifest)

        if failed_results:
            raise ShardedIngestionError(
                "One or more selected shards failed: "
                + ", ".join(f"{index}:{failed_results[index]}" for index in failed_indices)
            )

        return {
            "collection_id": collection_id,
            "collection_dir": collection_dir.as_posix(),
            "collection_manifest": manifest_path.as_posix(),
            "status": manifest["status"],
            "completed_shard_count": manifest["completed_shard_count"],
            "selected_shard_count": manifest["selected_shard_count"],
            "new_shards_processed": processed_new,
            "accepted_games": manifest["counts"]["accepted_games"],
            "emitted_moves": manifest["counts"]["emitted_moves"],
            "portfolio_10m_satisfied": manifest["portfolio_10m_satisfied"],
            "worker_count": workers,
            "selected_shard_indices": selected_indices,
            "wall_clock_duration_seconds": run_metrics["wall_clock_duration_seconds"],
            "sum_worker_active_duration_seconds": run_metrics[
                "sum_worker_active_duration_seconds"
            ],
            "games_per_wall_clock_second": run_metrics["games_per_wall_clock_second"],
            "moves_per_wall_clock_second": run_metrics["moves_per_wall_clock_second"],
            "per_worker_peak_rss_bytes": run_metrics["per_worker_peak_rss_bytes"],
            "max_worker_peak_rss_bytes": run_metrics["max_worker_peak_rss_bytes"],
            "peak_process_tree_rss_bytes": run_metrics["peak_process_tree_rss_bytes"],
            "speedup_vs_workers_1": run_metrics["speedup_vs_workers_1"],
            "parallel_efficiency_vs_workers_1": run_metrics[
                "parallel_efficiency_vs_workers_1"
            ],
            "failed_shard_indices": failed_indices,
            "successful_uncommitted_shard_indices": successful_uncommitted,
        }


def _status(config_path: Path) -> dict[str, Any]:
    config = load_orchestrator_config(config_path)
    shard_dir = config.shard_root / config.source_month
    shard_manifest_path = shard_dir / SHARD_MANIFEST_FILENAME
    if not shard_manifest_path.exists():
        return {"shard_plan": "not_started"}
    plan_manifest = load_shard_manifest(shard_manifest_path)
    plan_entries = parse_shard_entries(plan_manifest)
    key_id = _resolve_player_key_id(config, require_secret=False)
    collection_id = derive_collection_id(
        parent_archive_sha256=str(plan_manifest.get("parent_archive_sha256", "")),
        source_month=config.source_month,
        shard_manifest_identity_hash=shard_manifest_identity_hash(plan_manifest),
        games_per_shard=config.games_per_shard,
        player_hmac_key_id=key_id,
    )
    _, _, manifest_path = _collection_paths(config, collection_id)
    completed = 0
    if manifest_path.exists():
        completed = len(parse_collection_entries(load_collection_manifest(manifest_path)))
    return {
        "shard_plan_status": plan_manifest.get("status"),
        "plan_shard_count": len(plan_entries),
        "collection_id": collection_id,
        "collection_shards_complete": completed,
        "collection_dir": (config.collection_root / collection_id).as_posix(),
    }


def _verify_only(config_path: Path) -> dict[str, Any]:
    config = load_orchestrator_config(config_path)
    plan = _load_plan(config)
    _verify_selected_shards(plan)
    key_id = _resolve_player_key_id(config, require_secret=False)
    collection_id = derive_collection_id(
        parent_archive_sha256=plan.parent_sha256,
        source_month=config.source_month,
        shard_manifest_identity_hash=shard_manifest_identity_hash(plan.manifest),
        games_per_shard=config.games_per_shard,
        player_hmac_key_id=key_id,
    )
    collection_dir, _, manifest_path = _collection_paths(config, collection_id)
    manifest = load_collection_manifest(manifest_path)
    validate_collection_reconciliation(manifest)
    for entry in parse_collection_entries(manifest):
        validate_published_dataset_root(collection_dir / entry.dataset_relpath)
    return {
        "collection_id": collection_id,
        "status": manifest.get("status"),
        "verified_datasets": manifest.get("completed_shard_count"),
        "reconciliation": "passed",
    }


def _dry_run(config_path: Path) -> dict[str, Any]:
    config = load_orchestrator_config(config_path)
    shard_dir = config.shard_root / config.source_month
    shard_manifest_path = shard_dir / SHARD_MANIFEST_FILENAME
    plan_status = "not_started"
    plan_shards = 0
    if shard_manifest_path.exists():
        plan_manifest = load_shard_manifest(shard_manifest_path)
        plan_status = str(plan_manifest.get("status"))
        plan_shards = len(parse_shard_entries(plan_manifest))
    estimate = estimate_sizes(
        config.expected_raw_games or 0, disk_probe_path=config.collection_root
    )
    return {
        "dry_run": True,
        "source_month": config.source_month,
        "shard_plan_status": plan_status,
        "plan_shard_count": plan_shards,
        "expected_raw_games": config.expected_raw_games,
        "games_per_shard": config.games_per_shard,
        "size_estimates": estimate.as_dict(),
        "next_action": (
            "complete run_sharding first"
            if plan_status != "complete"
            else "run with --max-new-shards N to ingest shards over multiple sessions"
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable sharded ingestion orchestrator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-new-shards", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--baseline-wall-clock-seconds", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config_path = Path(args.config)

    if args.status:
        payload: dict[str, Any] = _status(config_path)
    elif args.verify_only:
        payload = _verify_only(config_path)
    elif args.dry_run:
        payload = _dry_run(config_path)
    else:
        payload = run_sharded_ingestion(
            config_path,
            max_new_shards=args.max_new_shards,
            resume=args.resume,
            workers=args.workers,
            baseline_wall_clock_seconds=args.baseline_wall_clock_seconds,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
