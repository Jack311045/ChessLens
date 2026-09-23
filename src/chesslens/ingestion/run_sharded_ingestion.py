"""Resumable, multi-session sharded ingestion orchestrator (Phase 1.2c).

Reads a completed shard plan and ingests one or more shards per session by reusing
the existing single-dataset ingestion (`run_ingestion`) with a `SourceLineage` that
anchors every game's identity to the original parent archive and its global index.
Each processed shard is published independently and recorded in an atomically-written
collection manifest so the run can stop and resume across sessions.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def _resolve_player_key_id(config: OrchestratorConfig) -> str | None:
    if config.player_hash_mode == "fixture_placeholder":
        return None
    if not config.player_hmac_key_env or not config.player_hmac_key_id_env:
        raise ShardedIngestionError(
            "hmac_sha256 mode requires player_hmac_key_env and player_hmac_key_id_env"
        )
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


@dataclass(frozen=True)
class _Plan:
    shard_dir: Path
    manifest: dict[str, Any]
    entries: list[ShardEntry]
    parent_filename: str
    parent_sha256: str


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


def _existing_entries(manifest_path: Path) -> dict[int, CollectionShardEntry]:
    if not manifest_path.exists():
        return {}
    manifest = load_collection_manifest(manifest_path)
    return {entry.shard_index: entry for entry in parse_collection_entries(manifest)}


def _persist_collection(
    *,
    config: OrchestratorConfig,
    plan: _Plan,
    collection_id: str,
    manifest_path: Path,
    entries: dict[int, CollectionShardEntry],
    key_id: str | None,
    started_at: str,
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
    atomic_write_json(manifest_path, manifest)
    return manifest


def run_sharded_ingestion(
    config_path: Path,
    *,
    max_new_shards: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config = load_orchestrator_config(config_path)
    plan = _load_plan(config)
    if plan.manifest.get("status") != "complete":
        raise ShardedIngestionError(
            "Shard plan is not complete; finish run_sharding before ingesting shards"
        )
    _verify_selected_shards(plan)

    key_id = _resolve_player_key_id(config)
    collection_id = derive_collection_id(
        parent_archive_sha256=plan.parent_sha256,
        source_month=config.source_month,
        shard_manifest_identity_hash=shard_manifest_identity_hash(plan.manifest),
        games_per_shard=config.games_per_shard,
        player_hmac_key_id=key_id,
    )
    collection_dir, shards_dir, manifest_path = _collection_paths(config, collection_id)
    shards_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_entries(manifest_path)
    started_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    if manifest_path.exists():
        started_at = str(
            load_collection_manifest(manifest_path)
            .get("timing", {})
            .get("started_at_utc", started_at)
        )

    entries: dict[int, CollectionShardEntry] = dict(existing)
    processed_new = 0
    for shard in plan.entries:
        if shard.shard_index in entries:
            continue
        if max_new_shards is not None and processed_new >= max_new_shards:
            break

        shard_output_root = shards_dir / f"shard-{shard.shard_index:05d}"
        lineage = SourceLineage(
            identity_archive_name=plan.parent_filename,
            identity_archive_sha256=plan.parent_sha256,
            global_game_index_offset=shard.start_global_index,
            shard_index=shard.shard_index,
            shard_filename=shard.filename,
            shard_sha256=shard.sha256,
        )
        result = run_ingestion(
            config_path=config.ingestion_config_path,
            overrides={
                "input_path": str(plan.shard_dir / shard.filename),
                "output_root": str(shard_output_root),
                "expected_source_sha256": shard.sha256,
                "source_month": config.source_month,
                "max_games": None,
            },
            source_lineage=lineage,
        )
        dataset_relpath = result.dataset_path.relative_to(collection_dir).as_posix()
        entries[shard.shard_index] = CollectionShardEntry(
            shard_index=shard.shard_index,
            dataset_id=result.dataset_id,
            dataset_relpath=dataset_relpath,
            start_global_index=shard.start_global_index,
            end_global_index=shard.end_global_index,
            scanned_games=result.scanned_games,
            accepted_games=result.accepted_games,
            rejected_games=result.rejected_games,
            emitted_moves=result.emitted_moves,
            error_records=result.error_records,
            output_bytes=_read_output_bytes(result.dataset_path),
            processing_active_seconds=result.processing_and_validation_duration_seconds,
            peak_rss_bytes=result.peak_rss_bytes,
        )
        processed_new += 1
        _persist_collection(
            config=config,
            plan=plan,
            collection_id=collection_id,
            manifest_path=manifest_path,
            entries=entries,
            key_id=key_id,
            started_at=started_at,
        )

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
        "new_shards_processed": processed_new,
        "accepted_games": manifest["counts"]["accepted_games"],
        "emitted_moves": manifest["counts"]["emitted_moves"],
        "portfolio_10m_satisfied": manifest["portfolio_10m_satisfied"],
    }


def _status(config_path: Path) -> dict[str, Any]:
    config = load_orchestrator_config(config_path)
    shard_dir = config.shard_root / config.source_month
    shard_manifest_path = shard_dir / SHARD_MANIFEST_FILENAME
    if not shard_manifest_path.exists():
        return {"shard_plan": "not_started"}
    plan_manifest = load_shard_manifest(shard_manifest_path)
    plan_entries = parse_shard_entries(plan_manifest)
    key_id = _resolve_player_key_id(config)
    collection_id = derive_collection_id(
        parent_archive_sha256=str(plan_manifest.get("parent_archive_sha256", "")),
        source_month=config.source_month,
        shard_manifest_identity_hash=shard_manifest_identity_hash(plan_manifest),
        games_per_shard=config.games_per_shard,
        player_hmac_key_id=key_id,
    )
    _, _, manifest_path = _collection_paths(config, collection_id)
    completed = len(_existing_entries(manifest_path))
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
    key_id = _resolve_player_key_id(config)
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
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
