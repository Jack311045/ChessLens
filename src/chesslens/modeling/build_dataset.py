"""Phase 2.1 modeling dataset builder with staged, idempotent publication."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import shutil
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from chesslens.ingestion.date_resolution import CANONICAL_PLAYED_DATE_RESOLVER_VERSION
from chesslens.ingestion.rss import PeakRssSampler
from chesslens.ingestion.shard_manifest import atomic_write_json
from chesslens.modeling.config import ModelingConfig, apply_cli_overrides, load_modeling_config
from chesslens.modeling.contracts import (
    GAME_ASSIGNMENT_ARROW_SCHEMA,
    LABEL_DEFINITION_VERSION,
    NOVEL_POSITION_TEST_SPLIT,
    POLICY_EXAMPLE_ARROW_SCHEMA,
    TEMPORAL_SPLITS,
    default_feature_label_contract,
    validate_feature_label_contract,
)
from chesslens.modeling.labels import (
    mover_and_opponent_ratings,
    policy_target_action_index,
    rating_band,
    time_control_category,
    value_label_from_result,
)
from chesslens.modeling.leakage import (
    compute_novel_position_test_count,
    compute_position_overlap_metrics,
)
from chesslens.modeling.provenance import (
    WarehouseProvenance,
    load_warehouse_provenance,
    warehouse_effective_rate_percent,
)
from chesslens.modeling.sampling import effective_rate_percent, select_by_hash_mod
from chesslens.modeling.splits import assign_temporal_split, parse_played_date
from chesslens.modeling.validation import (
    canonical_json,
    read_json_object,
    sha256_text,
    validate_manifest_identity,
    validate_relative_artifact_paths,
)
from chesslens.modeling.writer import ModelingParquetWriter, PartitionWriteStats
from chesslens.runtime_paths import (
    RuntimePathResolutionError,
    resolve_collection_root_override_or_env,
    resolve_date_enrichment_manifest_override_or_env,
    resolve_duckdb_path_override_or_env,
    resolve_warehouse_provenance_path_override_or_env,
)


@dataclass(frozen=True)
class ModelingBuildResult:
    run_id: str
    modeling_dataset_id: str
    collection_id: str
    dataset_path: Path
    manifest_path: Path
    reused_existing: bool
    selected_games: int
    selected_examples: int
    split_counts_games: dict[str, int]
    split_counts_examples: dict[str, int]
    novel_position_test_rows: int
    duration_seconds: float | None
    peak_rss_bytes: int | None
    dry_run: bool
    validate_only: bool


@dataclass(frozen=True)
class DateEnrichmentInput:
    manifest_path: Path
    date_enrichment_id: str
    manifest_sha256: str
    source_month: str
    parent_archive_sha256: str
    resolver_version: str
    parquet_files: list[str]


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


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


def _sql_file_list(paths: list[str]) -> str:
    escaped = [p.replace("'", "''") for p in paths]
    return "[" + ", ".join(f"'{item}'" for item in escaped) + "]"


_HEX_DESC_TRANSLATION = str.maketrans("0123456789abcdef", "fedcba9876543210")


def _descending_lex_key(token: str) -> str:
    """Return a key where smaller value means lexicographically larger token."""
    return token.lower().translate(_HEX_DESC_TRANSLATION)


def _iter_cursor_rows(
    cursor: duckdb.DuckDBPyConnection,
    *,
    chunk_size: int,
) -> Any:
    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break
        yield from rows


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Manifest field {field_name} must be an object")
    return value


def _as_required_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Manifest field {field_name} must be an integer") from exc


def _as_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_required_int(value, field_name=field_name)


def _as_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Manifest field {field_name} must be numeric") from exc


def _as_required_str(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise RuntimeError(f"Manifest field {field_name} must be non-empty")
    return text


def _int_mapping(value: Any, *, field_name: str) -> dict[str, int]:
    payload = _as_mapping(value, field_name=field_name)
    result: dict[str, int] = {}
    for key, item in payload.items():
        result[str(key)] = _as_required_int(item, field_name=f"{field_name}.{key}")
    return result


def _result_from_manifest(
    *,
    run_id: str,
    modeling_dataset_id: str,
    dataset_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    reused_existing: bool,
    validate_only: bool,
) -> ModelingBuildResult:
    upstream = _as_mapping(manifest.get("upstream"), field_name="upstream")
    collection_id = str(upstream.get("collection_id", "")).strip()
    if not collection_id:
        raise RuntimeError("Manifest field upstream.collection_id must be non-empty")

    counts = _as_mapping(manifest.get("counts"), field_name="counts")
    split_counts_games = _int_mapping(
        counts.get("game_assignments_by_split"),
        field_name="counts.game_assignments_by_split",
    )
    split_counts_examples = _int_mapping(
        counts.get("policy_examples_by_split"),
        field_name="counts.policy_examples_by_split",
    )

    selected_games = _as_required_int(
        counts.get("selected_games"), field_name="counts.selected_games"
    )
    selected_examples = _as_required_int(
        counts.get("selected_policy_examples"),
        field_name="counts.selected_policy_examples",
    )
    novel_position_test_rows = _as_required_int(
        counts.get("novel_position_test_rows"),
        field_name="counts.novel_position_test_rows",
    )

    performance_raw = manifest.get("performance", {})
    if not isinstance(performance_raw, dict):
        raise RuntimeError("Manifest field performance must be an object")
    performance = performance_raw
    duration_seconds = _as_optional_float(
        performance.get("duration_seconds"), field_name="performance.duration_seconds"
    )
    peak_rss_bytes = _as_optional_int(
        performance.get("peak_rss_bytes"), field_name="performance.peak_rss_bytes"
    )

    return ModelingBuildResult(
        run_id=run_id,
        modeling_dataset_id=modeling_dataset_id,
        collection_id=collection_id,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        reused_existing=reused_existing,
        selected_games=selected_games,
        selected_examples=selected_examples,
        split_counts_games=split_counts_games,
        split_counts_examples=split_counts_examples,
        novel_position_test_rows=novel_position_test_rows,
        duration_seconds=duration_seconds,
        peak_rss_bytes=peak_rss_bytes,
        dry_run=False,
        validate_only=validate_only,
    )


def modeling_build_result_to_json(result: ModelingBuildResult) -> dict[str, Any]:
    return {
        "collection_id": result.collection_id,
        "dry_run": result.dry_run,
        "duration_seconds": result.duration_seconds,
        "manifest_path": _display_path(result.manifest_path),
        "modeling_dataset_id": result.modeling_dataset_id,
        "novel_position_test_rows": result.novel_position_test_rows,
        "peak_rss_bytes": result.peak_rss_bytes,
        "reused_existing": result.reused_existing,
        "run_id": result.run_id,
        "selected_examples": result.selected_examples,
        "selected_games": result.selected_games,
        "split_counts_examples": result.split_counts_examples,
        "split_counts_games": result.split_counts_games,
        "validate_only": result.validate_only,
    }


def _load_collection_identity(config: ModelingConfig) -> dict[str, Any]:
    manifest_path = config.input.collection_root / "_collection_manifest.json"
    manifest = read_json_object(manifest_path)

    collection_id = str(manifest.get("collection_id", "")).strip()
    if not collection_id:
        raise RuntimeError(
            f"Collection manifest does not contain collection_id: {manifest_path.as_posix()}"
        )

    if config.input.expected_collection_id and collection_id != config.input.expected_collection_id:
        raise RuntimeError(
            "Configured expected_collection_id does not match collection manifest: "
            f"{config.input.expected_collection_id!r} != {collection_id!r}"
        )

    manifest_hash = sha256_text(canonical_json(manifest))
    return {
        "collection_id": collection_id,
        "collection_manifest_sha256": manifest_hash,
        "manifest": manifest,
    }


def _load_date_enrichment_input(
    *,
    config: ModelingConfig,
    upstream: dict[str, Any],
) -> DateEnrichmentInput | None:
    manifest_path = config.input.date_enrichment_manifest_path
    if manifest_path is None:
        if config.input.require_date_enrichment:
            raise RuntimeError(
                "Date enrichment is required but no manifest path is configured. "
                "Set input.date_enrichment_manifest_path, --date-enrichment-manifest-path, "
                "or CHESSLENS_DATE_ENRICHMENT_MANIFEST_PATH."
            )
        return None

    manifest = read_json_object(manifest_path)
    status = str(manifest.get("status", "")).strip()
    if status != "complete":
        raise RuntimeError(
            "Date enrichment manifest must be complete before modeling: "
            f"{manifest_path.as_posix()}"
        )

    date_enrichment_id = _as_required_str(
        manifest.get("date_enrichment_id"),
        field_name="date_enrichment_id",
    )
    source_month = _as_required_str(manifest.get("source_month"), field_name="source_month")
    parent_archive_sha256 = _as_required_str(
        manifest.get("parent_archive_sha256"),
        field_name="parent_archive_sha256",
    )
    resolver_version = _as_required_str(
        manifest.get("resolver_version"),
        field_name="resolver_version",
    )

    if resolver_version != CANONICAL_PLAYED_DATE_RESOLVER_VERSION:
        raise RuntimeError(
            "Unsupported date enrichment resolver version: "
            f"{resolver_version!r}. Supported value is "
            f"{CANONICAL_PLAYED_DATE_RESOLVER_VERSION!r}."
        )

    upstream_manifest = _as_mapping(upstream.get("manifest"), field_name="collection_manifest")
    upstream_source_month = str(upstream_manifest.get("source_month", "")).strip()
    if upstream_source_month and upstream_source_month != source_month:
        raise RuntimeError(
            "Date enrichment source_month does not match collection manifest: "
            f"{source_month!r} != {upstream_source_month!r}"
        )

    upstream_parent_sha = str(upstream_manifest.get("parent_archive_sha256", "")).strip()
    if upstream_parent_sha and upstream_parent_sha != parent_archive_sha256:
        raise RuntimeError(
            "Date enrichment parent_archive_sha256 does not match collection manifest: "
            f"{parent_archive_sha256!r} != {upstream_parent_sha!r}"
        )

    shards_raw = manifest.get("shards", [])
    if not isinstance(shards_raw, list):
        raise RuntimeError("Date enrichment manifest field shards must be a list")
    if not shards_raw:
        raise RuntimeError("Date enrichment manifest has no shard artifacts")

    parquet_files: list[str] = []
    seen_relpaths: set[str] = set()
    for index, shard_payload in enumerate(shards_raw):
        shard = _as_mapping(shard_payload, field_name=f"shards[{index}]")
        relpath = _as_required_str(shard.get("output_relpath"), field_name="output_relpath")
        rel = Path(relpath)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(
                "Date enrichment manifest contains unsafe output_relpath: "
                f"{relpath!r}"
            )
        if relpath in seen_relpaths:
            raise RuntimeError(
                "Date enrichment manifest has duplicate output_relpath entries: "
                f"{relpath!r}"
            )
        seen_relpaths.add(relpath)

        full_path = (manifest_path.parent / rel).resolve()
        if not full_path.exists():
            raise RuntimeError(
                "Date enrichment artifact file does not exist: "
                f"{full_path.as_posix()}"
            )
        parquet_files.append(full_path.as_posix())

    manifest_sha = sha256_text(canonical_json(manifest))
    return DateEnrichmentInput(
        manifest_path=manifest_path,
        date_enrichment_id=date_enrichment_id,
        manifest_sha256=manifest_sha,
        source_month=source_month,
        parent_archive_sha256=parent_archive_sha256,
        resolver_version=resolver_version,
        parquet_files=parquet_files,
    )


def _validate_date_enrichment_against_games_relation(
    *,
    connection: duckdb.DuckDBPyConnection,
    config: ModelingConfig,
    date_enrichment: DateEnrichmentInput,
) -> dict[str, int]:
    enrichment_sql = f"SELECT * FROM read_parquet({_sql_file_list(date_enrichment.parquet_files)})"

    duplicate_game_ids_row = connection.execute(
        "SELECT COUNT(*) FROM ("
        f"  SELECT game_id FROM ({enrichment_sql}) GROUP BY game_id HAVING COUNT(*) > 1"
        ")"
    ).fetchone()
    duplicate_game_ids = int(duplicate_game_ids_row[0]) if duplicate_game_ids_row else 0
    if duplicate_game_ids > 0:
        raise RuntimeError(
            "Date enrichment relation has duplicate game_id rows; expected one row per game"
        )

    coverage_row = connection.execute(
        "SELECT "
        "  SUM(CASE WHEN e.game_id IS NULL THEN 1 ELSE 0 END) AS missing_games, "
        "  SUM(CASE WHEN e.canonical_date_source = 'missing_or_invalid' THEN 1 ELSE 0 END) "
        "    AS missing_or_invalid_games "
        f"FROM {config.input.games_relation} g "
        f"LEFT JOIN ({enrichment_sql}) e ON g.game_id = e.game_id"
    ).fetchone()
    if coverage_row is None:
        raise RuntimeError("Unable to validate date enrichment coverage against games relation")

    missing_games = int(coverage_row[0] or 0)
    missing_or_invalid_games = int(coverage_row[1] or 0)
    if missing_games > 0:
        raise RuntimeError(
            "Date enrichment does not fully cover the modeling games relation. "
            f"Missing {missing_games} game_id values from enrichment sidecar."
        )

    return {
        "missing_games": missing_games,
        "missing_or_invalid_games": missing_or_invalid_games,
    }


def _dataset_identity_payload(
    config: ModelingConfig,
    upstream: dict[str, Any],
    warehouse_provenance: WarehouseProvenance,
    date_enrichment: DateEnrichmentInput | None,
) -> dict[str, Any]:
    return {
        "upstream": {
            "collection_id": upstream["collection_id"],
            "collection_manifest_sha256": upstream["collection_manifest_sha256"],
            "warehouse_provenance_version": warehouse_provenance.provenance_version,
            "warehouse_provenance_sha256": warehouse_provenance.warehouse_provenance_sha256,
            "warehouse_kind": warehouse_provenance.warehouse_kind,
        },
        "versions": {
            "modeling_pipeline_version": config.versions.modeling_pipeline_version,
            "split_definition_version": config.versions.split_definition_version,
            "feature_schema_version": config.versions.feature_schema_version,
            "label_definition_version": config.versions.label_definition_version,
            "schema_version": config.versions.schema_version,
            "position_normalization_version": config.versions.position_normalization_version,
            "board_encoding_version": config.versions.board_encoding_version,
            "action_encoding_version": config.versions.action_encoding_version,
        },
        "sampling": {
            "rule_version": config.sampling.rule_version,
            "seed": config.sampling.seed,
            "hash_modulus": config.sampling.hash_modulus,
            "hash_threshold": config.sampling.hash_threshold,
            "max_games": config.sampling.max_games,
        },
        "splits": {
            "train": {
                "start_date": config.splits.train.start_date.isoformat(),
                "end_date": config.splits.train.end_date.isoformat(),
            },
            "validation": {
                "start_date": config.splits.validation.start_date.isoformat(),
                "end_date": config.splits.validation.end_date.isoformat(),
            },
            "test": {
                "start_date": config.splits.test.start_date.isoformat(),
                "end_date": config.splits.test.end_date.isoformat(),
            },
            "missing_or_invalid_date_policy": config.splits.missing_or_invalid_date_policy,
        },
        "player_holdout": {
            "rule_version": config.player_holdout.rule_version,
            "seed": config.player_holdout.seed,
            "hash_modulus": config.player_holdout.hash_modulus,
            "hash_threshold": config.player_holdout.hash_threshold,
            "missing_player_hash_policy": config.player_holdout.missing_player_hash_policy,
        },
        "output_affecting": {
            "max_examples": config.output.max_examples,
            "batch_rows": config.output.batch_rows,
            "parquet_compression": config.output.parquet_compression,
            "parquet_row_group_size": config.output.parquet_row_group_size,
            "move_context_relation": config.input.move_context_relation,
            "games_relation": config.input.games_relation,
            "date_enrichment_required": config.input.require_date_enrichment,
            "date_enrichment_id": (
                None if date_enrichment is None else date_enrichment.date_enrichment_id
            ),
            "date_enrichment_manifest_sha256": (
                None if date_enrichment is None else date_enrichment.manifest_sha256
            ),
            "date_enrichment_resolver_version": (
                None if date_enrichment is None else date_enrichment.resolver_version
            ),
            "strict": config.behavior.strict,
        },
    }


def _derive_modeling_dataset_id(
    config: ModelingConfig,
    upstream: dict[str, Any],
    warehouse_provenance: WarehouseProvenance,
    date_enrichment: DateEnrichmentInput | None,
) -> str:
    payload = _dataset_identity_payload(
        config,
        upstream,
        warehouse_provenance,
        date_enrichment,
    )
    return sha256_text(canonical_json(payload))


def _collection_counts_from_manifest(upstream: dict[str, Any]) -> tuple[int, int]:
    manifest = _as_mapping(upstream.get("manifest"), field_name="collection_manifest")
    counts = _as_mapping(manifest.get("counts"), field_name="collection_manifest.counts")
    return (
        _as_required_int(
            counts.get("accepted_games"),
            field_name="collection_manifest.counts.accepted_games",
        ),
        _as_required_int(
            counts.get("emitted_moves"),
            field_name="collection_manifest.counts.emitted_moves",
        ),
    )


def _relation_row_count(connection: duckdb.DuckDBPyConnection, relation: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()
    if row is None:
        raise RuntimeError(f"Unable to count rows for relation {relation!r}")
    return int(row[0])


def _validate_warehouse_provenance_contract(
    *,
    config: ModelingConfig,
    upstream: dict[str, Any],
    warehouse_provenance: WarehouseProvenance,
) -> None:
    collection_id = str(upstream["collection_id"])
    if warehouse_provenance.collection_id != collection_id:
        raise RuntimeError(
            "Warehouse provenance collection_id does not match collection manifest: "
            f"{warehouse_provenance.collection_id!r} != {collection_id!r}"
        )

    if warehouse_provenance.games_relation != config.input.games_relation:
        raise RuntimeError(
            "Warehouse provenance games_relation does not match modeling config "
            "input.games_relation: "
            f"{warehouse_provenance.games_relation!r} != {config.input.games_relation!r}"
        )

    if warehouse_provenance.move_context_relation != config.input.move_context_relation:
        raise RuntimeError(
            "Warehouse provenance move_context_relation does not match modeling config "
            "input.move_context_relation: "
            f"{warehouse_provenance.move_context_relation!r} != "
            f"{config.input.move_context_relation!r}"
        )


def _validate_warehouse_provenance_against_duckdb(
    *,
    connection: duckdb.DuckDBPyConnection,
    config: ModelingConfig,
    upstream: dict[str, Any],
    warehouse_provenance: WarehouseProvenance,
) -> dict[str, int]:
    actual_games = _relation_row_count(connection, config.input.games_relation)
    actual_moves = _relation_row_count(connection, config.input.move_context_relation)

    if warehouse_provenance.snapshot_counts.games != actual_games:
        raise RuntimeError(
            "Warehouse provenance snapshot game count mismatch for relation "
            f"{config.input.games_relation!r}: declared "
            f"{warehouse_provenance.snapshot_counts.games}, actual {actual_games}"
        )

    if warehouse_provenance.snapshot_counts.moves != actual_moves:
        raise RuntimeError(
            "Warehouse provenance snapshot move count mismatch for relation "
            f"{config.input.move_context_relation!r}: declared "
            f"{warehouse_provenance.snapshot_counts.moves}, actual {actual_moves}"
        )

    collection_games, collection_moves = _collection_counts_from_manifest(upstream)

    if warehouse_provenance.warehouse_kind == "full":
        if actual_games != collection_games or actual_moves != collection_moves:
            raise RuntimeError(
                "Warehouse provenance declares warehouse_kind='full' but DuckDB relation counts "
                "do not match the parent collection manifest counts. This warehouse appears "
                "sampled "
                "or filtered and must be declared as deterministic_sample or fixture."
            )

    if warehouse_provenance.warehouse_kind == "deterministic_sample":
        sampling = warehouse_provenance.sampling
        if sampling is None:
            raise RuntimeError(
                "deterministic_sample warehouse provenance is missing sampling metadata"
            )

        if sampling.parent_collection_id != str(upstream["collection_id"]):
            raise RuntimeError(
                "Warehouse provenance sampling.parent_collection_id does not match "
                "collection manifest collection_id: "
                f"{sampling.parent_collection_id!r} != {upstream['collection_id']!r}"
            )

        if sampling.parent_full_counts.games != collection_games:
            raise RuntimeError(
                "Warehouse provenance sampling.parent_full_counts.games does not match "
                "collection manifest accepted_games: "
                f"{sampling.parent_full_counts.games} != {collection_games}"
            )

        if sampling.parent_full_counts.moves != collection_moves:
            raise RuntimeError(
                "Warehouse provenance sampling.parent_full_counts.moves does not match "
                "collection manifest emitted_moves: "
                f"{sampling.parent_full_counts.moves} != {collection_moves}"
            )

        if actual_games > collection_games or actual_moves > collection_moves:
            raise RuntimeError(
                "Warehouse provenance declares deterministic_sample but sampled relation counts "
                "exceed parent collection counts"
            )

    return {
        "actual_games": actual_games,
        "actual_moves": actual_moves,
        "collection_games": collection_games,
        "collection_moves": collection_moves,
    }


def _open_validated_duckdb_connection(
    *,
    config: ModelingConfig,
    upstream: dict[str, Any],
    warehouse_provenance: WarehouseProvenance,
) -> duckdb.DuckDBPyConnection:
    if not config.input.duckdb_path.exists():
        raise FileNotFoundError(
            "Configured duckdb_path does not exist. Run preflight and dbt build first: "
            f"{config.input.duckdb_path.as_posix()}"
        )

    connection = duckdb.connect(str(config.input.duckdb_path), read_only=True)
    try:
        _ensure_relation_exists(connection, config.input.games_relation)
        _ensure_relation_exists(connection, config.input.move_context_relation)
        _validate_warehouse_provenance_against_duckdb(
            connection=connection,
            config=config,
            upstream=upstream,
            warehouse_provenance=warehouse_provenance,
        )
    except Exception:
        connection.close()
        raise
    return connection


def _sampling_stage_summary(
    *,
    config: ModelingConfig,
    warehouse_provenance: WarehouseProvenance,
) -> dict[str, Any]:
    warehouse_requested_rate = (
        warehouse_provenance.sampling.requested_rate_percent
        if warehouse_provenance.sampling is not None
        else (100.0 if warehouse_provenance.warehouse_kind == "full" else None)
    )
    warehouse_effective_rate = warehouse_effective_rate_percent(warehouse_provenance)

    modeling_effective_rate = effective_rate_percent(
        hash_modulus=config.sampling.hash_modulus,
        hash_threshold=config.sampling.hash_threshold,
    )
    cumulative_effective_rate = None
    if warehouse_effective_rate is not None:
        cumulative_effective_rate = (warehouse_effective_rate * modeling_effective_rate) / 100.0

    return {
        "warehouse": {
            "warehouse_kind": warehouse_provenance.warehouse_kind,
            "requested_rate_percent_of_full_collection": warehouse_requested_rate,
            "effective_rate_percent_of_full_collection": warehouse_effective_rate,
            "sampling": warehouse_provenance.payload.get("sampling"),
        },
        "modeling": {
            "requested_rate_percent_of_upstream_warehouse": config.sampling.requested_rate_percent,
            "effective_rate_percent_of_upstream_warehouse": modeling_effective_rate,
        },
        "cumulative_effective_rate_percent_of_full_collection": cumulative_effective_rate,
    }


def _validate_existing_dataset(
    *,
    dataset_path: Path,
    expected_dataset_id: str,
    expected_identity_payload_hash: str,
) -> dict[str, Any]:
    manifest_path = dataset_path / "_manifest.json"
    manifest = read_json_object(manifest_path)

    validate_manifest_identity(
        manifest=manifest,
        expected_paths={
            ("status",): "complete",
            ("modeling_dataset_id",): expected_dataset_id,
            ("identity_payload_sha256",): expected_identity_payload_hash,
        },
    )

    relative_paths: list[str] = []
    datasets = manifest.get("datasets", {})
    if isinstance(datasets, dict):
        for partitions in datasets.values():
            if not isinstance(partitions, dict):
                continue
            for stats in partitions.values():
                if not isinstance(stats, dict):
                    continue
                files = stats.get("relative_files", [])
                if isinstance(files, list):
                    relative_paths.extend(str(item) for item in files)

    leakage_files = manifest.get("leakage_audit_files", [])
    if isinstance(leakage_files, list):
        relative_paths.extend(str(item) for item in leakage_files)

    validate_relative_artifact_paths(dataset_path, relative_paths)

    success_marker = dataset_path / "_SUCCESS"
    if not success_marker.exists():
        raise RuntimeError(f"Published modeling dataset is missing _SUCCESS: {success_marker}")

    return manifest


def _ensure_relation_exists(connection: duckdb.DuckDBPyConnection, relation: str) -> None:
    try:
        connection.execute(f"SELECT * FROM {relation} LIMIT 0")
    except duckdb.Error as exc:
        raise RuntimeError(f"Unable to read relation {relation!r}: {exc}") from exc


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _is_holdout_player(config: ModelingConfig, player_hash: str) -> bool:
    selected, _ = select_by_hash_mod(
        namespace="player_holdout",
        token=player_hash,
        seed=config.player_holdout.seed,
        hash_modulus=config.player_holdout.hash_modulus,
        hash_threshold=config.player_holdout.hash_threshold,
    )
    return selected


def _build_game_assignments(
    *,
    connection: duckdb.DuckDBPyConnection,
    config: ModelingConfig,
    writer: ModelingParquetWriter,
    date_enrichment: DateEnrichmentInput | None,
) -> tuple[dict[str, Any], list[str], Counter[str], Counter[str], Counter[str]]:
    query: str
    if date_enrichment is None:
        query = (
            "SELECT "
            "  g.game_id, g.source_month, g.played_date, g.played_date, "
            "  'legacy_played_date' AS played_date_source, "
            "  g.result, g.white_player_hash, g.black_player_hash, "
            "  g.white_rating, g.black_rating, g.time_control_raw, g.eco, g.opening, g.ply_count "
            f"FROM {config.input.games_relation} g "
            "ORDER BY g.game_id"
        )
    else:
        enrichment_sql = (
            "SELECT * FROM read_parquet("
            f"{_sql_file_list(date_enrichment.parquet_files)}"
            ")"
        )
        query = (
            "SELECT "
            "  g.game_id, g.source_month, g.played_date, e.canonical_played_date_iso, "
            "  COALESCE(e.canonical_date_source, 'missing_or_invalid') AS played_date_source, "
            "  g.result, g.white_player_hash, g.black_player_hash, "
            "  g.white_rating, g.black_rating, g.time_control_raw, g.eco, g.opening, g.ply_count "
            f"FROM {config.input.games_relation} g "
            f"LEFT JOIN ({enrichment_sql}) e ON g.game_id = e.game_id "
            "ORDER BY g.game_id"
        )

    game_rejections: Counter[str] = Counter()
    sampling_stats: Counter[str] = Counter()
    date_source_counts: Counter[str] = Counter()
    assignments_buffer: list[dict[str, Any]] = []

    max_games = config.sampling.max_games
    selected_candidates: list[dict[str, Any]] = []

    if max_games is None:
        cursor = connection.execute(query)
        for row in _iter_cursor_rows(cursor, chunk_size=config.output.batch_rows):
            sampling_stats["games_scanned"] += 1
            game_id = str(row[0])
            selected, sample_score = select_by_hash_mod(
                namespace="game_sampling",
                token=game_id,
                seed=config.sampling.seed,
                hash_modulus=config.sampling.hash_modulus,
                hash_threshold=config.sampling.hash_threshold,
            )
            if not selected:
                continue

            sampling_stats["games_selected_by_rule"] += 1
            candidate = {
                "game_id": game_id,
                "source_month": row[1],
                "played_date_raw": row[2],
                "played_date_for_split": row[3],
                "played_date_source": row[4],
                "result": row[5],
                "white_player_hash": row[6],
                "black_player_hash": row[7],
                "white_rating": row[8],
                "black_rating": row[9],
                "time_control_raw": row[10],
                "eco": row[11],
                "opening": row[12],
                "ply_count": row[13],
                "sample_score_u64": sample_score,
            }
            _append_assignment_candidate(
                candidate=candidate,
                config=config,
                out_buffer=assignments_buffer,
                game_rejections=game_rejections,
                sampling_stats=sampling_stats,
                date_source_counts=date_source_counts,
            )
            if len(assignments_buffer) >= config.output.batch_rows:
                writer.write_rows(
                    dataset_name="game_assignments",
                    rows=assignments_buffer,
                    schema=GAME_ASSIGNMENT_ARROW_SCHEMA,
                    partition_column="temporal_split",
                )
                assignments_buffer.clear()
    else:
        # Deterministic top-k by sample score using bounded memory.
        heap: list[tuple[int, str, dict[str, Any]]] = []
        if max_games > 0:
            cursor = connection.execute(query)
            for row in _iter_cursor_rows(cursor, chunk_size=config.output.batch_rows):
                sampling_stats["games_scanned"] += 1
                game_id = str(row[0])
                selected, sample_score = select_by_hash_mod(
                    namespace="game_sampling",
                    token=game_id,
                    seed=config.sampling.seed,
                    hash_modulus=config.sampling.hash_modulus,
                    hash_threshold=config.sampling.hash_threshold,
                )
                if not selected:
                    continue

                sampling_stats["games_selected_by_rule"] += 1
                candidate = {
                    "game_id": game_id,
                    "source_month": row[1],
                    "played_date_raw": row[2],
                    "played_date_for_split": row[3],
                    "played_date_source": row[4],
                    "result": row[5],
                    "white_player_hash": row[6],
                    "black_player_hash": row[7],
                    "white_rating": row[8],
                    "black_rating": row[9],
                    "time_control_raw": row[10],
                    "eco": row[11],
                    "opening": row[12],
                    "ply_count": row[13],
                    "sample_score_u64": sample_score,
                }

                # The heap root is the current worst retained candidate.
                item = (-sample_score, _descending_lex_key(game_id), candidate)
                if len(heap) < max_games:
                    heapq.heappush(heap, item)
                    continue

                if item > heap[0]:
                    heapq.heapreplace(heap, item)

        selected_candidates = [entry[2] for entry in heap]
        selected_candidates.sort(
            key=lambda row: (int(row["sample_score_u64"]), str(row["game_id"]))
        )
        sampling_stats["games_selected_after_cap"] = len(selected_candidates)

        for candidate in selected_candidates:
            _append_assignment_candidate(
                candidate=candidate,
                config=config,
                out_buffer=assignments_buffer,
                game_rejections=game_rejections,
                sampling_stats=sampling_stats,
                date_source_counts=date_source_counts,
            )
            if len(assignments_buffer) >= config.output.batch_rows:
                writer.write_rows(
                    dataset_name="game_assignments",
                    rows=assignments_buffer,
                    schema=GAME_ASSIGNMENT_ARROW_SCHEMA,
                    partition_column="temporal_split",
                )
                assignments_buffer.clear()

    if assignments_buffer:
        writer.write_rows(
            dataset_name="game_assignments",
            rows=assignments_buffer,
            schema=GAME_ASSIGNMENT_ARROW_SCHEMA,
            partition_column="temporal_split",
        )
        assignments_buffer.clear()

    writer_stats = writer.stats().get("game_assignments", {})
    assignment_files: list[str] = []
    for stats in writer_stats.values():
        assignment_files.extend(stats.relative_files)

    if not assignment_files:
        raise RuntimeError(
            "No game assignments were written; check sampling and split configuration"
        )

    assignment_files_for_sql = [
        (writer.dataset_root / rel_path).resolve().as_posix()
        for rel_path in assignment_files
    ]

    assignment_counts = _counts_by_split_from_assignment_stats(writer_stats)
    return (
        assignment_counts,
        assignment_files_for_sql,
        game_rejections,
        sampling_stats,
        date_source_counts,
    )


def _counts_by_split_from_assignment_stats(
    stats_by_partition: dict[str, PartitionWriteStats],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split, stats in stats_by_partition.items():
        counts[split] = stats.row_count
    return counts


def _append_assignment_candidate(
    *,
    candidate: dict[str, Any],
    config: ModelingConfig,
    out_buffer: list[dict[str, Any]],
    game_rejections: Counter[str],
    sampling_stats: Counter[str],
    date_source_counts: Counter[str],
) -> None:
    parsed = parse_played_date(_as_optional_str(candidate["played_date_for_split"]))
    date_source = _as_optional_str(candidate.get("played_date_source")) or "missing_or_invalid"
    date_source_counts[date_source] += 1

    split, split_reason = assign_temporal_split(
        parsed_date=parsed.parsed_date,
        date_status=parsed.status,
        train=config.splits.train,
        validation=config.splits.validation,
        test=config.splits.test,
        missing_or_invalid_date_policy=config.splits.missing_or_invalid_date_policy,
    )
    if split is None:
        game_rejections[split_reason] += 1
        return

    white_hash = _as_optional_str(candidate["white_player_hash"])
    black_hash = _as_optional_str(candidate["black_player_hash"])

    is_holdout = False
    if white_hash is not None and _is_holdout_player(config, white_hash):
        is_holdout = True
    if black_hash is not None and _is_holdout_player(config, black_hash):
        is_holdout = True

    missing_player_hash = white_hash is None or black_hash is None
    eligible_for_player_disjoint_training = (
        split == "train"
        and not is_holdout
        and (
            config.player_holdout.missing_player_hash_policy
            == "include_in_player_disjoint_training"
            or not missing_player_hash
        )
    )

    out_buffer.append(
        {
            "game_id": str(candidate["game_id"]),
            "source_month": _as_optional_str(candidate["source_month"]),
            "played_date_raw": _as_optional_str(candidate["played_date_raw"]),
            "played_date_iso": parsed.played_date_iso,
            "played_date_source": date_source,
            "temporal_split": split,
            "temporal_split_reason": split_reason,
            "sample_score_u64": int(candidate["sample_score_u64"]),
            "white_player_hash": white_hash,
            "black_player_hash": black_hash,
            "is_player_holdout_game": is_holdout,
            "player_disjoint_training_eligible": eligible_for_player_disjoint_training,
            "result": _as_optional_str(candidate["result"]),
            "white_rating": _to_int_or_none(candidate["white_rating"]),
            "black_rating": _to_int_or_none(candidate["black_rating"]),
            "time_control_raw": _as_optional_str(candidate["time_control_raw"]),
            "eco": _as_optional_str(candidate["eco"]),
            "opening": _as_optional_str(candidate["opening"]),
            "ply_count": _to_int_or_none(candidate["ply_count"]),
        }
    )
    sampling_stats["games_selected_after_split_policy"] += 1


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _build_policy_examples(
    *,
    connection: duckdb.DuckDBPyConnection,
    config: ModelingConfig,
    writer: ModelingParquetWriter,
    assignment_files: list[str],
) -> tuple[dict[str, int], list[str], Counter[str]]:
    assignments_sql = f"SELECT * FROM read_parquet({_sql_file_list(assignment_files)})"

    query = (
        "SELECT "
        "  m.game_id, m.ply, m.source_month, m.position_id, m.pre_move_fen, "
        "  m.normalized_pre_move_fen, m.side_to_move, m.played_move_uci, "
        "  m.white_rating, m.black_rating, m.time_control_raw, m.eco, m.opening, "
        "  m.result, m.termination, "
        "  a.temporal_split, a.player_disjoint_training_eligible, a.is_player_holdout_game, "
        "  a.played_date_iso, a.played_date_source, a.white_player_hash, a.black_player_hash "
        f"FROM {config.input.move_context_relation} m "
        f"INNER JOIN ({assignments_sql}) a ON m.game_id = a.game_id "
        "ORDER BY m.game_id, m.ply"
    )

    cursor = connection.execute(query)
    policy_buffer: list[dict[str, Any]] = []
    policy_rejections: Counter[str] = Counter()
    examples_written = 0

    for row in _iter_cursor_rows(cursor, chunk_size=config.output.batch_rows):
        if (
            config.output.max_examples is not None
            and examples_written >= config.output.max_examples
        ):
            break

        try:
            action_index = policy_target_action_index(
                pre_move_fen=str(row[4]),
                played_move_uci=str(row[7]),
            )
            value_label = value_label_from_result(
                side_to_move=str(row[6]), result=_as_optional_str(row[13])
            )
            mover_rating, opponent_rating, rating_diff = mover_and_opponent_ratings(
                side_to_move=str(row[6]),
                white_rating=_to_int_or_none(row[8]),
                black_rating=_to_int_or_none(row[9]),
            )
        except ValueError as exc:
            policy_rejections[type(exc).__name__] += 1
            if config.behavior.strict:
                raise RuntimeError(f"Policy label validation failed: {exc}") from exc
            continue

        policy_buffer.append(
            {
                "game_id": str(row[0]),
                "ply": int(row[1]),
                "temporal_split": str(row[15]),
                "source_month": _as_optional_str(row[2]),
                "played_date_iso": _as_optional_str(row[18]),
                "played_date_source": str(row[19]),
                "position_id": str(row[3]),
                "pre_move_fen": str(row[4]),
                "normalized_pre_move_fen": str(row[5]),
                "side_to_move": str(row[6]),
                "time_control_raw": _as_optional_str(row[10]),
                "time_control_category": time_control_category(_as_optional_str(row[10])),
                "eco": _as_optional_str(row[11]),
                "opening": _as_optional_str(row[12]),
                "white_player_hash": _as_optional_str(row[20]),
                "black_player_hash": _as_optional_str(row[21]),
                "mover_rating": mover_rating,
                "opponent_rating": opponent_rating,
                "rating_difference": rating_diff,
                "mover_rating_band": rating_band(mover_rating),
                "player_disjoint_training_eligible": bool(row[16]),
                "is_player_holdout_game": bool(row[17]),
                "played_move_uci_target": str(row[7]),
                "policy_target_action_index": int(action_index),
                "value_target_wdl": value_label,
                "final_result_target": _as_optional_str(row[13]),
                "termination_target": _as_optional_str(row[14]),
                "schema_version": config.versions.schema_version,
                "position_normalization_version": config.versions.position_normalization_version,
                "board_encoding_version": config.versions.board_encoding_version,
                "action_encoding_version": config.versions.action_encoding_version,
                "feature_schema_version": config.versions.feature_schema_version,
                "label_definition_version": config.versions.label_definition_version,
                "value_label_version": LABEL_DEFINITION_VERSION,
            }
        )
        examples_written += 1

        if len(policy_buffer) >= config.output.batch_rows:
            writer.write_rows(
                dataset_name="policy_examples",
                rows=policy_buffer,
                schema=POLICY_EXAMPLE_ARROW_SCHEMA,
                partition_column="temporal_split",
            )
            policy_buffer.clear()

    if policy_buffer:
        writer.write_rows(
            dataset_name="policy_examples",
            rows=policy_buffer,
            schema=POLICY_EXAMPLE_ARROW_SCHEMA,
            partition_column="temporal_split",
        )
        policy_buffer.clear()

    stats_by_partition = writer.stats().get("policy_examples", {})
    primary_policy_files: list[str] = []
    move_counts_by_split: dict[str, int] = {}
    for partition_value, stats in stats_by_partition.items():
        if partition_value == NOVEL_POSITION_TEST_SPLIT:
            continue
        if partition_value not in TEMPORAL_SPLITS:
            continue
        primary_policy_files.extend(stats.relative_files)
        move_counts_by_split[partition_value] = stats.row_count

    if not primary_policy_files:
        raise RuntimeError("No policy example rows were written for temporal train/validation/test")

    primary_policy_files_for_sql = [
        (writer.dataset_root / rel_path).resolve().as_posix()
        for rel_path in primary_policy_files
    ]

    return move_counts_by_split, primary_policy_files_for_sql, policy_rejections


def _write_novel_position_slice(
    *,
    connection: duckdb.DuckDBPyConnection,
    writer: ModelingParquetWriter,
    primary_policy_files: list[str],
    batch_rows: int,
) -> int:
    primary_sql = f"SELECT * FROM read_parquet({_sql_file_list(primary_policy_files)})"

    select_columns = [
        "game_id",
        "ply",
        "temporal_split",
        "source_month",
        "played_date_iso",
        "played_date_source",
        "position_id",
        "pre_move_fen",
        "normalized_pre_move_fen",
        "side_to_move",
        "time_control_raw",
        "time_control_category",
        "eco",
        "opening",
        "white_player_hash",
        "black_player_hash",
        "mover_rating",
        "opponent_rating",
        "rating_difference",
        "mover_rating_band",
        "player_disjoint_training_eligible",
        "is_player_holdout_game",
        "played_move_uci_target",
        "policy_target_action_index",
        "value_target_wdl",
        "final_result_target",
        "termination_target",
        "schema_version",
        "position_normalization_version",
        "board_encoding_version",
        "action_encoding_version",
        "feature_schema_version",
        "label_definition_version",
        "value_label_version",
    ]

    query = (
        "WITH base AS ("
        f"  SELECT * FROM ({primary_sql})"
        "), train_positions AS ("
        "  SELECT DISTINCT position_id FROM base WHERE temporal_split = 'train'"
        ") "
        "SELECT "
        + ", ".join(select_columns)
        + " "
        "FROM base "
        "WHERE temporal_split = 'test' "
        "AND position_id NOT IN (SELECT position_id FROM train_positions) "
        "ORDER BY game_id, ply"
    )

    cursor = connection.execute(query)
    buffer: list[dict[str, Any]] = []
    count = 0

    for row in _iter_cursor_rows(cursor, chunk_size=batch_rows):
        record = {select_columns[i]: row[i] for i in range(len(select_columns))}
        record["temporal_split"] = NOVEL_POSITION_TEST_SPLIT
        buffer.append(record)
        count += 1
        if len(buffer) >= batch_rows:
            writer.write_rows(
                dataset_name="policy_examples",
                rows=buffer,
                schema=POLICY_EXAMPLE_ARROW_SCHEMA,
                partition_column="temporal_split",
            )
            buffer.clear()

    if buffer:
        writer.write_rows(
            dataset_name="policy_examples",
            rows=buffer,
            schema=POLICY_EXAMPLE_ARROW_SCHEMA,
            partition_column="temporal_split",
        )
        buffer.clear()

    return count


def _validate_split_and_holdout_constraints(
    *,
    connection: duckdb.DuckDBPyConnection,
    assignments_sql: str,
    primary_policy_sql: str,
    config: ModelingConfig,
) -> None:
    duplicate_games = connection.execute(
        "SELECT COUNT(*) FROM ("
        f"  SELECT game_id FROM ({assignments_sql}) GROUP BY game_id HAVING COUNT(*) > 1"
        ")"
    ).fetchone()
    if duplicate_games is not None and int(duplicate_games[0]) > 0:
        raise RuntimeError("Validation failed: game assignments are not unique per game_id")

    bad_move_splits = connection.execute(
        "SELECT COUNT(*) FROM ("
        f"  SELECT game_id FROM ({primary_policy_sql}) "
        "  GROUP BY game_id HAVING COUNT(DISTINCT temporal_split) > 1"
        ")"
    ).fetchone()
    if bad_move_splits is not None and int(bad_move_splits[0]) > 0:
        raise RuntimeError(
            "Validation failed: moves from a game appear in multiple temporal splits"
        )

    assignments_cursor = connection.execute(
        "SELECT "
        "  game_id, temporal_split, player_disjoint_training_eligible, "
        "  is_player_holdout_game, white_player_hash, black_player_hash "
        f"FROM ({assignments_sql}) "
        "ORDER BY game_id"
    )

    holdout_flag_mismatches = 0
    heldout_leak_rows = 0
    eligibility_mismatches = 0
    holdout_flag_samples: list[str] = []
    leak_samples: list[str] = []
    eligibility_samples: list[str] = []

    for row in _iter_cursor_rows(assignments_cursor, chunk_size=10000):
        game_id = str(row[0])
        temporal_split = str(row[1])
        actual_player_disjoint_eligible = bool(row[2])
        actual_is_player_holdout_game = bool(row[3])
        white_player_hash = _as_optional_str(row[4])
        black_player_hash = _as_optional_str(row[5])

        white_is_holdout = (
            white_player_hash is not None
            and _is_holdout_player(config, white_player_hash)
        )
        black_is_holdout = (
            black_player_hash is not None
            and _is_holdout_player(config, black_player_hash)
        )
        expected_is_player_holdout_game = white_is_holdout or black_is_holdout

        if actual_is_player_holdout_game != expected_is_player_holdout_game:
            holdout_flag_mismatches += 1
            if len(holdout_flag_samples) < 5:
                holdout_flag_samples.append(
                    f"{game_id}: actual={actual_is_player_holdout_game}, "
                    f"expected={expected_is_player_holdout_game}, "
                    f"white_is_holdout={white_is_holdout}, "
                    f"black_is_holdout={black_is_holdout}"
                )

        missing_player_hash = white_player_hash is None or black_player_hash is None
        expected_player_disjoint_eligible = (
            temporal_split == "train"
            and not expected_is_player_holdout_game
            and (
                config.player_holdout.missing_player_hash_policy
                == "include_in_player_disjoint_training"
                or not missing_player_hash
            )
        )

        if actual_player_disjoint_eligible and expected_is_player_holdout_game:
            heldout_leak_rows += 1
            if len(leak_samples) < 5:
                leak_samples.append(
                    f"{game_id}: temporal_split={temporal_split}, "
                    f"white_is_holdout={white_is_holdout}, "
                    f"black_is_holdout={black_is_holdout}, "
                    f"white_player_hash={white_player_hash}, "
                    f"black_player_hash={black_player_hash}"
                )

        if actual_player_disjoint_eligible != expected_player_disjoint_eligible:
            eligibility_mismatches += 1
            if len(eligibility_samples) < 5:
                eligibility_samples.append(
                    f"{game_id}: actual={actual_player_disjoint_eligible}, "
                    f"expected={expected_player_disjoint_eligible}, "
                    f"temporal_split={temporal_split}, "
                    f"white_is_holdout={white_is_holdout}, "
                    f"black_is_holdout={black_is_holdout}, "
                    f"missing_player_hash={missing_player_hash}"
                )

    if holdout_flag_mismatches > 0:
        raise RuntimeError(
            "Validation failed: is_player_holdout_game does not match per-player "
            "holdout eligibility. "
            f"mismatch_rows={holdout_flag_mismatches}. "
            f"sample_rows={holdout_flag_samples}"
        )

    if heldout_leak_rows > 0:
        raise RuntimeError(
            "Validation failed: held-out player hash leaked into "
            "player-disjoint training population. "
            f"leak_rows={heldout_leak_rows}. "
            f"sample_rows={leak_samples}"
        )

    if eligibility_mismatches > 0:
        raise RuntimeError(
            "Validation failed: player_disjoint_training_eligible does not match "
            "recomputed player-holdout policy. "
            f"mismatch_rows={eligibility_mismatches}. "
            f"sample_rows={eligibility_samples}"
        )

    invalid_action = connection.execute(
        "SELECT COUNT(*) FROM ("
        f"  SELECT policy_target_action_index FROM ({primary_policy_sql}) "
        "  WHERE policy_target_action_index < 0 OR policy_target_action_index > 4671"
        ")"
    ).fetchone()
    if invalid_action is not None and int(invalid_action[0]) > 0:
        raise RuntimeError("Validation failed: policy_target_action_index outside [0, 4671]")


def _collect_split_summary(
    *,
    connection: duckdb.DuckDBPyConnection,
    assignments_sql: str,
    primary_policy_sql: str,
) -> dict[str, Any]:
    split_games_rows = connection.execute(
        "SELECT temporal_split, COUNT(*) FROM ("
        f"  {assignments_sql}"
        ") GROUP BY temporal_split ORDER BY temporal_split"
    ).fetchall()

    split_moves_rows = connection.execute(
        "SELECT temporal_split, COUNT(*) FROM ("
        f"  {primary_policy_sql}"
        ") GROUP BY temporal_split ORDER BY temporal_split"
    ).fetchall()

    split_players_rows = connection.execute(
        "WITH assignments AS ("
        f"  {assignments_sql}"
        "), players AS ("
        "  SELECT temporal_split, white_player_hash AS player_hash FROM assignments "
        "  UNION ALL "
        "  SELECT temporal_split, black_player_hash AS player_hash FROM assignments"
        ") "
        "SELECT temporal_split, COUNT(DISTINCT player_hash) "
        "FROM players WHERE player_hash IS NOT NULL "
        "GROUP BY temporal_split ORDER BY temporal_split"
    ).fetchall()

    split_positions_rows = connection.execute(
        "SELECT temporal_split, COUNT(DISTINCT position_id) FROM ("
        f"  {primary_policy_sql}"
        ") GROUP BY temporal_split ORDER BY temporal_split"
    ).fetchall()

    rating_band_rows = connection.execute(
        "SELECT temporal_split, mover_rating_band, COUNT(*) FROM ("
        f"  {primary_policy_sql}"
        ") GROUP BY temporal_split, mover_rating_band "
        "ORDER BY temporal_split, mover_rating_band"
    ).fetchall()

    time_control_rows = connection.execute(
        "SELECT temporal_split, time_control_category, COUNT(*) FROM ("
        f"  {primary_policy_sql}"
        ") GROUP BY temporal_split, time_control_category "
        "ORDER BY temporal_split, time_control_category"
    ).fetchall()

    temporal_order_rows = connection.execute(
        "SELECT temporal_split, MIN(played_date_iso), MAX(played_date_iso) FROM ("
        f"  {assignments_sql}"
        ") WHERE played_date_iso IS NOT NULL "
        "GROUP BY temporal_split"
    ).fetchall()

    order_map = {str(row[0]): (row[1], row[2]) for row in temporal_order_rows}
    train_max = order_map.get("train", (None, None))[1]
    validation_min = order_map.get("validation", (None, None))[0]
    validation_max = order_map.get("validation", (None, None))[1]
    test_min = order_map.get("test", (None, None))[0]

    if (
        train_max is not None
        and validation_min is not None
        and str(train_max) >= str(validation_min)
    ):
        raise RuntimeError("Validation failed: max(train_date) is not < min(validation_date)")
    if validation_max is not None and test_min is not None and str(validation_max) >= str(test_min):
        raise RuntimeError("Validation failed: max(validation_date) is not < min(test_date)")

    summary: dict[str, Any] = {
        "games_by_split": {str(split): int(count) for split, count in split_games_rows},
        "moves_by_split": {str(split): int(count) for split, count in split_moves_rows},
        "distinct_players_by_split": {
            str(split): int(count) for split, count in split_players_rows
        },
        "distinct_positions_by_split": {
            str(split): int(count) for split, count in split_positions_rows
        },
        "rating_band_rows": [
            {
                "temporal_split": str(split),
                "mover_rating_band": str(band),
                "row_count": int(count),
            }
            for split, band, count in rating_band_rows
        ],
        "time_control_rows": [
            {
                "temporal_split": str(split),
                "time_control_category": str(category),
                "row_count": int(count),
            }
            for split, category, count in time_control_rows
        ],
        "temporal_date_bounds": {
            str(split): {
                "min_played_date_iso": min_date,
                "max_played_date_iso": max_date,
            }
            for split, min_date, max_date in temporal_order_rows
        },
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 2.1 modeling dataset")
    parser.add_argument("--config", default="configs/modeling/fixture.yaml")
    parser.add_argument("--collection-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--warehouse-provenance-path", default=None)
    parser.add_argument("--date-enrichment-manifest-path", default=None)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def run_modeling_dataset_build(
    *,
    config_path: Path,
    collection_root_override: str | None = None,
    output_root_override: str | None = None,
    duckdb_path_override: str | None = None,
    warehouse_provenance_path_override: str | None = None,
    date_enrichment_manifest_path_override: str | None = None,
    max_games_override: int | None = None,
    max_examples_override: int | None = None,
    dry_run: bool = False,
    validate_only: bool = False,
) -> ModelingBuildResult:
    contract = default_feature_label_contract()
    validate_feature_label_contract(contract)

    config = load_modeling_config(config_path)
    config = apply_cli_overrides(
        config,
        collection_root=collection_root_override,
        output_root=output_root_override,
        duckdb_path=duckdb_path_override,
        warehouse_provenance_path=warehouse_provenance_path_override,
        date_enrichment_manifest_path=date_enrichment_manifest_path_override,
        max_games=max_games_override,
        max_examples=max_examples_override,
    )

    upstream = _load_collection_identity(config)
    date_enrichment = _load_date_enrichment_input(config=config, upstream=upstream)
    warehouse_provenance = load_warehouse_provenance(config.input.warehouse_provenance_path)
    _validate_warehouse_provenance_contract(
        config=config,
        upstream=upstream,
        warehouse_provenance=warehouse_provenance,
    )

    identity_payload = _dataset_identity_payload(
        config,
        upstream,
        warehouse_provenance,
        date_enrichment,
    )
    identity_payload_hash = sha256_text(canonical_json(identity_payload))
    modeling_dataset_id = _derive_modeling_dataset_id(
        config,
        upstream,
        warehouse_provenance,
        date_enrichment,
    )

    run_id = f"modeling-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    dataset_root = config.output.output_root / "datasets"
    final_dataset_path = dataset_root / modeling_dataset_id
    manifest_path = final_dataset_path / "_manifest.json"

    if final_dataset_path.exists():
        connection = _open_validated_duckdb_connection(
            config=config,
            upstream=upstream,
            warehouse_provenance=warehouse_provenance,
        )
        try:
            if date_enrichment is not None:
                _validate_date_enrichment_against_games_relation(
                    connection=connection,
                    config=config,
                    date_enrichment=date_enrichment,
                )
        finally:
            connection.close()

        existing_manifest = _validate_existing_dataset(
            dataset_path=final_dataset_path,
            expected_dataset_id=modeling_dataset_id,
            expected_identity_payload_hash=identity_payload_hash,
        )
        return _result_from_manifest(
            run_id=run_id,
            modeling_dataset_id=modeling_dataset_id,
            dataset_path=final_dataset_path,
            manifest_path=manifest_path,
            manifest=existing_manifest,
            reused_existing=True,
            validate_only=validate_only,
        )

    if validate_only:
        raise RuntimeError(
            "validate-only requested but published modeling dataset does not exist: "
            f"{final_dataset_path.as_posix()}"
        )

    if dry_run:
        return ModelingBuildResult(
            run_id=run_id,
            modeling_dataset_id=modeling_dataset_id,
            collection_id=str(upstream["collection_id"]),
            dataset_path=final_dataset_path,
            manifest_path=manifest_path,
            reused_existing=False,
            selected_games=0,
            selected_examples=0,
            split_counts_games={},
            split_counts_examples={},
            novel_position_test_rows=0,
            duration_seconds=None,
            peak_rss_bytes=None,
            dry_run=True,
            validate_only=False,
        )

    staging_root = config.output.output_root / "staging" / f"{modeling_dataset_id}__{run_id}"
    _safe_rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=False)

    connection = _open_validated_duckdb_connection(
        config=config,
        upstream=upstream,
        warehouse_provenance=warehouse_provenance,
    )

    date_enrichment_coverage: dict[str, int] | None = None
    if date_enrichment is not None:
        date_enrichment_coverage = _validate_date_enrichment_against_games_relation(
            connection=connection,
            config=config,
            date_enrichment=date_enrichment,
        )

    started = time.perf_counter()
    sampler = PeakRssSampler(interval_seconds=0.05)
    sampler.start()
    published = False

    try:
        sampling_stages = _sampling_stage_summary(
            config=config,
            warehouse_provenance=warehouse_provenance,
        )

        writer = ModelingParquetWriter(
            dataset_root=staging_root,
            compression=config.output.parquet_compression,
            row_group_size=config.output.parquet_row_group_size,
        )

        assignment_counts, assignment_files, game_rejections, sampling_stats, date_source_counts = (
            _build_game_assignments(
                connection=connection,
                config=config,
                writer=writer,
                date_enrichment=date_enrichment,
            )
        )

        move_counts, primary_policy_files, policy_rejections = _build_policy_examples(
            connection=connection,
            config=config,
            writer=writer,
            assignment_files=assignment_files,
        )

        novel_slice_count = _write_novel_position_slice(
            connection=connection,
            writer=writer,
            primary_policy_files=primary_policy_files,
            batch_rows=config.output.batch_rows,
        )

        assignments_sql = f"SELECT * FROM read_parquet({_sql_file_list(assignment_files)})"
        primary_policy_sql = f"SELECT * FROM read_parquet({_sql_file_list(primary_policy_files)})"

        _validate_split_and_holdout_constraints(
            connection=connection,
            assignments_sql=assignments_sql,
            primary_policy_sql=primary_policy_sql,
            config=config,
        )
        split_summary = _collect_split_summary(
            connection=connection,
            assignments_sql=assignments_sql,
            primary_policy_sql=primary_policy_sql,
        )

        leakage_metrics = compute_position_overlap_metrics(
            connection=connection,
            relation_sql=primary_policy_sql,
        )
        novel_count_check = compute_novel_position_test_count(
            connection=connection,
            relation_sql=primary_policy_sql,
        )
        if novel_count_check != novel_slice_count:
            raise RuntimeError(
                "Novel-position test row count mismatch between query and written output: "
                f"{novel_count_check} != {novel_slice_count}"
            )

        leakage_dir = staging_root / "leakage_audits"
        leakage_dir.mkdir(parents=True, exist_ok=True)
        leakage_path = leakage_dir / "temporal_position_overlap.json"
        leakage_payload = {
            "phase": "2.1",
            "evidence_type": "position_overlap_leakage_audit",
            "novel_position_test_row_count": novel_slice_count,
            "metrics": leakage_metrics,
        }
        leakage_path.write_text(
            json.dumps(leakage_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        writer_stats = writer.stats()
        duration_seconds = max(time.perf_counter() - started, 0.0)
        peak_rss_bytes = sampler.stop()

        manifest: dict[str, Any] = {
            "manifest_version": "1.0.0",
            "status": "complete",
            "modeling_dataset_id": modeling_dataset_id,
            "run_id": run_id,
            "created_at_utc": _utc_now_iso(),
            "git_commit": _git_commit(),
            "identity_payload_sha256": identity_payload_hash,
            "upstream": {
                "input_kind": "collection",
                "collection_id": upstream["collection_id"],
                "collection_manifest_sha256": upstream["collection_manifest_sha256"],
                "warehouse_provenance_version": warehouse_provenance.provenance_version,
                "warehouse_provenance_sha256": warehouse_provenance.warehouse_provenance_sha256,
                "warehouse_kind": warehouse_provenance.warehouse_kind,
                "warehouse_provenance": warehouse_provenance.payload,
                "date_enrichment": (
                    None
                    if date_enrichment is None
                    else {
                        "date_enrichment_id": date_enrichment.date_enrichment_id,
                        "manifest_sha256": date_enrichment.manifest_sha256,
                        "source_month": date_enrichment.source_month,
                        "parent_archive_sha256": date_enrichment.parent_archive_sha256,
                        "resolver_version": date_enrichment.resolver_version,
                        "coverage": date_enrichment_coverage,
                    }
                ),
            },
            "versions": {
                "modeling_pipeline_version": config.versions.modeling_pipeline_version,
                "split_definition_version": config.versions.split_definition_version,
                "feature_schema_version": config.versions.feature_schema_version,
                "label_definition_version": config.versions.label_definition_version,
                "schema_version": config.versions.schema_version,
                "position_normalization_version": config.versions.position_normalization_version,
                "board_encoding_version": config.versions.board_encoding_version,
                "action_encoding_version": config.versions.action_encoding_version,
                "canonical_played_date_resolver_version": (
                    CANONICAL_PLAYED_DATE_RESOLVER_VERSION
                    if date_enrichment is not None
                    else None
                ),
            },
            "sampling": {
                "rule_version": config.sampling.rule_version,
                "seed": config.sampling.seed,
                "hash_modulus": config.sampling.hash_modulus,
                "hash_threshold": config.sampling.hash_threshold,
                "requested_rate_percent": config.sampling.requested_rate_percent,
                "effective_rate_percent": effective_rate_percent(
                    hash_modulus=config.sampling.hash_modulus,
                    hash_threshold=config.sampling.hash_threshold,
                ),
                "max_games": config.sampling.max_games,
                "games_scanned": int(sampling_stats.get("games_scanned", 0)),
                "games_selected_by_rule": int(sampling_stats.get("games_selected_by_rule", 0)),
                "games_selected_after_cap": (
                    int(sampling_stats.get("games_selected_after_cap", 0))
                    if config.sampling.max_games is not None
                    else None
                ),
                "games_selected_after_split_policy": int(
                    sampling_stats.get("games_selected_after_split_policy", 0)
                ),
            },
            "sampling_stages": sampling_stages,
            "splits": {
                "definition_version": config.versions.split_definition_version,
                "train": {
                    "start_date": config.splits.train.start_date.isoformat(),
                    "end_date": config.splits.train.end_date.isoformat(),
                },
                "validation": {
                    "start_date": config.splits.validation.start_date.isoformat(),
                    "end_date": config.splits.validation.end_date.isoformat(),
                },
                "test": {
                    "start_date": config.splits.test.start_date.isoformat(),
                    "end_date": config.splits.test.end_date.isoformat(),
                },
                "missing_or_invalid_date_policy": config.splits.missing_or_invalid_date_policy,
            },
            "player_holdout": {
                "rule_version": config.player_holdout.rule_version,
                "seed": config.player_holdout.seed,
                "hash_modulus": config.player_holdout.hash_modulus,
                "hash_threshold": config.player_holdout.hash_threshold,
                "requested_rate_percent": config.player_holdout.requested_rate_percent,
                "effective_rate_percent": effective_rate_percent(
                    hash_modulus=config.player_holdout.hash_modulus,
                    hash_threshold=config.player_holdout.hash_threshold,
                ),
                "missing_player_hash_policy": config.player_holdout.missing_player_hash_policy,
            },
            "counts": {
                "game_assignments_by_split": assignment_counts,
                "policy_examples_by_split": move_counts,
                "novel_position_test_rows": novel_slice_count,
                "game_assignments_by_played_date_source": dict(date_source_counts),
                "selected_games": int(sum(assignment_counts.values())),
                "selected_policy_examples": int(sum(move_counts.values())),
                "rejected_games": int(sum(game_rejections.values())),
                "rejected_policy_examples": int(sum(policy_rejections.values())),
            },
            "rejections": {
                "games": dict(game_rejections),
                "policy_examples": dict(policy_rejections),
            },
            "split_summary": split_summary,
            "leakage_audit": leakage_payload,
            "datasets": {
                name: {
                    partition: {
                        "part_count": stats.part_count,
                        "row_count": stats.row_count,
                        "total_bytes": stats.total_bytes,
                        "relative_files": list(stats.relative_files),
                    }
                    for partition, stats in partitions.items()
                }
                for name, partitions in writer_stats.items()
            },
            "leakage_audit_files": [leakage_path.relative_to(staging_root).as_posix()],
            "performance": {
                "duration_seconds": round(duration_seconds, 6),
                "peak_rss_bytes": int(peak_rss_bytes),
                "games_per_second": round(
                    int(sum(assignment_counts.values())) / duration_seconds,
                    4,
                )
                if duration_seconds > 0
                else 0.0,
                "examples_per_second": round(
                    int(sum(move_counts.values())) / duration_seconds,
                    4,
                )
                if duration_seconds > 0
                else 0.0,
            },
        }

        atomic_write_json(staging_root / "_manifest.json", manifest)
        (staging_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")

        dataset_root.mkdir(parents=True, exist_ok=True)
        if final_dataset_path.exists():
            raise RuntimeError(
                "Modeling dataset appeared during build; refusing to overwrite: "
                f"{final_dataset_path.as_posix()}"
            )
        os.replace(staging_root, final_dataset_path)
        published = True

        return _result_from_manifest(
            run_id=run_id,
            modeling_dataset_id=modeling_dataset_id,
            dataset_path=final_dataset_path,
            manifest_path=final_dataset_path / "_manifest.json",
            manifest=manifest,
            reused_existing=False,
            validate_only=False,
        )
    finally:
        connection.close()
        if not published:
            sampler.stop()
            _safe_rmtree(staging_root)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        collection_root_override = resolve_collection_root_override_or_env(args.collection_root)
    except RuntimePathResolutionError as exc:
        parser.error(str(exc))

    duckdb_path_override = resolve_duckdb_path_override_or_env(
        args.duckdb_path,
        default_path=None,
    )
    warehouse_provenance_path_override = resolve_warehouse_provenance_path_override_or_env(
        args.warehouse_provenance_path
    )
    date_enrichment_manifest_path_override = (
        resolve_date_enrichment_manifest_override_or_env(
            args.date_enrichment_manifest_path
        )
    )

    result = run_modeling_dataset_build(
        config_path=Path(args.config),
        collection_root_override=(
            None
            if collection_root_override is None
            else collection_root_override.as_posix()
        ),
        output_root_override=args.output_root,
        duckdb_path_override=(
            None if duckdb_path_override is None else duckdb_path_override.as_posix()
        ),
        warehouse_provenance_path_override=(
            None
            if warehouse_provenance_path_override is None
            else warehouse_provenance_path_override.as_posix()
        ),
        date_enrichment_manifest_path_override=(
            None
            if date_enrichment_manifest_path_override is None
            else date_enrichment_manifest_path_override.as_posix()
        ),
        max_games_override=args.max_games,
        max_examples_override=args.max_examples,
        dry_run=bool(args.dry_run),
        validate_only=bool(args.validate_only),
    )
    print(json.dumps(modeling_build_result_to_json(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
