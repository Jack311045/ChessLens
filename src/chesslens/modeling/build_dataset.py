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


@dataclass(frozen=True)
class ModelingBuildResult:
    run_id: str
    modeling_dataset_id: str
    dataset_path: Path
    manifest_path: Path
    reused_existing: bool
    selected_games: int
    selected_examples: int


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


def _dataset_identity_payload(config: ModelingConfig, upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "upstream": {
            "collection_id": upstream["collection_id"],
            "collection_manifest_sha256": upstream["collection_manifest_sha256"],
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
            "strict": config.behavior.strict,
        },
    }


def _derive_modeling_dataset_id(config: ModelingConfig, upstream: dict[str, Any]) -> str:
    return sha256_text(canonical_json(_dataset_identity_payload(config, upstream)))


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
) -> tuple[dict[str, Any], list[str], Counter[str], Counter[str]]:
    query = (
        "SELECT game_id, source_month, played_date, result, white_player_hash, "
        "black_player_hash, white_rating, black_rating, time_control_raw, eco, opening, ply_count "
        f"FROM {config.input.games_relation} ORDER BY game_id"
    )

    game_rejections: Counter[str] = Counter()
    sampling_stats: Counter[str] = Counter()
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
                "played_date": row[2],
                "result": row[3],
                "white_player_hash": row[4],
                "black_player_hash": row[5],
                "white_rating": row[6],
                "black_rating": row[7],
                "time_control_raw": row[8],
                "eco": row[9],
                "opening": row[10],
                "ply_count": row[11],
                "sample_score_u64": sample_score,
            }
            _append_assignment_candidate(
                candidate=candidate,
                config=config,
                out_buffer=assignments_buffer,
                game_rejections=game_rejections,
                sampling_stats=sampling_stats,
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
                    "played_date": row[2],
                    "result": row[3],
                    "white_player_hash": row[4],
                    "black_player_hash": row[5],
                    "white_rating": row[6],
                    "black_rating": row[7],
                    "time_control_raw": row[8],
                    "eco": row[9],
                    "opening": row[10],
                    "ply_count": row[11],
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
    return assignment_counts, assignment_files_for_sql, game_rejections, sampling_stats


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
) -> None:
    parsed = parse_played_date(_as_optional_str(candidate["played_date"]))
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
            "played_date_raw": parsed.played_date_raw,
            "played_date_iso": parsed.played_date_iso,
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
        "  a.played_date_iso, a.white_player_hash, a.black_player_hash "
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
                "position_id": str(row[3]),
                "pre_move_fen": str(row[4]),
                "normalized_pre_move_fen": str(row[5]),
                "side_to_move": str(row[6]),
                "time_control_raw": _as_optional_str(row[10]),
                "time_control_category": time_control_category(_as_optional_str(row[10])),
                "eco": _as_optional_str(row[11]),
                "opening": _as_optional_str(row[12]),
                "white_player_hash": _as_optional_str(row[19]),
                "black_player_hash": _as_optional_str(row[20]),
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

    heldout_leak = connection.execute(
        "WITH assignments AS ("
        f"  SELECT * FROM ({assignments_sql})"
        "), heldout_players AS ("
        "  SELECT DISTINCT white_player_hash AS player_hash FROM assignments "
        "  WHERE is_player_holdout_game AND white_player_hash IS NOT NULL "
        "  UNION "
        "  SELECT DISTINCT black_player_hash AS player_hash FROM assignments "
        "  WHERE is_player_holdout_game AND black_player_hash IS NOT NULL"
        ") "
        "SELECT COUNT(*) FROM assignments a "
        "WHERE a.player_disjoint_training_eligible "
        "AND (a.white_player_hash IN (SELECT player_hash FROM heldout_players) "
        "  OR a.black_player_hash IN (SELECT player_hash FROM heldout_players))"
    ).fetchone()
    if heldout_leak is not None and int(heldout_leak[0]) > 0:
        raise RuntimeError(
            "Validation failed: held-out player hash leaked into "
            "player-disjoint training population"
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
        max_games=max_games_override,
        max_examples=max_examples_override,
    )

    upstream = _load_collection_identity(config)
    identity_payload = _dataset_identity_payload(config, upstream)
    identity_payload_hash = sha256_text(canonical_json(identity_payload))
    modeling_dataset_id = _derive_modeling_dataset_id(config, upstream)

    run_id = f"modeling-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    dataset_root = config.output.output_root / "datasets"
    final_dataset_path = dataset_root / modeling_dataset_id
    manifest_path = final_dataset_path / "_manifest.json"

    if final_dataset_path.exists():
        _validate_existing_dataset(
            dataset_path=final_dataset_path,
            expected_dataset_id=modeling_dataset_id,
            expected_identity_payload_hash=identity_payload_hash,
        )
        return ModelingBuildResult(
            run_id=run_id,
            modeling_dataset_id=modeling_dataset_id,
            dataset_path=final_dataset_path,
            manifest_path=manifest_path,
            reused_existing=True,
            selected_games=0,
            selected_examples=0,
        )

    if validate_only:
        raise RuntimeError(
            "validate-only requested but published modeling dataset does not exist: "
            f"{final_dataset_path.as_posix()}"
        )

    if dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "modeling_dataset_id": modeling_dataset_id,
                    "collection_id": upstream["collection_id"],
                    "identity_payload_sha256": identity_payload_hash,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return ModelingBuildResult(
            run_id=run_id,
            modeling_dataset_id=modeling_dataset_id,
            dataset_path=final_dataset_path,
            manifest_path=manifest_path,
            reused_existing=False,
            selected_games=0,
            selected_examples=0,
        )

    if not config.input.duckdb_path.exists():
        raise FileNotFoundError(
            "Configured duckdb_path does not exist. Run preflight and dbt build first: "
            f"{config.input.duckdb_path.as_posix()}"
        )

    staging_root = config.output.output_root / "staging" / f"{modeling_dataset_id}__{run_id}"
    _safe_rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    sampler = PeakRssSampler(interval_seconds=0.05)
    sampler.start()

    connection = duckdb.connect(str(config.input.duckdb_path))
    published = False

    try:
        _ensure_relation_exists(connection, config.input.games_relation)
        _ensure_relation_exists(connection, config.input.move_context_relation)

        writer = ModelingParquetWriter(
            dataset_root=staging_root,
            compression=config.output.parquet_compression,
            row_group_size=config.output.parquet_row_group_size,
        )

        assignment_counts, assignment_files, game_rejections, sampling_stats = (
            _build_game_assignments(
                connection=connection,
                config=config,
                writer=writer,
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

        result = ModelingBuildResult(
            run_id=run_id,
            modeling_dataset_id=modeling_dataset_id,
            dataset_path=final_dataset_path,
            manifest_path=final_dataset_path / "_manifest.json",
            reused_existing=False,
            selected_games=int(sum(assignment_counts.values())),
            selected_examples=int(sum(move_counts.values())),
        )

        print(
            json.dumps(
                {
                    "modeling_dataset_id": result.modeling_dataset_id,
                    "run_id": result.run_id,
                    "collection_id": upstream["collection_id"],
                    "selected_games": result.selected_games,
                    "selected_examples": result.selected_examples,
                    "split_counts_games": assignment_counts,
                    "split_counts_examples": move_counts,
                    "novel_position_test_rows": novel_slice_count,
                    "reused_existing": result.reused_existing,
                    "manifest_path": _display_path(result.manifest_path),
                    "duration_seconds": manifest["performance"]["duration_seconds"],
                    "peak_rss_bytes": manifest["performance"]["peak_rss_bytes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return result
    finally:
        connection.close()
        if not published:
            sampler.stop()
            _safe_rmtree(staging_root)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_modeling_dataset_build(
        config_path=Path(args.config),
        collection_root_override=args.collection_root,
        output_root_override=args.output_root,
        duckdb_path_override=args.duckdb_path,
        max_games_override=args.max_games,
        max_examples_override=args.max_examples,
        dry_run=bool(args.dry_run),
        validate_only=bool(args.validate_only),
    )


if __name__ == "__main__":
    main()
