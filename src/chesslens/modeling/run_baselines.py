"""Phase 2.2 classical baselines runner with leakage-aware evaluation."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import shutil
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from chesslens.features.action_encoding import encode_move_to_index, legal_move_indices
from chesslens.ingestion.rss import PeakRssSampler
from chesslens.ingestion.shard_manifest import atomic_write_json
from chesslens.modeling.baselines_config import (
    BaselineConfig,
    apply_baseline_cli_overrides,
    baseline_config_identity_payload,
    load_baseline_config,
)
from chesslens.modeling.baselines_metrics import (
    RankingGroup,
    compute_ranking_metrics,
    expected_calibration_error,
    multiclass_brier_score,
    reliability_bins_per_class,
)
from chesslens.modeling.labels import rating_band, time_control_category
from chesslens.modeling.validation import (
    canonical_json,
    read_json_object,
    sha256_text,
    validate_manifest_identity,
)

WDL_CLASS_ORDER: tuple[str, str, str] = ("win", "draw", "loss")

POLICY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "ply",
    "temporal_split",
    "source_month",
    "position_id",
    "pre_move_fen",
    "normalized_pre_move_fen",
    "side_to_move",
    "time_control_raw",
    "time_control_category",
    "eco",
    "opening",
    "mover_rating",
    "opponent_rating",
    "rating_difference",
    "mover_rating_band",
    "player_disjoint_training_eligible",
    "is_player_holdout_game",
    "policy_target_action_index",
    "value_target_wdl",
)

GAME_ASSIGNMENT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "temporal_split",
    "white_player_hash",
    "black_player_hash",
)

FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "final_result_target",
    "policy_target_action_index",
    "played_move_uci_target",
    "played_move_uci",
    "post_move_fen",
    "engine_centipawn_score",
    "engine_mate_score",
    "blunder_label",
    "future_moves",
    "future_clock_annotation",
    "termination_target",
    "value_target_wdl",
)

FEATURE_DEFINITIONS: dict[str, str] = {
    "side_to_move": "Color to move from pre-move position.",
    "ply": "Move ply index in game.",
    "legal_move_count": "Number of legal moves from pre-move board.",
    "side_in_check": "Whether side to move is currently in check.",
    "white_castle_k": "White king-side castling right flag.",
    "white_castle_q": "White queen-side castling right flag.",
    "black_castle_k": "Black king-side castling right flag.",
    "black_castle_q": "Black queen-side castling right flag.",
    "material_balance": "White material minus black material (pawns=1,...,queens=9).",
    "phase": "Opening/middlegame/endgame phase derived from board material.",
    "mover_rating": "Side-to-move rating from upstream pre-move context.",
    "opponent_rating": "Opponent rating from upstream pre-move context.",
    "rating_difference": "Mover rating minus opponent rating.",
    "mover_rating_band": "Versioned rating band bucket.",
    "time_control_category": "Bullet/blitz/rapid/classical/unknown category.",
    "eco": "ECO code if present at prediction point.",
    "opening_family": "Opening family derived from ECO prefix.",
    "action_index": "Canonical legal action index for candidate move.",
    "from_square": "Candidate move from-square index.",
    "to_square": "Candidate move to-square index.",
    "moving_piece_type": "Piece type making candidate move.",
    "is_capture": "Candidate move is capture flag.",
    "captured_piece_type": "Captured piece type for capture moves.",
    "is_promotion": "Candidate move promotion flag.",
    "promotion_piece_type": "Promotion piece type when promoted.",
    "is_castling": "Candidate move castling flag.",
    "is_en_passant": "Candidate move en-passant flag.",
    "gives_check": "Candidate move gives-check flag.",
    "delta_file": "File delta between from-square and to-square.",
    "delta_rank": "Rank delta between from-square and to-square.",
    "distance_chebyshev": "Max absolute file/rank delta.",
    "distance_manhattan": "Manhattan move distance.",
}


@dataclass(frozen=True)
class PositionExample:
    split: str
    game_id: str
    ply: int
    source_month: str | None
    position_id: str
    pre_move_fen: str
    normalized_pre_move_fen: str
    side_to_move: str
    time_control_raw: str | None
    time_control_category: str
    eco: str | None
    opening: str | None
    mover_rating: int | None
    opponent_rating: int | None
    rating_difference: int | None
    mover_rating_band: str
    player_disjoint_training_eligible: bool
    is_player_holdout_game: bool
    policy_target_action_index: int
    value_target_wdl: str
    novel_position_test: bool


@dataclass(frozen=True)
class CandidateGroup:
    split: str
    game_id: str
    group_id: str
    action_indices: tuple[int, ...]
    labels: tuple[int, ...]
    feature_dicts: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreflightSummary:
    modeling_manifest_path: Path
    dataset_path: Path
    modeling_dataset_id: str
    modeling_manifest_sha256: str
    collection_id: str
    feature_schema_version: str
    split_definition_version: str
    source_months: tuple[str, ...]
    policy_paths_by_split: dict[str, list[Path]]
    game_assignment_paths_by_split: dict[str, list[Path]]
    novel_position_paths: list[Path]
    split_row_counts: dict[str, int]
    game_id_overlap_counts: dict[str, int]
    normalized_fen_overlap_counts: dict[str, int]
    player_overlap_counts: dict[str, int]
    label_legality_checked_rows: int


@dataclass(frozen=True)
class BaselineRunResult:
    run_id: str
    experiment_id: str
    reused_existing: bool
    output_path: Path
    manifest_path: Path
    duration_seconds: float | None
    peak_rss_bytes: int | None
    dry_run: bool
    validate_only: bool
    summary: dict[str, Any] | None = None


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


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _sql_file_list(paths: list[Path]) -> str:
    escaped = [p.as_posix().replace("'", "''") for p in paths]
    return "[" + ", ".join(f"'{item}'" for item in escaped) + "]"


def _stable_hash_u64(payload: str) -> int:
    digest = sha256_text(payload)
    return int(digest[:16], 16)


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _load_manifest_identity(manifest_path: Path) -> tuple[dict[str, Any], str]:
    payload = read_json_object(manifest_path)
    manifest_hash = sha256_text(canonical_json(payload))
    return payload, manifest_hash


def _required_partition_paths(
    manifest: dict[str, Any],
    *,
    dataset_name: str,
    partition: str,
    dataset_path: Path,
) -> list[Path]:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise RuntimeError("Modeling manifest missing datasets object")
    dataset_payload = datasets.get(dataset_name)
    if not isinstance(dataset_payload, dict):
        raise RuntimeError(f"Modeling manifest missing datasets.{dataset_name}")
    partition_payload = dataset_payload.get(partition)
    if not isinstance(partition_payload, dict):
        raise RuntimeError(
            f"Modeling manifest missing datasets.{dataset_name}.{partition} partition"
        )
    relative_files = partition_payload.get("relative_files")
    if not isinstance(relative_files, list) or not relative_files:
        raise RuntimeError(
            f"Modeling manifest partition datasets.{dataset_name}.{partition} has no files"
        )

    resolved: list[Path] = []
    for rel in relative_files:
        rel_text = str(rel)
        full = dataset_path / rel_text
        if not full.exists():
            raise RuntimeError(
                f"Modeling manifest references missing file {rel_text!r} in partition {partition!r}"
            )
        resolved.append(full)
    return resolved


def _required_columns_exist(paths: list[Path], required_columns: tuple[str, ...]) -> None:
    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            f"SELECT * FROM read_parquet({_sql_file_list(paths)}) LIMIT 0"
        )
        actual = [item[0] for item in cursor.description]
    finally:
        connection.close()

    missing = sorted(column for column in required_columns if column not in actual)
    if missing:
        raise RuntimeError("Required columns missing from modeling dataset: " + ", ".join(missing))


def _count_rows(paths: list[Path]) -> int:
    connection = duckdb.connect()
    try:
        row = connection.execute(
            f"SELECT COUNT(*) FROM read_parquet({_sql_file_list(paths)})"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("Unable to count rows for split files")
    return int(row[0])


def _distinct_source_months(policy_paths_by_split: dict[str, list[Path]]) -> tuple[str, ...]:
    all_paths: list[Path] = []
    for paths in policy_paths_by_split.values():
        all_paths.extend(paths)

    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "SELECT DISTINCT source_month FROM read_parquet(" + _sql_file_list(all_paths) + ") "
            "WHERE source_month IS NOT NULL ORDER BY source_month"
        ).fetchall()
    finally:
        connection.close()

    return tuple(str(row[0]) for row in rows if row[0] is not None)


def _overlap_count(paths_a: list[Path], paths_b: list[Path], *, column: str) -> int:
    connection = duckdb.connect()
    try:
        query = (
            "WITH a AS (SELECT DISTINCT "
            + column
            + " AS key FROM read_parquet("
            + _sql_file_list(paths_a)
            + ") WHERE "
            + column
            + " IS NOT NULL), "
            "b AS (SELECT DISTINCT "
            + column
            + " AS key FROM read_parquet("
            + _sql_file_list(paths_b)
            + ") WHERE "
            + column
            + " IS NOT NULL) "
            "SELECT COUNT(*) FROM (SELECT key FROM a INTERSECT SELECT key FROM b)"
        )
        row = connection.execute(query).fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError("Unable to compute overlap count")
    return int(row[0])


def _player_overlap(
    game_assignment_paths_by_split: dict[str, list[Path]],
    split_a: str,
    split_b: str,
) -> int:
    paths_a = game_assignment_paths_by_split[split_a]
    paths_b = game_assignment_paths_by_split[split_b]

    connection = duckdb.connect()
    try:
        query = (
            "WITH a AS ("
            "  SELECT white_player_hash AS player_hash FROM read_parquet("
            + _sql_file_list(paths_a)
            + ") UNION ALL "
            "  SELECT black_player_hash AS player_hash FROM read_parquet("
            + _sql_file_list(paths_a)
            + ")"
            "), b AS ("
            "  SELECT white_player_hash AS player_hash FROM read_parquet("
            + _sql_file_list(paths_b)
            + ") UNION ALL "
            "  SELECT black_player_hash AS player_hash FROM read_parquet("
            + _sql_file_list(paths_b)
            + ")"
            ") "
            "SELECT COUNT(*) FROM ("
            "SELECT DISTINCT player_hash FROM a WHERE player_hash IS NOT NULL "
            "INTERSECT "
            "SELECT DISTINCT player_hash FROM b WHERE player_hash IS NOT NULL"
            ")"
        )
        row = connection.execute(query).fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError("Unable to compute player overlap")
    return int(row[0])


def _validate_label_legality(policy_paths_by_split: dict[str, list[Path]]) -> int:
    checked = 0
    for split, paths in policy_paths_by_split.items():
        connection = duckdb.connect()
        try:
            cursor = connection.execute(
                "SELECT game_id, ply, pre_move_fen, policy_target_action_index "
                "FROM read_parquet(" + _sql_file_list(paths) + ")"
            )
            while True:
                rows = cursor.fetchmany(512)
                if not rows:
                    break
                for game_id, ply, fen, target in rows:
                    checked += 1
                    board = chess.Board(str(fen))
                    legal_indices = set(legal_move_indices(board))
                    if int(target) not in legal_indices:
                        raise RuntimeError(
                            "Illegal policy target action index in modeling dataset: "
                            f"split={split}, game_id={game_id}, ply={ply}, target={target}"
                        )
        finally:
            connection.close()

    return checked


def _preflight_modeling_dataset(config: BaselineConfig) -> PreflightSummary:
    manifest_path = config.input.modeling_manifest_path
    manifest, manifest_hash = _load_manifest_identity(manifest_path)

    if str(manifest.get("status", "")) != "complete":
        raise RuntimeError("Modeling dataset manifest status is not complete")

    dataset_path = manifest_path.parent
    success_marker = dataset_path / "_SUCCESS"
    if not success_marker.exists():
        raise RuntimeError(
            "Modeling dataset is missing _SUCCESS marker: "
            f"{success_marker.as_posix()}"
        )

    modeling_dataset_id = str(manifest.get("modeling_dataset_id", "")).strip()
    if not modeling_dataset_id:
        raise RuntimeError("Modeling manifest missing modeling_dataset_id")

    upstream = manifest.get("upstream")
    if not isinstance(upstream, dict):
        raise RuntimeError("Modeling manifest missing upstream object")
    collection_id = str(upstream.get("collection_id", "")).strip()
    if not collection_id:
        raise RuntimeError("Modeling manifest missing upstream.collection_id")

    versions = manifest.get("versions")
    if not isinstance(versions, dict):
        raise RuntimeError("Modeling manifest missing versions object")

    feature_schema_version = str(versions.get("feature_schema_version", "")).strip()
    split_definition_version = str(
        manifest.get("splits", {}).get("definition_version", "")
    ).strip()
    if feature_schema_version != config.versions.feature_schema_version:
        raise RuntimeError(
            "Feature schema version mismatch: "
            f"manifest={feature_schema_version!r}, "
            f"config={config.versions.feature_schema_version!r}"
        )
    if split_definition_version != config.versions.split_definition_version:
        raise RuntimeError(
            "Split definition version mismatch: "
            f"manifest={split_definition_version!r}, "
            f"config={config.versions.split_definition_version!r}"
        )

    policy_paths_by_split = {
        split: _required_partition_paths(
            manifest,
            dataset_name="policy_examples",
            partition=split,
            dataset_path=dataset_path,
        )
        for split in ("train", "validation", "test")
    }

    game_assignment_paths_by_split = {
        split: _required_partition_paths(
            manifest,
            dataset_name="game_assignments",
            partition=split,
            dataset_path=dataset_path,
        )
        for split in ("train", "validation", "test")
    }

    novel_position_paths = _required_partition_paths(
        manifest,
        dataset_name="policy_examples",
        partition="novel_position_test",
        dataset_path=dataset_path,
    )

    for split in ("train", "validation", "test"):
        _required_columns_exist(policy_paths_by_split[split], POLICY_REQUIRED_COLUMNS)
        _required_columns_exist(
            game_assignment_paths_by_split[split],
            GAME_ASSIGNMENT_REQUIRED_COLUMNS,
        )

    split_row_counts = {split: _count_rows(paths) for split, paths in policy_paths_by_split.items()}

    game_id_overlap_counts = {
        "train_validation": _overlap_count(
            policy_paths_by_split["train"], policy_paths_by_split["validation"], column="game_id"
        ),
        "train_test": _overlap_count(
            policy_paths_by_split["train"], policy_paths_by_split["test"], column="game_id"
        ),
        "validation_test": _overlap_count(
            policy_paths_by_split["validation"], policy_paths_by_split["test"], column="game_id"
        ),
    }

    if any(count > 0 for count in game_id_overlap_counts.values()):
        raise RuntimeError(
            "Game-ID overlap detected across train/validation/test splits: "
            + json.dumps(game_id_overlap_counts, sort_keys=True)
        )

    normalized_fen_overlap_counts = {
        "train_validation": _overlap_count(
            policy_paths_by_split["train"],
            policy_paths_by_split["validation"],
            column="normalized_pre_move_fen",
        ),
        "train_test": _overlap_count(
            policy_paths_by_split["train"],
            policy_paths_by_split["test"],
            column="normalized_pre_move_fen",
        ),
        "validation_test": _overlap_count(
            policy_paths_by_split["validation"],
            policy_paths_by_split["test"],
            column="normalized_pre_move_fen",
        ),
    }

    player_overlap_counts = {
        "train_validation": _player_overlap(game_assignment_paths_by_split, "train", "validation"),
        "train_test": _player_overlap(game_assignment_paths_by_split, "train", "test"),
        "validation_test": _player_overlap(
            game_assignment_paths_by_split, "validation", "test"
        ),
    }

    label_legality_checked_rows = _validate_label_legality(policy_paths_by_split)

    source_months = _distinct_source_months(policy_paths_by_split)

    return PreflightSummary(
        modeling_manifest_path=manifest_path,
        dataset_path=dataset_path,
        modeling_dataset_id=modeling_dataset_id,
        modeling_manifest_sha256=manifest_hash,
        collection_id=collection_id,
        feature_schema_version=feature_schema_version,
        split_definition_version=split_definition_version,
        source_months=source_months,
        policy_paths_by_split=policy_paths_by_split,
        game_assignment_paths_by_split=game_assignment_paths_by_split,
        novel_position_paths=novel_position_paths,
        split_row_counts=split_row_counts,
        game_id_overlap_counts=game_id_overlap_counts,
        normalized_fen_overlap_counts=normalized_fen_overlap_counts,
        player_overlap_counts=player_overlap_counts,
        label_legality_checked_rows=label_legality_checked_rows,
    )


def _novel_position_ids(novel_paths: list[Path]) -> set[str]:
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "SELECT DISTINCT position_id FROM read_parquet(" + _sql_file_list(novel_paths) + ")"
        ).fetchall()
    finally:
        connection.close()
    return {str(row[0]) for row in rows if row[0] is not None}


def _load_position_examples(
    paths: list[Path],
    *,
    split: str,
    limit: int | None,
    seed: int,
    novel_position_ids: set[str],
) -> list[PositionExample]:
    cols = ", ".join(POLICY_REQUIRED_COLUMNS)
    base = "SELECT " + cols + " FROM read_parquet(" + _sql_file_list(paths) + ")"

    if limit is not None and limit > 0:
        seed_token = str(seed).replace("'", "''")
        query = (
            base
            + " ORDER BY hash(concat(game_id, ':', CAST(ply AS VARCHAR), ':', '"
            + seed_token
            + "')) LIMIT "
            + str(limit)
        )
    else:
        query = base

    connection = duckdb.connect()
    try:
        rows = connection.execute(query).fetchall()
    finally:
        connection.close()

    examples: list[PositionExample] = []
    for row in rows:
        (
            game_id,
            ply,
            temporal_split,
            source_month,
            position_id,
            pre_move_fen,
            normalized_pre_move_fen,
            side_to_move,
            time_control_raw,
            tc_category,
            eco,
            opening,
            mover_rating,
            opponent_rating,
            rating_difference,
            mover_rating_band,
            player_disjoint,
            is_holdout,
            policy_target_action_index,
            value_target_wdl,
        ) = row

        actual_split = str(temporal_split)
        if actual_split != split:
            raise RuntimeError(
                "Unexpected temporal_split row while loading split data: "
                f"expected={split!r}, got={actual_split!r}"
            )

        example = PositionExample(
            split=split,
            game_id=str(game_id),
            ply=int(ply),
            source_month=(None if source_month is None else str(source_month)),
            position_id=str(position_id),
            pre_move_fen=str(pre_move_fen),
            normalized_pre_move_fen=str(normalized_pre_move_fen),
            side_to_move=str(side_to_move),
            time_control_raw=(None if time_control_raw is None else str(time_control_raw)),
            time_control_category=str(tc_category),
            eco=(None if eco is None else str(eco)),
            opening=(None if opening is None else str(opening)),
            mover_rating=(None if mover_rating is None else int(mover_rating)),
            opponent_rating=(None if opponent_rating is None else int(opponent_rating)),
            rating_difference=(None if rating_difference is None else int(rating_difference)),
            mover_rating_band=str(mover_rating_band),
            player_disjoint_training_eligible=bool(player_disjoint),
            is_player_holdout_game=bool(is_holdout),
            policy_target_action_index=int(policy_target_action_index),
            value_target_wdl=str(value_target_wdl),
            novel_position_test=str(position_id) in novel_position_ids,
        )
        examples.append(example)

    return examples


def _material_points(piece_type: int) -> int:
    if piece_type == chess.PAWN:
        return 1
    if piece_type == chess.KNIGHT:
        return 3
    if piece_type == chess.BISHOP:
        return 3
    if piece_type == chess.ROOK:
        return 5
    if piece_type == chess.QUEEN:
        return 9
    return 0


def _count_piece(board: chess.Board, *, piece_type: int, color: chess.Color) -> int:
    return len(board.pieces(piece_type, color))


def _pawn_structure_counts(board: chess.Board, color: chess.Color, prefix: str) -> dict[str, int]:
    pawns = list(board.pieces(chess.PAWN, color))
    files: list[int] = [chess.square_file(square) for square in pawns]
    file_counts: Counter[int] = Counter(files)

    doubled = sum(count - 1 for count in file_counts.values() if count > 1)

    isolated = 0
    for file_idx in files:
        has_left = (file_idx - 1) in file_counts
        has_right = (file_idx + 1) in file_counts
        if not has_left and not has_right:
            isolated += 1

    islands = 0
    for file_idx in sorted(file_counts):
        if (file_idx - 1) not in file_counts:
            islands += 1

    return {
        f"{prefix}_pawn_count": len(pawns),
        f"{prefix}_doubled_pawns": doubled,
        f"{prefix}_isolated_pawns": isolated,
        f"{prefix}_pawn_islands": islands,
    }


def _game_phase(board: chess.Board, ply: int) -> str:
    non_pawn_non_king = 0
    for piece in board.piece_map().values():
        if piece.piece_type in {chess.PAWN, chess.KING}:
            continue
        non_pawn_non_king += 1

    if non_pawn_non_king >= 12 and ply < 40:
        return "opening"
    if non_pawn_non_king >= 6:
        return "middlegame"
    return "endgame"


def _opening_family(eco: str | None) -> str:
    if eco is None:
        return "unknown"
    text = eco.strip().upper()
    if not text:
        return "unknown"
    return text[0]


def _position_context_features(example: PositionExample, board: chess.Board) -> dict[str, Any]:
    white_material = 0
    black_material = 0
    for piece in board.piece_map().values():
        points = _material_points(piece.piece_type)
        if piece.color == chess.WHITE:
            white_material += points
        else:
            black_material += points

    mover_rating_band = example.mover_rating_band
    if not mover_rating_band:
        mover_rating_band = rating_band(example.mover_rating)

    raw_tc_category = example.time_control_category
    if not raw_tc_category or raw_tc_category == "unknown":
        raw_tc_category = time_control_category(example.time_control_raw)

    context: dict[str, Any] = {
        "side_to_move": example.side_to_move,
        "ply": example.ply,
        "legal_move_count": board.legal_moves.count(),
        "side_in_check": int(board.is_check()),
        "white_castle_k": int(board.has_kingside_castling_rights(chess.WHITE)),
        "white_castle_q": int(board.has_queenside_castling_rights(chess.WHITE)),
        "black_castle_k": int(board.has_kingside_castling_rights(chess.BLACK)),
        "black_castle_q": int(board.has_queenside_castling_rights(chess.BLACK)),
        "white_material": white_material,
        "black_material": black_material,
        "material_balance": white_material - black_material,
        "phase": _game_phase(board, example.ply),
        "mover_rating": (example.mover_rating if example.mover_rating is not None else -1),
        "opponent_rating": (
            example.opponent_rating if example.opponent_rating is not None else -1
        ),
        "rating_difference": (
            example.rating_difference if example.rating_difference is not None else 0
        ),
        "mover_rating_band": mover_rating_band,
        "time_control_category": raw_tc_category,
        "eco": (example.eco if example.eco is not None else "unknown"),
        "opening_family": _opening_family(example.eco),
    }
    context.update(_pawn_structure_counts(board, chess.WHITE, "white"))
    context.update(_pawn_structure_counts(board, chess.BLACK, "black"))

    return context


def _candidate_features(board: chess.Board, move: chess.Move, action_index: int) -> dict[str, Any]:
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    delta_file = to_file - from_file
    delta_rank = to_rank - from_rank

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        raise RuntimeError("Missing moving piece for legal candidate move")

    is_capture = board.is_capture(move)
    captured_piece_type = 0
    if is_capture:
        if board.is_en_passant(move):
            captured_piece_type = int(chess.PAWN)
        else:
            captured_piece = board.piece_at(move.to_square)
            captured_piece_type = int(captured_piece.piece_type) if captured_piece else 0

    return {
        "action_index": action_index,
        "from_square": move.from_square,
        "to_square": move.to_square,
        "moving_piece_type": int(moving_piece.piece_type),
        "is_capture": int(is_capture),
        "captured_piece_type": captured_piece_type,
        "is_promotion": int(move.promotion is not None),
        "promotion_piece_type": int(move.promotion) if move.promotion is not None else 0,
        "is_castling": int(board.is_castling(move)),
        "is_en_passant": int(board.is_en_passant(move)),
        "gives_check": int(board.gives_check(move)),
        "delta_file": delta_file,
        "delta_rank": delta_rank,
        "distance_chebyshev": max(abs(delta_file), abs(delta_rank)),
        "distance_manhattan": abs(delta_file) + abs(delta_rank),
    }


def _assert_no_forbidden_features(feature_names: set[str]) -> None:
    overlap = sorted(feature_names & set(FORBIDDEN_FEATURE_COLUMNS))
    if overlap:
        raise RuntimeError("Forbidden leakage features detected: " + ", ".join(overlap))


@dataclass(frozen=True)
class FrequencyPolicyModel:
    level1_counts: dict[tuple[str, str, int], int]
    level2_counts: dict[tuple[str, int], int]
    global_counts: dict[int, int]


@dataclass(frozen=True)
class FrequencyScore:
    level: int
    count: int
    score: float


def _fit_frequency_model(train_examples: list[PositionExample]) -> FrequencyPolicyModel:
    level1: dict[tuple[str, str, int], int] = defaultdict(int)
    level2: dict[tuple[str, int], int] = defaultdict(int)
    global_counts: dict[int, int] = defaultdict(int)

    for example in train_examples:
        board = chess.Board(example.pre_move_fen)
        phase = _game_phase(board, example.ply)
        rb = example.mover_rating_band or rating_band(example.mover_rating)
        action = example.policy_target_action_index

        level1[(rb, phase, action)] += 1
        level2[(phase, action)] += 1
        global_counts[action] += 1

    return FrequencyPolicyModel(
        level1_counts=dict(level1),
        level2_counts=dict(level2),
        global_counts=dict(global_counts),
    )


def _frequency_score(
    model: FrequencyPolicyModel,
    *,
    rating_band_value: str,
    phase: str,
    action: int,
) -> FrequencyScore:
    count1 = model.level1_counts.get((rating_band_value, phase, action), 0)
    if count1 > 0:
        return FrequencyScore(level=1, count=count1, score=3_000_000.0 + float(count1))

    count2 = model.level2_counts.get((phase, action), 0)
    if count2 > 0:
        return FrequencyScore(level=2, count=count2, score=2_000_000.0 + float(count2))

    global_count = model.global_counts.get(action, 0)
    return FrequencyScore(level=3, count=global_count, score=1_000_000.0 + float(global_count))


def _sample_negative_actions(
    *,
    group_id: str,
    negatives: list[int],
    max_negatives: int,
    seed: int,
) -> list[int]:
    scored = sorted(
        (
            (_stable_hash_u64(f"{seed}|{group_id}|{action}"), action)
            for action in negatives
        ),
        key=lambda item: (item[0], item[1]),
    )
    return [action for _, action in scored[:max_negatives]]


def _build_candidate_groups(
    examples: list[PositionExample],
    *,
    split: str,
    include_all_candidates: bool,
    max_negative_candidates_per_train_position: int | None,
    seed: int,
) -> list[CandidateGroup]:
    groups: list[CandidateGroup] = []

    for example in examples:
        board = chess.Board(example.pre_move_fen)
        context_features = _position_context_features(example, board)

        legal_pairs = sorted(
            ((encode_move_to_index(move, board), move) for move in board.legal_moves),
            key=lambda item: item[0],
        )
        legal_actions = [action for action, _ in legal_pairs]

        positive_matches = [
            action
            for action in legal_actions
            if action == example.policy_target_action_index
        ]
        if len(positive_matches) != 1:
            raise RuntimeError(
                "Expected exactly one positive legal candidate per group: "
                f"split={split}, game_id={example.game_id}, ply={example.ply}, "
                f"target={example.policy_target_action_index}"
            )

        group_id = f"{example.game_id}:{example.ply}"
        negatives = [
            action
            for action in legal_actions
            if action != example.policy_target_action_index
        ]

        selected_actions: set[int]
        if split == "train" and max_negative_candidates_per_train_position is not None:
            sampled = _sample_negative_actions(
                group_id=group_id,
                negatives=negatives,
                max_negatives=max_negative_candidates_per_train_position,
                seed=seed,
            )
            selected_actions = set(sampled)
            selected_actions.add(example.policy_target_action_index)
        elif include_all_candidates:
            selected_actions = set(legal_actions)
        else:
            selected_actions = set(negatives)
            selected_actions.add(example.policy_target_action_index)

        action_indices: list[int] = []
        labels: list[int] = []
        feature_dicts: list[dict[str, Any]] = []

        for action, move in legal_pairs:
            if action not in selected_actions:
                continue
            candidate = dict(context_features)
            candidate.update(_candidate_features(board, move, action))
            _assert_no_forbidden_features(set(candidate.keys()))

            action_indices.append(action)
            labels.append(1 if action == example.policy_target_action_index else 0)
            feature_dicts.append(candidate)

        if sum(labels) != 1:
            raise RuntimeError(
                "Candidate group does not contain exactly one positive label after sampling: "
                f"group_id={group_id!r}"
            )

        groups.append(
            CandidateGroup(
                split=split,
                game_id=example.game_id,
                group_id=group_id,
                action_indices=tuple(action_indices),
                labels=tuple(labels),
                feature_dicts=tuple(feature_dicts),
                metadata={
                    "split": split,
                    "game_id": example.game_id,
                    "ply": example.ply,
                    "position_id": example.position_id,
                    "normalized_pre_move_fen": example.normalized_pre_move_fen,
                    "is_player_holdout_game": example.is_player_holdout_game,
                    "player_disjoint_training_eligible": (
                        example.player_disjoint_training_eligible
                    ),
                    "mover_rating_band": example.mover_rating_band,
                    "time_control_category": example.time_control_category,
                    "game_phase": context_features["phase"],
                    "opening_family": _opening_family(example.eco),
                    "novel_position_test": example.novel_position_test,
                    "source_month": example.source_month,
                },
            )
        )

    return groups


def _flatten_candidate_groups(
    groups: list[CandidateGroup],
) -> tuple[list[dict[str, Any]], np.ndarray, list[int]]:
    features: list[dict[str, Any]] = []
    labels: list[int] = []
    group_sizes: list[int] = []

    for group in groups:
        group_sizes.append(len(group.feature_dicts))
        for feature_dict, label in zip(group.feature_dicts, group.labels, strict=True):
            features.append(feature_dict)
            labels.append(int(label))

    return features, np.asarray(labels, dtype=np.int32), group_sizes


def _ranking_groups_from_scores(
    groups: list[CandidateGroup],
    scores: np.ndarray,
) -> list[RankingGroup]:
    output: list[RankingGroup] = []
    offset = 0

    for group in groups:
        group_size = len(group.labels)
        group_scores = tuple(float(item) for item in scores[offset : offset + group_size])
        offset += group_size

        output.append(
            RankingGroup(
                game_id=group.game_id,
                group_id=group.group_id,
                action_indices=group.action_indices,
                labels=group.labels,
                scores=group_scores,
                metadata=group.metadata,
            )
        )

    if offset != len(scores):
        raise RuntimeError("Ranking score vector length does not match grouped candidate rows")

    return output


def _evaluate_frequency(
    model: FrequencyPolicyModel,
    groups: list[CandidateGroup],
) -> list[RankingGroup]:
    output: list[RankingGroup] = []
    for group in groups:
        rb = str(group.metadata.get("mover_rating_band", "unknown"))
        phase = str(group.metadata.get("game_phase", "unknown"))
        scores: list[float] = []
        for action in group.action_indices:
            score = _frequency_score(model, rating_band_value=rb, phase=phase, action=action)
            scores.append(score.score)

        output.append(
            RankingGroup(
                game_id=group.game_id,
                group_id=group.group_id,
                action_indices=group.action_indices,
                labels=group.labels,
                scores=tuple(scores),
                metadata=group.metadata,
            )
        )

    return output


def _position_feature_dict(example: PositionExample) -> dict[str, Any]:
    board = chess.Board(example.pre_move_fen)
    features = _position_context_features(example, board)
    _assert_no_forbidden_features(set(features.keys()))
    return features


def _validate_wdl_labels(labels: list[str]) -> None:
    unknown = sorted(set(labels) - set(WDL_CLASS_ORDER))
    if unknown:
        raise RuntimeError("Unexpected W/D/L labels encountered: " + ", ".join(unknown))


def _fit_logistic_pipeline(
    train_examples: list[PositionExample],
    *,
    seed: int,
    c: float,
    max_iter: int,
    solver: str,
) -> Pipeline:
    train_features = [_position_feature_dict(item) for item in train_examples]
    train_labels = [item.value_target_wdl for item in train_examples]

    _validate_wdl_labels(train_labels)
    missing_classes = sorted(set(WDL_CLASS_ORDER) - set(train_labels))
    if missing_classes:
        raise RuntimeError(
            "Training split does not cover all W/D/L classes required for multinomial fit: "
            + ", ".join(missing_classes)
        )

    logistic_kwargs: dict[str, Any] = {
        "random_state": seed,
        "solver": solver,
        "C": c,
        "max_iter": max_iter,
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        logistic_kwargs["multi_class"] = "multinomial"

    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "model",
                LogisticRegression(**logistic_kwargs),
            ),
        ]
    )
    pipeline.fit(train_features, train_labels)

    classes = tuple(str(item) for item in pipeline.named_steps["model"].classes_)
    if set(classes) != set(WDL_CLASS_ORDER):
        raise RuntimeError("Logistic classifier classes do not match expected W/D/L labels")

    return pipeline


def _logistic_probabilities(
    pipeline: Pipeline,
    examples: list[PositionExample],
) -> np.ndarray:
    features = [_position_feature_dict(item) for item in examples]
    probs_raw = pipeline.predict_proba(features)

    raw_classes = tuple(str(item) for item in pipeline.named_steps["model"].classes_)
    index_by_label = {label: idx for idx, label in enumerate(raw_classes)}

    probs = np.zeros((probs_raw.shape[0], len(WDL_CLASS_ORDER)), dtype=np.float64)
    for target_idx, label in enumerate(WDL_CLASS_ORDER):
        if label not in index_by_label:
            raise RuntimeError(f"Missing class probability for label {label!r}")
        probs[:, target_idx] = probs_raw[:, index_by_label[label]]

    row_sums = np.sum(probs, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise RuntimeError("Logistic probabilities do not sum to one")

    return probs


def _wdl_indices(labels: list[str]) -> np.ndarray:
    index_by_label = {label: idx for idx, label in enumerate(WDL_CLASS_ORDER)}
    values: list[int] = []
    for label in labels:
        if label not in index_by_label:
            raise RuntimeError(f"Unexpected W/D/L label {label!r}")
        values.append(index_by_label[label])
    return np.asarray(values, dtype=np.int32)


def _logistic_metrics(
    *,
    examples: list[PositionExample],
    probs: np.ndarray,
    ece_bins: int,
) -> dict[str, Any]:
    y_true_labels = [item.value_target_wdl for item in examples]
    y_true = _wdl_indices(y_true_labels)
    y_pred = np.argmax(probs, axis=1)

    ece, ece_bins_payload = expected_calibration_error(y_true, probs, bins=ece_bins)
    precision, recall, _, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(WDL_CLASS_ORDER))),
        zero_division=0,
    )

    return {
        "class_order": list(WDL_CLASS_ORDER),
        "row_count": len(examples),
        "log_loss": float(log_loss(y_true, probs, labels=list(range(len(WDL_CLASS_ORDER))))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "multiclass_brier_score": float(
            multiclass_brier_score(y_true, probs, class_count=len(WDL_CLASS_ORDER))
        ),
        "ece": float(ece),
        "ece_bins": ece_bins_payload,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(WDL_CLASS_ORDER))),
        ).tolist(),
        "per_class_precision_recall": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(WDL_CLASS_ORDER)
        },
        "reliability_bins": reliability_bins_per_class(
            y_true,
            probs,
            class_labels=WDL_CLASS_ORDER,
            bins=ece_bins,
        ),
    }


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    lower = float(np.percentile(values, 2.5))
    upper = float(np.percentile(values, 97.5))
    return lower, upper


def _ranking_bootstrap_metrics(
    groups: list[RankingGroup],
    *,
    cutoffs: tuple[int, ...],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    point = compute_ranking_metrics(groups, cutoffs)
    if iterations <= 0:
        return {
            "iterations": 0,
            "metrics": {
                "recall_at_1": {
                    "point_estimate": float(point.get("recall_at_1", 0.0)),
                    "lower_95": float(point.get("recall_at_1", 0.0)),
                    "upper_95": float(point.get("recall_at_1", 0.0)),
                },
                "mrr": {
                    "point_estimate": float(point.get("mrr", 0.0)),
                    "lower_95": float(point.get("mrr", 0.0)),
                    "upper_95": float(point.get("mrr", 0.0)),
                },
            },
        }

    by_game: dict[str, list[RankingGroup]] = defaultdict(list)
    for group in groups:
        by_game[group.game_id].append(group)

    game_ids = sorted(by_game)
    if not game_ids:
        raise RuntimeError("Cannot bootstrap ranking metrics without game IDs")

    rng = np.random.default_rng(seed)
    recall_values: list[float] = []
    mrr_values: list[float] = []

    for _ in range(iterations):
        sampled_ids = rng.choice(game_ids, size=len(game_ids), replace=True)
        sampled_groups: list[RankingGroup] = []
        for game_id in sampled_ids:
            sampled_groups.extend(by_game[str(game_id)])
        metrics = compute_ranking_metrics(sampled_groups, cutoffs)
        recall_values.append(float(metrics.get("recall_at_1", 0.0)))
        mrr_values.append(float(metrics.get("mrr", 0.0)))

    recall_low, recall_high = _bootstrap_ci(recall_values)
    mrr_low, mrr_high = _bootstrap_ci(mrr_values)

    return {
        "iterations": iterations,
        "metrics": {
            "recall_at_1": {
                "point_estimate": float(point.get("recall_at_1", 0.0)),
                "lower_95": recall_low,
                "upper_95": recall_high,
            },
            "mrr": {
                "point_estimate": float(point.get("mrr", 0.0)),
                "lower_95": mrr_low,
                "upper_95": mrr_high,
            },
        },
    }


def _logistic_bootstrap_metrics(
    examples: list[PositionExample],
    probs: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    labels = [item.value_target_wdl for item in examples]
    y_true = _wdl_indices(labels)
    point_log_loss = float(log_loss(y_true, probs, labels=list(range(len(WDL_CLASS_ORDER)))))

    if iterations <= 0:
        return {
            "iterations": 0,
            "metrics": {
                "log_loss": {
                    "point_estimate": point_log_loss,
                    "lower_95": point_log_loss,
                    "upper_95": point_log_loss,
                }
            },
        }

    game_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        game_to_indices[example.game_id].append(idx)

    game_ids = sorted(game_to_indices)
    if not game_ids:
        raise RuntimeError("Cannot bootstrap logistic metrics without game IDs")

    rng = np.random.default_rng(seed)
    values: list[float] = []

    for _ in range(iterations):
        sampled_game_ids = rng.choice(game_ids, size=len(game_ids), replace=True)
        sampled_indices: list[int] = []
        for game_id in sampled_game_ids:
            sampled_indices.extend(game_to_indices[str(game_id)])

        sampled_y = y_true[np.asarray(sampled_indices, dtype=np.int32)]
        sampled_probs = probs[np.asarray(sampled_indices, dtype=np.int32)]
        value = float(log_loss(sampled_y, sampled_probs, labels=list(range(len(WDL_CLASS_ORDER)))))
        values.append(value)

    lower, upper = _bootstrap_ci(values)
    return {
        "iterations": iterations,
        "metrics": {
            "log_loss": {
                "point_estimate": point_log_loss,
                "lower_95": lower,
                "upper_95": upper,
            }
        },
    }


def _subgroup_payload(groups: list[RankingGroup], cutoffs: tuple[int, ...]) -> dict[str, Any]:
    if not groups:
        return {"available": False, "reason": "no examples in subgroup"}
    return {"available": True, "metrics": compute_ranking_metrics(groups, cutoffs)}


def _ranking_subgroups(groups: list[RankingGroup], cutoffs: tuple[int, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    payload["player_holdout_true"] = _subgroup_payload(
        [item for item in groups if bool(item.metadata.get("is_player_holdout_game"))],
        cutoffs,
    )
    payload["novel_position_test_true"] = _subgroup_payload(
        [item for item in groups if bool(item.metadata.get("novel_position_test"))],
        cutoffs,
    )

    rating_bands = sorted(
        {str(item.metadata.get("mover_rating_band", "unknown")) for item in groups}
    )
    payload["rating_band"] = {
        key: _subgroup_payload(
            [
                item
                for item in groups
                if str(item.metadata.get("mover_rating_band", "unknown")) == key
            ],
            cutoffs,
        )
        for key in rating_bands
    }

    time_controls = sorted(
        {str(item.metadata.get("time_control_category", "unknown")) for item in groups}
    )
    payload["time_control_category"] = {
        key: _subgroup_payload(
            [
                item
                for item in groups
                if str(item.metadata.get("time_control_category", "unknown")) == key
            ],
            cutoffs,
        )
        for key in time_controls
    }

    opening_families = sorted(
        {str(item.metadata.get("opening_family", "unknown")) for item in groups}
    )
    payload["opening_family"] = {
        key: _subgroup_payload(
            [item for item in groups if str(item.metadata.get("opening_family", "unknown")) == key],
            cutoffs,
        )
        for key in opening_families
    }

    phases = sorted({str(item.metadata.get("game_phase", "unknown")) for item in groups})
    payload["game_phase"] = {
        key: _subgroup_payload(
            [item for item in groups if str(item.metadata.get("game_phase", "unknown")) == key],
            cutoffs,
        )
        for key in phases
    }

    return payload


def _logistic_subgroups(
    examples: list[PositionExample],
    probs: np.ndarray,
    *,
    ece_bins: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    def subset_metrics(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"available": False, "reason": "no examples in subgroup"}
        subset_examples = [examples[idx] for idx in indices]
        subset_probs = probs[np.asarray(indices, dtype=np.int32)]
        return {
            "available": True,
            "metrics": _logistic_metrics(
                examples=subset_examples,
                probs=subset_probs,
                ece_bins=ece_bins,
            ),
        }

    payload["player_holdout_true"] = subset_metrics(
        [idx for idx, ex in enumerate(examples) if ex.is_player_holdout_game]
    )
    payload["novel_position_test_true"] = subset_metrics(
        [idx for idx, ex in enumerate(examples) if ex.novel_position_test]
    )

    for field_name in ("mover_rating_band", "time_control_category"):
        values = sorted({getattr(ex, field_name) for ex in examples})
        payload[field_name] = {
            str(value): subset_metrics(
                [idx for idx, ex in enumerate(examples) if getattr(ex, field_name) == value]
            )
            for value in values
        }

    opening_families = sorted({_opening_family(ex.eco) for ex in examples})
    payload["opening_family"] = {
        family: subset_metrics(
            [idx for idx, ex in enumerate(examples) if _opening_family(ex.eco) == family]
        )
        for family in opening_families
    }

    phases = sorted({_game_phase(chess.Board(ex.pre_move_fen), ex.ply) for ex in examples})
    payload["game_phase"] = {
        phase: subset_metrics(
            [
                idx
                for idx, ex in enumerate(examples)
                if _game_phase(chess.Board(ex.pre_move_fen), ex.ply) == phase
            ]
        )
        for phase in phases
    }

    return payload


def _prediction_samples(
    *,
    ranking_groups: list[RankingGroup],
    logistic_examples: list[PositionExample],
    logistic_probs: np.ndarray,
    row_limit: int,
) -> pa.Table:
    rows: list[dict[str, Any]] = []

    for group in ranking_groups:
        order = sorted(
            range(len(group.scores)),
            key=lambda idx: (-group.scores[idx], group.action_indices[idx]),
        )
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "model": "ranking_lightgbm",
                    "split": str(group.metadata.get("split", "unknown")),
                    "game_id": group.game_id,
                    "group_id": group.group_id,
                    "action_index": int(group.action_indices[idx]),
                    "score": float(group.scores[idx]),
                    "is_positive": int(group.labels[idx]),
                    "rank": rank,
                    "wdl_true": None,
                    "wdl_prob_win": None,
                    "wdl_prob_draw": None,
                    "wdl_prob_loss": None,
                }
            )
            if len(rows) >= row_limit:
                return pa.Table.from_pylist(rows)

    for idx, example in enumerate(logistic_examples):
        rows.append(
            {
                "model": "value_logistic",
                "split": example.split,
                "game_id": example.game_id,
                "group_id": f"{example.game_id}:{example.ply}",
                "action_index": None,
                "score": None,
                "is_positive": None,
                "rank": None,
                "wdl_true": example.value_target_wdl,
                "wdl_prob_win": float(logistic_probs[idx, 0]),
                "wdl_prob_draw": float(logistic_probs[idx, 1]),
                "wdl_prob_loss": float(logistic_probs[idx, 2]),
            }
        )
        if len(rows) >= row_limit:
            break

    return pa.Table.from_pylist(rows)


def _experiment_identity_payload(
    *,
    config: BaselineConfig,
    preflight: PreflightSummary,
    git_commit: str | None,
) -> dict[str, Any]:
    config_payload = baseline_config_identity_payload(config)
    config_payload["input"].pop("modeling_manifest_path", None)

    return {
        "modeling_dataset_id": preflight.modeling_dataset_id,
        "modeling_manifest_sha256": preflight.modeling_manifest_sha256,
        "feature_schema_version": preflight.feature_schema_version,
        "split_definition_version": preflight.split_definition_version,
        "baseline_pipeline_version": config.versions.baseline_pipeline_version,
        "rating_band_definition_version": config.versions.rating_band_definition_version,
        "game_phase_definition_version": config.versions.game_phase_definition_version,
        "effective_config": config_payload,
        "git_commit": git_commit,
    }


def _validate_existing_experiment(
    *,
    experiment_path: Path,
    expected_experiment_id: str,
    expected_identity_hash: str,
) -> dict[str, Any]:
    manifest = read_json_object(experiment_path / "_manifest.json")
    validate_manifest_identity(
        manifest=manifest,
        expected_paths={
            ("status",): "complete",
            ("experiment_id",): expected_experiment_id,
            ("identity_payload_sha256",): expected_identity_hash,
        },
    )

    if not (experiment_path / "_SUCCESS").exists():
        raise RuntimeError("Existing experiment missing _SUCCESS marker")

    required_files = [
        "effective_config.json",
        "aggregate_metrics.json",
        "subgroup_metrics.json",
        "bootstrap_confidence_intervals.json",
        "lightgbm_model.txt",
        "logistic_pipeline.pkl",
        "frequency_model.json",
        "leakage_audit.json",
        "leakage_audit.md",
        "baseline_report.md",
    ]

    missing = [name for name in required_files if not (experiment_path / name).exists()]
    if missing:
        raise RuntimeError(
            "Existing experiment is incomplete; missing artifacts: " + ", ".join(missing)
        )

    return manifest


def _result_json(result: BaselineRunResult) -> dict[str, Any]:
    payload = {
        "run_id": result.run_id,
        "experiment_id": result.experiment_id,
        "reused_existing": result.reused_existing,
        "output_path": _display_path(result.output_path),
        "manifest_path": _display_path(result.manifest_path),
        "duration_seconds": result.duration_seconds,
        "peak_rss_bytes": result.peak_rss_bytes,
        "dry_run": result.dry_run,
        "validate_only": result.validate_only,
    }
    if result.summary is not None:
        payload["summary"] = result.summary
    return payload


def run_baselines(
    *,
    config_path: Path,
    modeling_manifest_override: str | None = None,
    output_root_override: str | None = None,
    max_train_positions_override: int | None = None,
    max_validation_positions_override: int | None = None,
    max_test_positions_override: int | None = None,
    threads_override: int | None = None,
    seed_override: int | None = None,
    dry_run: bool = False,
    validate_only: bool = False,
) -> BaselineRunResult:
    config = load_baseline_config(config_path)
    config = apply_baseline_cli_overrides(
        config,
        modeling_manifest_path=modeling_manifest_override,
        output_root=output_root_override,
        max_train_positions=max_train_positions_override,
        max_validation_positions=max_validation_positions_override,
        max_test_positions=max_test_positions_override,
        threads=threads_override,
        seed=seed_override,
    )

    preflight = _preflight_modeling_dataset(config)
    novel_position_ids = _novel_position_ids(preflight.novel_position_paths)

    train_examples = _load_position_examples(
        preflight.policy_paths_by_split["train"],
        split="train",
        limit=config.limits.max_train_positions,
        seed=config.runtime.seed,
        novel_position_ids=novel_position_ids,
    )
    validation_examples = _load_position_examples(
        preflight.policy_paths_by_split["validation"],
        split="validation",
        limit=config.limits.max_validation_positions,
        seed=config.runtime.seed,
        novel_position_ids=novel_position_ids,
    )
    test_examples = _load_position_examples(
        preflight.policy_paths_by_split["test"],
        split="test",
        limit=config.limits.max_test_positions,
        seed=config.runtime.seed,
        novel_position_ids=novel_position_ids,
    )

    if not train_examples:
        raise RuntimeError("No train examples selected for baseline training")
    if not validation_examples:
        raise RuntimeError("No validation examples selected for baseline evaluation")
    if not test_examples:
        raise RuntimeError("No test examples selected for baseline evaluation")

    base_output_root = config.output.output_root
    git_commit = _git_commit()

    identity_payload = _experiment_identity_payload(
        config=config,
        preflight=preflight,
        git_commit=git_commit,
    )
    identity_hash = sha256_text(canonical_json(identity_payload))
    experiment_id = identity_hash

    run_id = f"baseline-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"

    experiment_root = base_output_root / "experiments" / experiment_id
    manifest_path = experiment_root / "_manifest.json"

    if experiment_root.exists():
        _validate_existing_experiment(
            experiment_path=experiment_root,
            expected_experiment_id=experiment_id,
            expected_identity_hash=identity_hash,
        )
        return BaselineRunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            reused_existing=True,
            output_path=experiment_root,
            manifest_path=manifest_path,
            duration_seconds=None,
            peak_rss_bytes=None,
            dry_run=False,
            validate_only=validate_only,
            summary=None,
        )

    train_group_estimate = sum(
        chess.Board(item.pre_move_fen).legal_moves.count()
        for item in train_examples
    )
    validation_group_estimate = sum(
        chess.Board(item.pre_move_fen).legal_moves.count() for item in validation_examples
    )
    test_group_estimate = sum(
        chess.Board(item.pre_move_fen).legal_moves.count()
        for item in test_examples
    )

    if dry_run:
        summary_payload = {
            "run_id": run_id,
            "dry_run": True,
            "validate_only": False,
            "modeling_manifest_path": _display_path(preflight.modeling_manifest_path),
            "modeling_dataset_id": preflight.modeling_dataset_id,
            "split_row_counts_full": preflight.split_row_counts,
            "split_row_counts_selected": {
                "train": len(train_examples),
                "validation": len(validation_examples),
                "test": len(test_examples),
            },
            "estimated_candidate_rows": {
                "train": train_group_estimate,
                "validation": validation_group_estimate,
                "test": test_group_estimate,
            },
            "selected_limits": {
                "max_train_positions": config.limits.max_train_positions,
                "max_validation_positions": config.limits.max_validation_positions,
                "max_test_positions": config.limits.max_test_positions,
                "max_negative_candidates_per_train_position": (
                    config.limits.max_negative_candidates_per_train_position
                ),
            },
            "estimated_memory_note": (
                "Candidate rows scale with legal move count per position; "
                "reduce max_*_positions if memory is constrained."
            ),
            "estimated_disk_note": (
                "Prediction artifact rows are capped by output.prediction_artifact_row_limit."
            ),
            "output_path": _display_path(experiment_root),
        }
        return BaselineRunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            reused_existing=False,
            output_path=experiment_root,
            manifest_path=manifest_path,
            duration_seconds=None,
            peak_rss_bytes=None,
            dry_run=True,
            validate_only=False,
            summary=summary_payload,
        )

    leakage_audit = {
        "game_id_overlap_counts": preflight.game_id_overlap_counts,
        "normalized_fen_overlap_counts": preflight.normalized_fen_overlap_counts,
        "player_overlap_counts": preflight.player_overlap_counts,
        "feature_allowlist": sorted(FEATURE_DEFINITIONS),
        "forbidden_feature_registry": list(FORBIDDEN_FEATURE_COLUMNS),
        "confirmations": {
            "frequency_counts_use_train_only": True,
            "preprocessing_fit_on_train_only": True,
            "lightgbm_early_stopping_uses_validation_only": True,
            "test_not_used_for_fitting_or_selection": True,
        },
        "duplicate_counts": {
            "train_position_duplicates": (
                len(train_examples)
                - len({(item.game_id, item.ply) for item in train_examples})
            ),
            "validation_position_duplicates": (
                len(validation_examples)
                - len({(item.game_id, item.ply) for item in validation_examples})
            ),
            "test_position_duplicates": (
                len(test_examples) - len({(item.game_id, item.ply) for item in test_examples})
            ),
        },
        "label_legality": {
            "checked_rows": preflight.label_legality_checked_rows,
            "status": "pass",
        },
    }

    if validate_only:
        summary = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "dry_run": False,
            "validate_only": True,
            "modeling_manifest_path": _display_path(preflight.modeling_manifest_path),
            "modeling_dataset_id": preflight.modeling_dataset_id,
            "split_row_counts_full": preflight.split_row_counts,
            "split_row_counts_selected": {
                "train": len(train_examples),
                "validation": len(validation_examples),
                "test": len(test_examples),
            },
            "leakage_audit": leakage_audit,
            "output_path": _display_path(experiment_root),
        }
        return BaselineRunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            reused_existing=False,
            output_path=experiment_root,
            manifest_path=manifest_path,
            duration_seconds=None,
            peak_rss_bytes=None,
            dry_run=False,
            validate_only=True,
            summary=summary,
        )

    staging_root = base_output_root / "staging" / f"{experiment_id}__{run_id}"
    _safe_rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    sampler = PeakRssSampler(interval_seconds=0.05)
    sampler.start()
    published = False

    try:
        train_groups = _build_candidate_groups(
            train_examples,
            split="train",
            include_all_candidates=False,
            max_negative_candidates_per_train_position=config.limits.max_negative_candidates_per_train_position,
            seed=config.runtime.seed,
        )

        validation_groups = _build_candidate_groups(
            validation_examples,
            split="validation",
            include_all_candidates=config.limits.evaluate_all_legal_candidates_validation,
            max_negative_candidates_per_train_position=None,
            seed=config.runtime.seed,
        )

        test_groups = _build_candidate_groups(
            test_examples,
            split="test",
            include_all_candidates=config.limits.evaluate_all_legal_candidates_test,
            max_negative_candidates_per_train_position=None,
            seed=config.runtime.seed,
        )

        frequency_model = _fit_frequency_model(train_examples)
        frequency_validation_ranked = _evaluate_frequency(frequency_model, validation_groups)
        frequency_test_ranked = _evaluate_frequency(frequency_model, test_groups)

        freq_validation_metrics = compute_ranking_metrics(
            frequency_validation_ranked,
            config.evaluation.ranking_cutoffs,
        )
        freq_test_metrics = compute_ranking_metrics(
            frequency_test_ranked,
            config.evaluation.ranking_cutoffs,
        )

        logistic_pipeline = _fit_logistic_pipeline(
            train_examples,
            seed=config.runtime.seed,
            c=config.logistic.c,
            max_iter=config.logistic.max_iter,
            solver=config.logistic.solver,
        )

        logistic_validation_probs = _logistic_probabilities(logistic_pipeline, validation_examples)
        logistic_test_probs = _logistic_probabilities(logistic_pipeline, test_examples)

        logistic_validation_metrics = _logistic_metrics(
            examples=validation_examples,
            probs=logistic_validation_probs,
            ece_bins=config.evaluation.ece_bins,
        )
        logistic_test_metrics = _logistic_metrics(
            examples=test_examples,
            probs=logistic_test_probs,
            ece_bins=config.evaluation.ece_bins,
        )

        rank_vectorizer = DictVectorizer(sparse=True)
        train_feature_dicts, train_labels, train_group_sizes = _flatten_candidate_groups(
            train_groups
        )
        (
            validation_feature_dicts,
            validation_labels,
            validation_group_sizes,
        ) = _flatten_candidate_groups(validation_groups)
        test_feature_dicts, test_labels, test_group_sizes = _flatten_candidate_groups(test_groups)

        if sum(train_group_sizes) != len(train_feature_dicts):
            raise RuntimeError("Invalid LightGBM train group sizes")
        if sum(validation_group_sizes) != len(validation_feature_dicts):
            raise RuntimeError("Invalid LightGBM validation group sizes")
        if sum(test_group_sizes) != len(test_feature_dicts):
            raise RuntimeError("Invalid LightGBM test group sizes")

        train_matrix = rank_vectorizer.fit_transform(train_feature_dicts)
        validation_matrix = rank_vectorizer.transform(validation_feature_dicts)
        test_matrix = rank_vectorizer.transform(test_feature_dicts)

        lgb_ranker = lgb.LGBMRanker(
            objective="lambdarank",
            random_state=config.runtime.seed,
            n_estimators=config.lightgbm.n_estimators,
            learning_rate=config.lightgbm.learning_rate,
            num_leaves=config.lightgbm.num_leaves,
            min_data_in_leaf=config.lightgbm.min_data_in_leaf,
            feature_fraction=config.lightgbm.feature_fraction,
            lambda_l2=float(config.lightgbm.lambda_l2),
            n_jobs=config.runtime.threads,
        )

        lgb_ranker.fit(
            train_matrix,
            train_labels,
            group=train_group_sizes,
            eval_set=[(validation_matrix, validation_labels)],
            eval_group=[validation_group_sizes],
            eval_at=list(config.evaluation.ranking_cutoffs),
            callbacks=[
                lgb.early_stopping(config.runtime.early_stopping_rounds, verbose=False),
            ],
        )

        lgb_validation_scores = lgb_ranker.predict(validation_matrix)
        lgb_test_scores = lgb_ranker.predict(test_matrix)

        lgb_validation_ranked = _ranking_groups_from_scores(
            validation_groups,
            np.asarray(lgb_validation_scores, dtype=np.float64),
        )
        lgb_test_ranked = _ranking_groups_from_scores(
            test_groups,
            np.asarray(lgb_test_scores, dtype=np.float64),
        )

        lgb_validation_metrics = compute_ranking_metrics(
            lgb_validation_ranked,
            config.evaluation.ranking_cutoffs,
        )
        lgb_test_metrics = compute_ranking_metrics(
            lgb_test_ranked,
            config.evaluation.ranking_cutoffs,
        )

        subgroup_metrics = {
            "ranking_frequency_test": _ranking_subgroups(
                frequency_test_ranked,
                config.evaluation.ranking_cutoffs,
            ),
            "ranking_lightgbm_test": _ranking_subgroups(
                lgb_test_ranked,
                config.evaluation.ranking_cutoffs,
            ),
            "logistic_wdl_test": _logistic_subgroups(
                test_examples,
                logistic_test_probs,
                ece_bins=config.evaluation.ece_bins,
            ),
        }

        bootstrap_payload = {
            "ranking_frequency_test": _ranking_bootstrap_metrics(
                frequency_test_ranked,
                cutoffs=config.evaluation.ranking_cutoffs,
                iterations=config.evaluation.bootstrap_iterations,
                seed=config.runtime.seed,
            ),
            "ranking_lightgbm_test": _ranking_bootstrap_metrics(
                lgb_test_ranked,
                cutoffs=config.evaluation.ranking_cutoffs,
                iterations=config.evaluation.bootstrap_iterations,
                seed=config.runtime.seed + 17,
            ),
            "logistic_wdl_test": _logistic_bootstrap_metrics(
                test_examples,
                logistic_test_probs,
                iterations=config.evaluation.bootstrap_iterations,
                seed=config.runtime.seed + 31,
            ),
        }

        aggregate_metrics = {
            "frequency_policy": {
                "validation": freq_validation_metrics,
                "test": freq_test_metrics,
                "backoff_levels": list(config.frequency.backoff_levels),
            },
            "lightgbm_policy": {
                "validation": lgb_validation_metrics,
                "test": lgb_test_metrics,
                "best_iteration": int(getattr(lgb_ranker, "best_iteration_", 0) or 0),
            },
            "logistic_wdl": {
                "validation": logistic_validation_metrics,
                "test": logistic_test_metrics,
                "class_order": list(WDL_CLASS_ORDER),
            },
        }

        prediction_sample_table = _prediction_samples(
            ranking_groups=lgb_test_ranked,
            logistic_examples=test_examples,
            logistic_probs=logistic_test_probs,
            row_limit=config.output.prediction_artifact_row_limit,
        )

        (staging_root / "effective_config.json").write_text(
            json.dumps(baseline_config_identity_payload(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging_root / "experiment_identity_payload.json").write_text(
            json.dumps(identity_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        environment_payload = {
            "python_version": subprocess.check_output(
                ["python", "-c", "import platform; print(platform.python_version())"],
                text=True,
            ).strip(),
            "git_commit": git_commit,
            "packages": {
                "duckdb": duckdb.__version__,
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "scikit_learn": __import__("sklearn").__version__,
            },
        }
        (staging_root / "environment.json").write_text(
            json.dumps(environment_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (staging_root / "feature_definitions.json").write_text(
            json.dumps(
                {
                    "feature_definitions": FEATURE_DEFINITIONS,
                    "forbidden_feature_registry": list(FORBIDDEN_FEATURE_COLUMNS),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        (staging_root / "frequency_model.json").write_text(
            json.dumps(
                {
                    "level1_counts": [
                        {
                            "rating_band": key[0],
                            "game_phase": key[1],
                            "action_index": key[2],
                            "count": value,
                        }
                        for key, value in sorted(frequency_model.level1_counts.items())
                    ],
                    "level2_counts": [
                        {
                            "game_phase": key[0],
                            "action_index": key[1],
                            "count": value,
                        }
                        for key, value in sorted(frequency_model.level2_counts.items())
                    ],
                    "global_counts": [
                        {"action_index": key, "count": value}
                        for key, value in sorted(frequency_model.global_counts.items())
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        with (staging_root / "logistic_pipeline.pkl").open("wb") as handle:
            pickle.dump(logistic_pipeline, handle)

        lgb_ranker.booster_.save_model((staging_root / "lightgbm_model.txt").as_posix())

        feature_names = rank_vectorizer.get_feature_names_out()
        importances_gain = lgb_ranker.booster_.feature_importance(importance_type="gain")
        importances_split = lgb_ranker.booster_.feature_importance(importance_type="split")
        (staging_root / "lightgbm_feature_importance.json").write_text(
            json.dumps(
                {
                    "features": [
                        {
                            "name": str(name),
                            "gain": float(gain),
                            "split": int(split),
                        }
                        for name, gain, split in sorted(
                            zip(
                                feature_names,
                                importances_gain,
                                importances_split,
                                strict=True,
                            ),
                            key=lambda item: (-float(item[1]), str(item[0])),
                        )
                    ]
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        (staging_root / "aggregate_metrics.json").write_text(
            json.dumps(aggregate_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging_root / "subgroup_metrics.json").write_text(
            json.dumps(subgroup_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging_root / "bootstrap_confidence_intervals.json").write_text(
            json.dumps(bootstrap_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (staging_root / "confusion_matrix.json").write_text(
            json.dumps(
                {
                    "class_order": list(WDL_CLASS_ORDER),
                    "validation": logistic_validation_metrics["confusion_matrix"],
                    "test": logistic_test_metrics["confusion_matrix"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        (staging_root / "calibration_bins.json").write_text(
            json.dumps(
                {
                    "validation": {
                        "ece": logistic_validation_metrics["ece"],
                        "ece_bins": logistic_validation_metrics["ece_bins"],
                        "reliability_bins": logistic_validation_metrics["reliability_bins"],
                    },
                    "test": {
                        "ece": logistic_test_metrics["ece"],
                        "ece_bins": logistic_test_metrics["ece_bins"],
                        "reliability_bins": logistic_test_metrics["reliability_bins"],
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        leakage_audit_md = "\n".join(
            [
                "# Leakage Audit",
                "",
                "## Split overlap checks",
                (
                    "- game_id overlaps: "
                    + json.dumps(preflight.game_id_overlap_counts, sort_keys=True)
                ),
                (
                    "- normalized FEN overlaps: "
                    + json.dumps(preflight.normalized_fen_overlap_counts, sort_keys=True)
                ),
                f"- player overlaps: {json.dumps(preflight.player_overlap_counts, sort_keys=True)}",
                "",
                "## Confirmations",
                "- frequency counts fit on train only: yes",
                "- preprocessing fit on train only: yes",
                "- LightGBM early stopping uses validation only: yes",
                "- test not used for fit/selection: yes",
                "- forbidden features rejected: yes",
                "- policy target legality checked: yes",
            ]
        )
        (staging_root / "leakage_audit.md").write_text(leakage_audit_md + "\n", encoding="utf-8")
        (staging_root / "leakage_audit.json").write_text(
            json.dumps(leakage_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        baseline_report = "\n".join(
            [
                "# Phase 2.2 Classical Baselines Report",
                "",
                f"- modeling_dataset_id: {preflight.modeling_dataset_id}",
                f"- experiment_id: {experiment_id}",
                f"- run_id: {run_id}",
                "",
                "## Policy ranking comparison (validation vs test)",
                (
                    "- frequency Recall@1: "
                    + f"val={freq_validation_metrics.get('recall_at_1', 0.0):.4f}, "
                    + f"test={freq_test_metrics.get('recall_at_1', 0.0):.4f}"
                ),
                (
                    "- lightgbm Recall@1: "
                    + f"val={lgb_validation_metrics.get('recall_at_1', 0.0):.4f}, "
                    + f"test={lgb_test_metrics.get('recall_at_1', 0.0):.4f}"
                ),
                (
                    "- frequency MRR: "
                    + f"val={freq_validation_metrics.get('mrr', 0.0):.4f}, "
                    + f"test={freq_test_metrics.get('mrr', 0.0):.4f}"
                ),
                (
                    "- lightgbm MRR: "
                    + f"val={lgb_validation_metrics.get('mrr', 0.0):.4f}, "
                    + f"test={lgb_test_metrics.get('mrr', 0.0):.4f}"
                ),
                "",
                "## Value baseline (multinomial logistic regression)",
                (
                    "- log loss: "
                    + f"val={logistic_validation_metrics.get('log_loss', 0.0):.4f}, "
                    + f"test={logistic_test_metrics.get('log_loss', 0.0):.4f}"
                ),
                (
                    "- macro-F1: "
                    + f"val={logistic_validation_metrics.get('macro_f1', 0.0):.4f}, "
                    + f"test={logistic_test_metrics.get('macro_f1', 0.0):.4f}"
                ),
                (
                    "- multiclass Brier: "
                    + f"val={logistic_validation_metrics.get('multiclass_brier_score', 0.0):.4f}, "
                    + f"test={logistic_test_metrics.get('multiclass_brier_score', 0.0):.4f}"
                ),
                (
                    "- ECE: "
                    + f"val={logistic_validation_metrics.get('ece', 0.0):.4f}, "
                    + f"test={logistic_test_metrics.get('ece', 0.0):.4f}"
                ),
                "",
                "## Limitations",
                "- Policy ranking target is observed human move, not objective best move.",
                "- No engine-score labels or blunder targets are used in this phase.",
                "- Subgroup coverage depends on available rows in selected dataset limits.",
            ]
        )
        (staging_root / "baseline_report.md").write_text(baseline_report + "\n", encoding="utf-8")

        pq.write_table(prediction_sample_table, staging_root / "prediction_samples.parquet")

        duration_seconds = time.perf_counter() - started
        peak_rss_bytes = sampler.stop()

        artifact_files = sorted(
            p.name
            for p in staging_root.iterdir()
            if p.is_file() and p.name not in {"_manifest.json", "_SUCCESS"}
        )

        manifest = {
            "status": "complete",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "identity_payload_sha256": identity_hash,
            "identity_payload": identity_payload,
            "modeling_input": {
                "manifest_path": _display_path(preflight.modeling_manifest_path),
                "modeling_dataset_id": preflight.modeling_dataset_id,
                "modeling_manifest_sha256": preflight.modeling_manifest_sha256,
                "collection_id": preflight.collection_id,
                "source_months": list(preflight.source_months),
                "feature_schema_version": preflight.feature_schema_version,
                "split_definition_version": preflight.split_definition_version,
            },
            "counts": {
                "selected_positions": {
                    "train": len(train_examples),
                    "validation": len(validation_examples),
                    "test": len(test_examples),
                },
                "candidate_groups": {
                    "train": len(train_groups),
                    "validation": len(validation_groups),
                    "test": len(test_groups),
                },
                "candidate_rows": {
                    "train": int(sum(len(g.labels) for g in train_groups)),
                    "validation": int(sum(len(g.labels) for g in validation_groups)),
                    "test": int(sum(len(g.labels) for g in test_groups)),
                },
            },
            "versions": {
                "baseline_pipeline_version": config.versions.baseline_pipeline_version,
                "feature_schema_version": preflight.feature_schema_version,
                "split_definition_version": preflight.split_definition_version,
                "rating_band_definition_version": config.versions.rating_band_definition_version,
                "game_phase_definition_version": config.versions.game_phase_definition_version,
            },
            "performance": {
                "duration_seconds": round(duration_seconds, 6),
                "peak_rss_bytes": int(peak_rss_bytes),
            },
            "created_at_utc": _utc_now_iso(),
            "git_commit": git_commit,
            "artifacts": artifact_files,
        }

        atomic_write_json(staging_root / "_manifest.json", manifest)
        (staging_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")

        experiment_root.parent.mkdir(parents=True, exist_ok=True)
        if experiment_root.exists():
            raise RuntimeError(
                "Experiment path appeared during execution; refusing overwrite: "
                + experiment_root.as_posix()
            )
        os.replace(staging_root, experiment_root)
        published = True

        return BaselineRunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            reused_existing=False,
            output_path=experiment_root,
            manifest_path=experiment_root / "_manifest.json",
            duration_seconds=round(duration_seconds, 6),
            peak_rss_bytes=int(peak_rss_bytes),
            dry_run=False,
            validate_only=False,
            summary=None,
        )
    finally:
        if not published:
            sampler.stop()
            _safe_rmtree(staging_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2.2 classical baselines")
    parser.add_argument("--config", default="configs/baselines/fixture_smoke.yaml")
    parser.add_argument("--modeling-manifest", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-train-positions", type=int, default=None)
    parser.add_argument("--max-validation-positions", type=int, default=None)
    parser.add_argument("--max-test-positions", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = run_baselines(
        config_path=Path(args.config),
        modeling_manifest_override=args.modeling_manifest,
        output_root_override=args.output_root,
        max_train_positions_override=args.max_train_positions,
        max_validation_positions_override=args.max_validation_positions,
        max_test_positions_override=args.max_test_positions,
        threads_override=args.threads,
        seed_override=args.seed,
        dry_run=bool(args.dry_run),
        validate_only=bool(args.validate_only),
    )
    print(json.dumps(_result_json(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
