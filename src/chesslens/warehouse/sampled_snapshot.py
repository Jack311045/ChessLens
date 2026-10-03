"""Utilities for inspecting and safely relocating sampled DuckDB warehouse snapshots."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

SQL_STRING_LITERAL_PATTERN = re.compile(r"'((?:''|[^'])*)'")
WINDOWS_ABS_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
COLLECTION_SEGMENT_PATTERN = re.compile(r"/processed/collections/([^/]+)/", re.IGNORECASE)
SAMPLED_MARKER_RELATIONS: frozenset[str] = frozenset(
    {
        "sampled_game_keys",
        "sampled_game_indices",
        "sample_counts",
    }
)


@dataclass(frozen=True)
class SampledSnapshotInspection:
    duckdb_path: Path
    relation_types: dict[str, str]
    view_sql_by_name: dict[str, str]
    stale_paths_by_view: dict[str, tuple[str, ...]]
    collection_ids_from_paths: tuple[str, ...]
    deterministic_sample_detected: bool


@dataclass(frozen=True)
class SampledSnapshotRelocationResult:
    source_duckdb: Path
    target_duckdb: Path
    preserved_filename: bool
    replaced_views: tuple[str, ...]
    replaced_collection_roots: tuple[str, ...]
    stg_games_rows: int
    int_move_context_rows: int
    sampled_game_keys_rows: int
    sampled_game_indices_rows: int
    distinct_game_ids: int
    moves_without_game_rows: int
    collection_id: str


class SampledSnapshotError(RuntimeError):
    """Raised when sampled snapshot relocation or validation fails."""


def _normalize_path(path_text: str) -> str:
    return path_text.replace("\\", "/")


def _extract_absolute_paths(sql_text: str) -> tuple[str, ...]:
    absolute_paths: set[str] = set()
    for literal in SQL_STRING_LITERAL_PATTERN.findall(sql_text):
        path_text = literal.replace("''", "'")
        if WINDOWS_ABS_PATH_PATTERN.match(path_text) or path_text.startswith("/"):
            absolute_paths.add(_normalize_path(path_text))
    return tuple(sorted(absolute_paths))


def _extract_collection_id(path_text: str) -> str | None:
    match = COLLECTION_SEGMENT_PATTERN.search(_normalize_path(path_text))
    if match is None:
        return None
    return match.group(1)


def _collection_root_from_path(path_text: str, collection_id: str) -> str | None:
    normalized = _normalize_path(path_text)
    marker = f"/processed/collections/{collection_id}/"
    index = normalized.lower().find(marker.lower())
    if index < 0:
        return None
    return normalized[: index + len(marker) - 1]


def _load_relation_types(connection: duckdb.DuckDBPyConnection) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    ).fetchall()
    return {str(name): str(table_type) for name, table_type in rows}


def _load_view_sql(connection: duckdb.DuckDBPyConnection) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT view_name, sql
        FROM duckdb_views()
        WHERE schema_name = 'main'
        ORDER BY view_name
        """
    ).fetchall()
    return {str(name): str(sql or "") for name, sql in rows}


def _detect_deterministic_sample_snapshot(
    relation_types: dict[str, str],
    view_sql_by_name: dict[str, str],
) -> bool:
    marker_count = len(SAMPLED_MARKER_RELATIONS.intersection(set(relation_types)))
    if marker_count < 2:
        return False

    bronze_games_sql = " ".join(view_sql_by_name.get("bronze_games", "").split()).lower()
    bronze_moves_sql = " ".join(view_sql_by_name.get("bronze_moves", "").split()).lower()

    predicate_markers = (
        "bronze_games_full" in bronze_games_sql
        and "bronze_moves_full" in bronze_moves_sql
        and "hash(game_id)" in bronze_games_sql
        and "hash(game_id)" in bronze_moves_sql
    )
    return predicate_markers


def inspect_sampled_snapshot(
    *,
    duckdb_path: Path,
    expected_collection_root: Path | None = None,
) -> SampledSnapshotInspection:
    if not duckdb_path.exists():
        raise SampledSnapshotError(f"DuckDB snapshot not found: {duckdb_path.as_posix()}")

    expected_root = (
        _normalize_path(expected_collection_root.resolve().as_posix())
        if expected_collection_root is not None
        else None
    )

    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        relation_types = _load_relation_types(connection)
        view_sql_by_name = _load_view_sql(connection)
    finally:
        connection.close()

    stale_paths_by_view: dict[str, tuple[str, ...]] = {}
    collection_ids: set[str] = set()
    for view_name, sql in view_sql_by_name.items():
        absolute_paths = _extract_absolute_paths(sql)
        for path in absolute_paths:
            collection_id = _extract_collection_id(path)
            if collection_id is not None:
                collection_ids.add(collection_id)

        if expected_root is None:
            stale_paths = absolute_paths
        else:
            stale_paths = tuple(
                sorted(
                    path
                    for path in absolute_paths
                    if not path.lower().startswith(expected_root.lower())
                )
            )

        if stale_paths:
            stale_paths_by_view[view_name] = stale_paths

    deterministic_sample_detected = _detect_deterministic_sample_snapshot(
        relation_types,
        view_sql_by_name,
    )

    return SampledSnapshotInspection(
        duckdb_path=duckdb_path,
        relation_types=relation_types,
        view_sql_by_name=view_sql_by_name,
        stale_paths_by_view=stale_paths_by_view,
        collection_ids_from_paths=tuple(sorted(collection_ids)),
        deterministic_sample_detected=deterministic_sample_detected,
    )


def ensure_safe_full_collection_registration_target(duckdb_path: Path) -> None:
    """Fail if full collection registration would mutate a sampled snapshot."""
    if not duckdb_path.exists():
        return

    inspection = inspect_sampled_snapshot(duckdb_path=duckdb_path)
    if inspection.deterministic_sample_detected:
        raise SampledSnapshotError(
            "Refusing to register full collection bronze views into a deterministic_sample "
            "DuckDB snapshot; this would cause sampled/full population drift. "
            "Use --skip-register-views or run the sampled relocation workflow."
        )


def _view_sql_to_replace_statement(sql_text: str) -> str:
    if re.match(r"^\s*CREATE\s+OR\s+REPLACE\s+VIEW\b", sql_text, flags=re.IGNORECASE):
        return sql_text

    rewritten = re.sub(
        r"^\s*CREATE\s+VIEW\b",
        "CREATE OR REPLACE VIEW",
        sql_text,
        count=1,
        flags=re.IGNORECASE,
    )
    if rewritten == sql_text:
        raise SampledSnapshotError("Expected CREATE VIEW statement in persisted view SQL")
    return rewritten


def _load_sampling_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SampledSnapshotError("Sampling evidence must be a JSON object")

    sampling = payload.get("sampling")
    if isinstance(sampling, dict):
        return sampling

    measured = payload.get("measured")
    if isinstance(measured, dict):
        return {
            "parent_collection_id": payload.get("collection_id"),
            "parent_full_counts": {
                "games": measured.get("full_games"),
                "moves": measured.get("full_moves"),
            },
            "sampled_counts": {
                "games": measured.get("sampled_games"),
                "moves": measured.get("sampled_moves"),
            },
        }

    raise SampledSnapshotError(
        "Sampling evidence must include either a 'sampling' object or measured "
        "sampled/full counts"
    )


def _build_relocated_collection_manifest_payload(
    *,
    collection_root: Path,
    collection_id: str,
    expected_sampled_games: int,
    expected_sampled_moves: int,
) -> dict[str, Any]:
    source_manifest_path = collection_root / "_collection_manifest.json"
    if not source_manifest_path.exists():
        raise SampledSnapshotError(
            "Collection manifest not found for relocated bronze manifest synthesis: "
            f"{source_manifest_path.as_posix()}"
        )

    source_payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_payload, dict):
        raise SampledSnapshotError("Collection manifest must be a JSON object")

    raw_timing = source_payload.get("timing")
    timing: dict[str, Any] = raw_timing if isinstance(raw_timing, dict) else {}
    now = datetime.now(tz=UTC).isoformat(timespec="seconds")

    return {
        "dataset_id": collection_id,
        "configuration_hash": source_payload.get("shard_manifest_identity_hash", "collection"),
        "status": "complete",
        "manifest_version": source_payload.get("manifest_version", "1.0.0"),
        "run_id": collection_id,
        "started_at_utc": timing.get("started_at_utc", now),
        "finished_at_utc": timing.get("updated_at_utc", now),
        "source": {
            "source_month": source_payload.get("source_month"),
            "archive_filename": source_payload.get("parent_archive_filename"),
            "archive_sha256": source_payload.get("parent_archive_sha256"),
        },
        "counts": {
            "accepted_games": expected_sampled_games,
            "rejected_games": 0,
            "emitted_moves": expected_sampled_moves,
            "error_records": 0,
        },
        "datasets": {
            "games": {"row_count": expected_sampled_games},
            "moves": {"row_count": expected_sampled_moves},
            "ingestion_errors": {"row_count": 0},
        },
    }


def _as_int(value: Any, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SampledSnapshotError(f"{field_name} must be an integer") from exc
    return parsed


def _query_count(connection: duckdb.DuckDBPyConnection, relation: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()
    if row is None:
        raise SampledSnapshotError(f"Unable to count rows for relation {relation}")
    return int(row[0])


def relocate_sampled_snapshot(
    *,
    source_duckdb: Path,
    target_directory: Path,
    collection_root: Path,
    collection_id: str,
    sampling_evidence_path: Path,
    expected_sampled_games: int,
    expected_sampled_moves: int,
    expected_full_games: int,
    expected_full_moves: int,
) -> SampledSnapshotRelocationResult:
    if not source_duckdb.exists():
        raise SampledSnapshotError(f"Source DuckDB snapshot not found: {source_duckdb.as_posix()}")

    target_directory.mkdir(parents=True, exist_ok=True)
    target_duckdb = target_directory / source_duckdb.name

    if source_duckdb.resolve() == target_duckdb.resolve():
        raise SampledSnapshotError("Source and target DuckDB paths must be different")
    if target_duckdb.exists():
        raise SampledSnapshotError(
            "Refusing to overwrite existing target DuckDB snapshot: "
            f"{target_duckdb.as_posix()}"
        )

    shutil.copy2(source_duckdb, target_duckdb)

    expected_collection_root = _normalize_path(collection_root.resolve().as_posix())
    stale_collection_roots: set[str] = set()
    replaced_views: list[str] = []
    exact_path_replacements: dict[str, str] = {}
    unsupported_stale_paths: set[str] = set()

    inspection = inspect_sampled_snapshot(
        duckdb_path=target_duckdb,
        expected_collection_root=collection_root,
    )
    if not inspection.deterministic_sample_detected:
        raise SampledSnapshotError(
            "Target snapshot does not appear to be a deterministic_sample DuckDB warehouse"
        )

    collection_ids = set(inspection.collection_ids_from_paths)
    if collection_ids and collection_ids != {collection_id}:
        raise SampledSnapshotError(
            "Snapshot references unexpected collection IDs: " + ", ".join(sorted(collection_ids))
        )

    relocated_manifest_path = target_directory / "collection_bronze_manifest.json"
    relocated_manifest_written = False
    for stale_paths in inspection.stale_paths_by_view.values():
        for stale_path in stale_paths:
            normalized = _normalize_path(stale_path)
            extracted_collection_id = _extract_collection_id(normalized)
            if extracted_collection_id == collection_id:
                root = _collection_root_from_path(normalized, collection_id)
                if root is not None and root.lower() != expected_collection_root.lower():
                    stale_collection_roots.add(root)
                continue

            if normalized.lower().endswith("/collection_bronze_manifest.json"):
                exact_path_replacements[normalized] = _normalize_path(
                    relocated_manifest_path.resolve().as_posix()
                )
                continue

            unsupported_stale_paths.add(normalized)

    if unsupported_stale_paths:
        raise SampledSnapshotError(
            "Unsupported stale absolute paths found in sampled snapshot view SQL. "
            "Refusing unsafe rewrite. Paths: " + ", ".join(sorted(unsupported_stale_paths))
        )

    if exact_path_replacements and not relocated_manifest_written:
        manifest_payload = _build_relocated_collection_manifest_payload(
            collection_root=collection_root,
            collection_id=collection_id,
            expected_sampled_games=expected_sampled_games,
            expected_sampled_moves=expected_sampled_moves,
        )
        relocated_manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        relocated_manifest_written = True

    connection = duckdb.connect(str(target_duckdb))
    try:
        for view_name, sql_text in inspection.view_sql_by_name.items():
            updated_sql = sql_text
            changed = False

            for stale_root in sorted(stale_collection_roots, key=len, reverse=True):
                replacement = expected_collection_root
                rewritten = updated_sql.replace(stale_root, replacement)
                rewritten = rewritten.replace(
                    stale_root.replace("/", "\\"),
                    replacement.replace("/", "\\"),
                )
                if rewritten != updated_sql:
                    changed = True
                    updated_sql = rewritten

            for old_path, new_path in exact_path_replacements.items():
                rewritten = updated_sql.replace(old_path, new_path)
                rewritten = rewritten.replace(
                    old_path.replace("/", "\\"),
                    new_path.replace("/", "\\"),
                )
                if rewritten != updated_sql:
                    changed = True
                    updated_sql = rewritten

            if not changed:
                continue

            statement = _view_sql_to_replace_statement(updated_sql)
            connection.execute(statement)
            replaced_views.append(view_name)
    finally:
        connection.close()

    if not replaced_views:
        raise SampledSnapshotError(
            "No stale view SQL path literals were relocated; refusing to publish unclear snapshot"
        )

    sampling = _load_sampling_evidence(sampling_evidence_path)
    raw_collection_id = sampling.get("parent_collection_id")
    evidence_collection_id = raw_collection_id.strip() if isinstance(raw_collection_id, str) else ""
    if evidence_collection_id and evidence_collection_id != collection_id:
        raise SampledSnapshotError(
            "Sampling evidence parent_collection_id does not match expected collection_id"
        )

    evidence_full_games = _as_int(
        sampling.get("parent_full_counts", {}).get("games"),
        field_name="sampling.parent_full_counts.games",
    )
    evidence_full_moves = _as_int(
        sampling.get("parent_full_counts", {}).get("moves"),
        field_name="sampling.parent_full_counts.moves",
    )
    evidence_sampled_games = _as_int(
        sampling.get("sampled_counts", {}).get("games"),
        field_name="sampling.sampled_counts.games",
    )
    evidence_sampled_moves = _as_int(
        sampling.get("sampled_counts", {}).get("moves"),
        field_name="sampling.sampled_counts.moves",
    )

    expected_pairs = {
        "full games": (expected_full_games, evidence_full_games),
        "full moves": (expected_full_moves, evidence_full_moves),
        "sampled games": (expected_sampled_games, evidence_sampled_games),
        "sampled moves": (expected_sampled_moves, evidence_sampled_moves),
    }
    for label, (expected_value, evidence_value) in expected_pairs.items():
        if expected_value != evidence_value:
            raise SampledSnapshotError(
                "Sampling evidence mismatch for "
                f"{label}: expected={expected_value} evidence={evidence_value}"
            )

    verify = duckdb.connect(str(target_duckdb), read_only=True)
    try:
        stg_games_rows = _query_count(verify, "main.stg_games")
        int_move_context_rows = _query_count(verify, "main.int_move_context")
        sampled_game_keys_rows = _query_count(verify, "main.sampled_game_keys")
        sampled_game_indices_rows = _query_count(verify, "main.sampled_game_indices")
        distinct_game_ids = _query_count(verify, "(SELECT DISTINCT game_id FROM main.stg_games)")
        moves_without_game_rows = _query_count(
            verify,
            """
            (
                SELECT m.game_id
                FROM main.int_move_context AS m
                LEFT JOIN main.stg_games AS g
                    ON m.game_id = g.game_id
                WHERE g.game_id IS NULL
            )
            """,
        )
        collection_id_row = verify.execute(
            "SELECT dataset_id FROM main.bronze_manifest LIMIT 1"
        ).fetchone()
        if collection_id_row is None:
            raise SampledSnapshotError("bronze_manifest returned no rows")
        current_collection_id = str(collection_id_row[0])
    finally:
        verify.close()

    if current_collection_id != collection_id:
        raise SampledSnapshotError(
            "Relocated snapshot collection_id mismatch: "
            f"expected={collection_id} actual={current_collection_id}"
        )

    if stg_games_rows != expected_sampled_games:
        raise SampledSnapshotError(
            "Relocated stg_games row count mismatch: "
            f"expected={expected_sampled_games} actual={stg_games_rows}"
        )
    if int_move_context_rows != expected_sampled_moves:
        raise SampledSnapshotError(
            "Relocated int_move_context row count mismatch: "
            f"expected={expected_sampled_moves} actual={int_move_context_rows}"
        )

    if stg_games_rows == expected_full_games or int_move_context_rows == expected_full_moves:
        raise SampledSnapshotError(
            "Relocated snapshot still reflects full-population counts for sampled relations"
        )

    if sampled_game_keys_rows != expected_sampled_games:
        raise SampledSnapshotError(
            "sampled_game_keys row count mismatch: "
            f"expected={expected_sampled_games} actual={sampled_game_keys_rows}"
        )
    if sampled_game_indices_rows != expected_sampled_games:
        raise SampledSnapshotError(
            "sampled_game_indices row count mismatch: "
            f"expected={expected_sampled_games} actual={sampled_game_indices_rows}"
        )

    if distinct_game_ids != expected_sampled_games:
        raise SampledSnapshotError(
            "Distinct stg_games game_id count mismatch: "
            f"expected={expected_sampled_games} actual={distinct_game_ids}"
        )
    if moves_without_game_rows != 0:
        raise SampledSnapshotError(
            "int_move_context contains rows without a matching stg_games game_id"
        )

    return SampledSnapshotRelocationResult(
        source_duckdb=source_duckdb,
        target_duckdb=target_duckdb,
        preserved_filename=(target_duckdb.name == source_duckdb.name),
        replaced_views=tuple(sorted(set(replaced_views))),
        replaced_collection_roots=tuple(sorted(stale_collection_roots)),
        stg_games_rows=stg_games_rows,
        int_move_context_rows=int_move_context_rows,
        sampled_game_keys_rows=sampled_game_keys_rows,
        sampled_game_indices_rows=sampled_game_indices_rows,
        distinct_game_ids=distinct_game_ids,
        moves_without_game_rows=moves_without_game_rows,
        collection_id=current_collection_id,
    )
