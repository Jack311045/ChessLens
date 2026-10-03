"""Generate a canonical warehouse provenance artifact for Phase 2.1b."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

import duckdb

from chesslens.ingestion.shard_manifest import atomic_write_json
from chesslens.modeling.provenance import (
    WAREHOUSE_PROVENANCE_VERSION,
    canonical_warehouse_provenance_sha256,
)
from chesslens.modeling.validation import canonical_json, read_json_object
from chesslens.runtime_paths import (
    RuntimePathResolutionError,
    resolve_collection_root_override_or_env,
    resolve_duckdb_path_override_or_env,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{field_name} must be an object")
    return value


def _as_non_empty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise RuntimeError(f"{field_name} must be non-empty")
    return text


def _as_non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be an integer") from exc
    if parsed < 0:
        raise RuntimeError(f"{field_name} must be non-negative")
    return parsed


def _as_rate_percent(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be numeric") from exc
    if parsed < 0.0 or parsed > 100.0:
        raise RuntimeError(f"{field_name} must be in [0, 100]")
    return parsed


def _as_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_non_negative_int(value, field_name=field_name)


def _load_collection_manifest(collection_root: Path) -> dict[str, Any]:
    manifest_path = collection_root / "_collection_manifest.json"
    manifest = read_json_object(manifest_path)

    collection_id = _as_non_empty_string(
        manifest.get("collection_id"), field_name="collection_manifest.collection_id"
    )
    counts = _as_mapping(
        manifest.get("counts"), field_name="collection_manifest.counts"
    )
    accepted_games = _as_non_negative_int(
        counts.get("accepted_games"),
        field_name="collection_manifest.counts.accepted_games",
    )
    emitted_moves = _as_non_negative_int(
        counts.get("emitted_moves"),
        field_name="collection_manifest.counts.emitted_moves",
    )

    return {
        "collection_id": collection_id,
        "accepted_games": accepted_games,
        "emitted_moves": emitted_moves,
        "path": manifest_path,
    }


def _query_relation_count(connection: duckdb.DuckDBPyConnection, relation: str) -> int:
    try:
        connection.execute(f"SELECT * FROM {relation} LIMIT 0")
    except duckdb.Error as exc:
        raise RuntimeError(f"Unable to read relation {relation!r}: {exc}") from exc

    row = connection.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()
    if row is None:
        raise RuntimeError(f"Unable to count rows for relation {relation!r}")
    return int(row[0])


def _workspace_relative(path: Path) -> Path:
    cwd = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(cwd)
    except ValueError as exc:
        raise RuntimeError(
            "dbt_project_root must be inside the current workspace"
        ) from exc


def _is_relevant_dbt_path(relative_path: Path, dbt_root_rel: Path) -> bool:
    if len(relative_path.parts) < len(dbt_root_rel.parts):
        return False
    if tuple(relative_path.parts[: len(dbt_root_rel.parts)]) != tuple(dbt_root_rel.parts):
        return False

    leaf = relative_path
    if leaf == dbt_root_rel / "dbt_project.yml":
        return True
    if leaf in {
        dbt_root_rel / "packages.yml",
        dbt_root_rel / "packages.lock",
        dbt_root_rel / "package-lock.yml",
    }:
        return True

    suffix = leaf.suffix.lower()
    if suffix not in {".sql", ".yml", ".yaml", ".csv"}:
        return False

    prefixes = {
        dbt_root_rel / "models",
        dbt_root_rel / "macros",
        dbt_root_rel / "seeds",
        dbt_root_rel / "snapshots",
    }
    return any(leaf.is_relative_to(prefix) for prefix in prefixes)


def _git_list_tracked_dbt_files(dbt_root_rel: Path) -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "--", dbt_root_rel.as_posix()],
            cwd=Path.cwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    candidates = [Path(line.strip()) for line in output.splitlines() if line.strip()]
    return [path for path in candidates if _is_relevant_dbt_path(path, dbt_root_rel)]


def _filesystem_list_dbt_files(dbt_project_root: Path, dbt_root_rel: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in dbt_project_root.rglob("*"):
        if not path.is_file():
            continue
        rel = _workspace_relative(path)
        if _is_relevant_dbt_path(rel, dbt_root_rel):
            candidates.append(rel)
    return candidates


def compute_dbt_transformation_fingerprint(
    dbt_project_root: Path,
) -> tuple[str, list[dict[str, str]]]:
    if not dbt_project_root.exists():
        raise RuntimeError(
            f"dbt_project_root does not exist: {dbt_project_root.as_posix()}"
        )

    dbt_root_rel = _workspace_relative(dbt_project_root)
    tracked = _git_list_tracked_dbt_files(dbt_root_rel)
    if tracked:
        files = sorted(
            {path.as_posix(): path for path in tracked}.values(),
            key=lambda p: p.as_posix(),
        )
    else:
        discovered = _filesystem_list_dbt_files(dbt_project_root, dbt_root_rel)
        files = sorted(
            {path.as_posix(): path for path in discovered}.values(),
            key=lambda p: p.as_posix(),
        )

    if not files:
        raise RuntimeError(
            "No relevant dbt transformation files were found for fingerprinting"
        )

    file_entries: list[dict[str, str]] = []
    for rel_path in files:
        abs_path = Path.cwd() / rel_path
        file_entries.append(
            {
                "path": rel_path.as_posix(),
                "sha256": _sha256_file(abs_path),
            }
        )

    payload = {
        "fingerprint_version": "dbt_project_fingerprint_v1",
        "files": file_entries,
    }
    return _sha256_bytes(canonical_json(payload).encode("utf-8")), file_entries


def _parse_mod_hash_predicate(predicate: str) -> tuple[int | None, int | None]:
    pattern = r"mod\(hash\(game_id\),\s*(\d+)\s*\)\s*<\s*(\d+)"
    match = re.search(pattern, predicate)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _effective_rate_percent(sampled_games: int, full_games: int) -> float | None:
    if full_games <= 0:
        return None
    return (sampled_games / full_games) * 100.0


def _normalize_sampling_evidence(
    *,
    sampling_evidence_path: Path,
    expected_collection_id: str,
    full_games: int,
    full_moves: int,
    sampled_games: int,
    sampled_moves: int,
) -> dict[str, Any]:
    payload = read_json_object(sampling_evidence_path)

    if "sampling" in payload and isinstance(payload["sampling"], dict):
        evidence = _as_mapping(payload["sampling"], field_name="sampling_evidence.sampling")
    elif "measured" in payload and isinstance(payload["measured"], dict):
        measured = _as_mapping(payload["measured"], field_name="sampling_evidence.measured")
        predicate = _as_non_empty_string(
            measured.get("sampling_predicate"),
            field_name="sampling_evidence.measured.sampling_predicate",
        )
        hash_modulus, hash_threshold = _parse_mod_hash_predicate(predicate)
        requested_rate = (
            (hash_threshold / hash_modulus) * 100.0
            if hash_modulus is not None and hash_modulus > 0 and hash_threshold is not None
            else _effective_rate_percent(
                _as_non_negative_int(
                    measured.get("sampled_games"),
                    field_name="sampling_evidence.measured.sampled_games",
                ),
                _as_non_negative_int(
                    measured.get("full_games"),
                    field_name="sampling_evidence.measured.full_games",
                ),
            )
        )
        evidence = {
            "parent_collection_id": payload.get("collection_id", expected_collection_id),
            "sampling_rule_version": measured.get("sampling_rule_version", "sha256_mod_v1"),
            "sampling_predicate": predicate,
            "requested_rate_percent": requested_rate,
            "hash_modulus": hash_modulus,
            "hash_threshold": hash_threshold,
            "parent_full_counts": {
                "games": measured.get("full_games"),
                "moves": measured.get("full_moves"),
            },
            "sampled_counts": {
                "games": measured.get("sampled_games"),
                "moves": measured.get("sampled_moves"),
            },
        }
    else:
        evidence = payload

    parent_collection_id = _as_non_empty_string(
        evidence.get("parent_collection_id"),
        field_name="sampling_evidence.parent_collection_id",
    )
    if parent_collection_id != expected_collection_id:
        raise RuntimeError(
            "Sampling evidence parent_collection_id does not match collection manifest: "
            f"{parent_collection_id!r} != {expected_collection_id!r}"
        )

    parent_full_counts = _as_mapping(
        evidence.get("parent_full_counts"),
        field_name="sampling_evidence.parent_full_counts",
    )
    evidence_full_games = _as_non_negative_int(
        parent_full_counts.get("games"),
        field_name="sampling_evidence.parent_full_counts.games",
    )
    evidence_full_moves = _as_non_negative_int(
        parent_full_counts.get("moves"),
        field_name="sampling_evidence.parent_full_counts.moves",
    )
    if evidence_full_games != full_games or evidence_full_moves != full_moves:
        raise RuntimeError(
            "Sampling evidence parent_full_counts do not match collection manifest counts: "
            f"games {evidence_full_games} vs {full_games}, "
            f"moves {evidence_full_moves} vs {full_moves}"
        )

    sampled_counts = _as_mapping(
        evidence.get("sampled_counts"), field_name="sampling_evidence.sampled_counts"
    )
    evidence_sampled_games = _as_non_negative_int(
        sampled_counts.get("games"),
        field_name="sampling_evidence.sampled_counts.games",
    )
    evidence_sampled_moves = _as_non_negative_int(
        sampled_counts.get("moves"),
        field_name="sampling_evidence.sampled_counts.moves",
    )
    if evidence_sampled_games != sampled_games or evidence_sampled_moves != sampled_moves:
        raise RuntimeError(
            "Sampling evidence sampled_counts do not match current DuckDB relation counts: "
            f"games {evidence_sampled_games} vs {sampled_games}, "
            f"moves {evidence_sampled_moves} vs {sampled_moves}"
        )

    hash_modulus = _as_optional_int(
        evidence.get("hash_modulus"),
        field_name="sampling_evidence.hash_modulus",
    )
    hash_threshold = _as_optional_int(
        evidence.get("hash_threshold"),
        field_name="sampling_evidence.hash_threshold",
    )
    if hash_modulus is not None and hash_modulus <= 0:
        raise RuntimeError("sampling_evidence.hash_modulus must be positive when provided")
    if hash_modulus is not None and hash_threshold is not None and hash_threshold > hash_modulus:
        raise RuntimeError(
            "sampling_evidence.hash_threshold must be <= sampling_evidence.hash_modulus"
        )

    requested_rate = _as_rate_percent(
        evidence.get("requested_rate_percent"),
        field_name="sampling_evidence.requested_rate_percent",
    )

    return {
        "parent_collection_id": parent_collection_id,
        "sampling_rule_version": _as_non_empty_string(
            evidence.get("sampling_rule_version"),
            field_name="sampling_evidence.sampling_rule_version",
        ),
        "sampling_predicate": _as_non_empty_string(
            evidence.get("sampling_predicate"),
            field_name="sampling_evidence.sampling_predicate",
        ),
        "requested_rate_percent": requested_rate,
        "hash_modulus": hash_modulus,
        "hash_threshold": hash_threshold,
        "parent_full_counts": {
            "games": full_games,
            "moves": full_moves,
        },
        "sampled_counts": {
            "games": sampled_games,
            "moves": sampled_moves,
        },
        "effective_rate_percent": _effective_rate_percent(sampled_games, full_games),
    }


def _build_provenance_payload(
    *,
    collection_manifest: dict[str, Any],
    games_relation: str,
    move_context_relation: str,
    sampled_games: int,
    sampled_moves: int,
    warehouse_kind: str,
    sampling_payload: dict[str, Any] | None,
    transformation_identity_value: str,
    transformation_identity_files: list[dict[str, str]],
) -> dict[str, Any]:
    payload_without_hash: dict[str, Any] = {
        "provenance_version": WAREHOUSE_PROVENANCE_VERSION,
        "warehouse_kind": warehouse_kind,
        "collection_id": collection_manifest["collection_id"],
        "relations": {
            "games_relation": games_relation,
            "move_context_relation": move_context_relation,
        },
        "snapshot_counts": {
            "games": sampled_games,
            "moves": sampled_moves,
        },
        "transformation_identity": {
            "identity_kind": "dbt_project_fingerprint_v1",
            "identity_value": transformation_identity_value,
        },
        "transformation_identity_details": {
            "fingerprint_version": "dbt_project_fingerprint_v1",
            "files": transformation_identity_files,
        },
        "sampling": sampling_payload,
    }

    payload = dict(payload_without_hash)
    payload["warehouse_provenance_sha256"] = canonical_warehouse_provenance_sha256(
        payload_without_hash
    )
    return payload


def generate_warehouse_provenance(
    *,
    collection_root: Path,
    duckdb_path: Path,
    games_relation: str,
    move_context_relation: str,
    warehouse_kind: str,
    sampling_evidence_path: Path | None,
    output_path: Path,
    dbt_project_root: Path,
    allow_overwrite_incompatible: bool,
) -> dict[str, Any]:
    if warehouse_kind not in {"full", "deterministic_sample", "fixture"}:
        raise RuntimeError("warehouse_kind must be one of: full, deterministic_sample, fixture")

    collection_manifest = _load_collection_manifest(collection_root)
    if not duckdb_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {duckdb_path.as_posix()}")

    connection = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        relation_games = _query_relation_count(connection, games_relation)
        relation_moves = _query_relation_count(connection, move_context_relation)
    finally:
        connection.close()

    accepted_games = int(collection_manifest["accepted_games"])
    emitted_moves = int(collection_manifest["emitted_moves"])

    sampling_payload: dict[str, Any] | None = None
    if warehouse_kind == "full":
        if relation_games != accepted_games or relation_moves != emitted_moves:
            raise RuntimeError(
                "warehouse_kind='full' requires relation counts to match collection counts: "
                f"games {relation_games} vs {accepted_games}, "
                f"moves {relation_moves} vs {emitted_moves}"
            )
        if sampling_evidence_path is not None:
            raise RuntimeError(
                "sampling_evidence_path is not allowed when warehouse_kind='full'"
            )
    elif warehouse_kind == "deterministic_sample":
        if sampling_evidence_path is None:
            raise RuntimeError(
                "sampling_evidence_path is required when warehouse_kind='deterministic_sample'"
            )
        if relation_games > accepted_games or relation_moves > emitted_moves:
            raise RuntimeError(
                "deterministic_sample relation counts cannot exceed collection counts"
            )
        sampling_payload = _normalize_sampling_evidence(
            sampling_evidence_path=sampling_evidence_path,
            expected_collection_id=str(collection_manifest["collection_id"]),
            full_games=accepted_games,
            full_moves=emitted_moves,
            sampled_games=relation_games,
            sampled_moves=relation_moves,
        )
    else:
        if sampling_evidence_path is not None:
            raise RuntimeError(
                "sampling_evidence_path is not allowed when warehouse_kind='fixture'"
            )

    transformation_identity, transformation_files = compute_dbt_transformation_fingerprint(
        dbt_project_root
    )

    payload = _build_provenance_payload(
        collection_manifest=collection_manifest,
        games_relation=games_relation,
        move_context_relation=move_context_relation,
        sampled_games=relation_games,
        sampled_moves=relation_moves,
        warehouse_kind=warehouse_kind,
        sampling_payload=sampling_payload,
        transformation_identity_value=transformation_identity,
        transformation_identity_files=transformation_files,
    )

    if output_path.exists():
        try:
            existing = read_json_object(output_path)
        except Exception as exc:
            if not allow_overwrite_incompatible:
                raise RuntimeError(
                    "Refusing to overwrite unreadable or invalid existing provenance artifact "
                    f"without --allow-overwrite-incompatible: {output_path.as_posix()} ({exc})"
                ) from exc
        else:
            if existing != payload and not allow_overwrite_incompatible:
                raise RuntimeError(
                    "Refusing to overwrite incompatible existing provenance artifact without "
                    f"--allow-overwrite-incompatible: {output_path.as_posix()}"
                )

    atomic_write_json(output_path, payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a warehouse provenance artifact for Phase 2.1b"
    )
    parser.add_argument("--collection-root", default=None)
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--games-relation", default="main.stg_games")
    parser.add_argument("--move-context-relation", default="main.int_move_context")
    parser.add_argument(
        "--warehouse-kind",
        required=True,
        choices=["full", "deterministic_sample", "fixture"],
    )
    parser.add_argument("--sampling-evidence-path", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dbt-project-root", default="dbt")
    parser.add_argument("--allow-overwrite-incompatible", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        collection_root = resolve_collection_root_override_or_env(args.collection_root)
    except RuntimePathResolutionError as exc:
        parser.error(str(exc))

    if collection_root is None:
        parser.error(
            "collection root is required; pass --collection-root or set "
            "CHESSLENS_COLLECTION_ROOT (or CHESSLENS_DATA_ROOT + CHESSLENS_COLLECTION_ID)"
        )

    duckdb_path = resolve_duckdb_path_override_or_env(args.duckdb_path, default_path=None)
    if duckdb_path is None:
        parser.error(
            "duckdb path is required; pass --duckdb-path or set CHESSLENS_DUCKDB_PATH "
            "(or CHESSLENS_TIER2_DB_PATH)"
        )

    payload = generate_warehouse_provenance(
        collection_root=collection_root,
        duckdb_path=duckdb_path,
        games_relation=str(args.games_relation),
        move_context_relation=str(args.move_context_relation),
        warehouse_kind=str(args.warehouse_kind),
        sampling_evidence_path=(
            Path(args.sampling_evidence_path)
            if args.sampling_evidence_path is not None
            else None
        ),
        output_path=Path(args.output),
        dbt_project_root=Path(args.dbt_project_root),
        allow_overwrite_incompatible=bool(args.allow_overwrite_incompatible),
    )
    print(canonical_json(payload))


if __name__ == "__main__":
    main()
