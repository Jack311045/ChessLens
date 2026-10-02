"""Warehouse provenance contract helpers for Phase 2.1b."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from chesslens.modeling.validation import canonical_json, read_json_object, sha256_text

WAREHOUSE_PROVENANCE_VERSION = "warehouse_provenance_v1"
WarehouseKind = Literal["full", "deterministic_sample", "fixture"]


@dataclass(frozen=True)
class WarehouseRowCounts:
    games: int
    moves: int


@dataclass(frozen=True)
class WarehouseSamplingProvenance:
    parent_collection_id: str
    sampling_rule_version: str
    sampling_predicate: str
    requested_rate_percent: float
    hash_modulus: int | None
    hash_threshold: int | None
    parent_full_counts: WarehouseRowCounts
    sampled_counts: WarehouseRowCounts
    effective_rate_percent: float | None


@dataclass(frozen=True)
class WarehouseTransformationIdentity:
    identity_kind: str
    identity_value: str


@dataclass(frozen=True)
class WarehouseProvenance:
    provenance_version: str
    warehouse_kind: WarehouseKind
    collection_id: str
    games_relation: str
    move_context_relation: str
    snapshot_counts: WarehouseRowCounts
    transformation_identity: WarehouseTransformationIdentity
    sampling: WarehouseSamplingProvenance | None
    warehouse_provenance_sha256: str
    payload: dict[str, Any]


def canonical_warehouse_provenance_sha256(payload_without_hash: dict[str, Any]) -> str:
    return sha256_text(canonical_json(payload_without_hash))


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _as_non_empty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _as_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _as_non_negative_int(value: Any, *, field_name: str) -> int:
    parsed = _as_int(value, field_name=field_name)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _as_positive_int(value: Any, *, field_name: str) -> int:
    parsed = _as_int(value, field_name=field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _as_optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_non_negative_int(value, field_name=field_name)


def _as_rate_percent(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if parsed < 0.0 or parsed > 100.0:
        raise ValueError(f"{field_name} must be in [0, 100]")
    return parsed


def _as_optional_rate_percent(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _as_rate_percent(value, field_name=field_name)


def _parse_counts(value: Any, *, field_name: str) -> WarehouseRowCounts:
    payload = _as_mapping(value, field_name=field_name)
    return WarehouseRowCounts(
        games=_as_non_negative_int(payload.get("games"), field_name=f"{field_name}.games"),
        moves=_as_non_negative_int(payload.get("moves"), field_name=f"{field_name}.moves"),
    )


def _without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "warehouse_provenance_sha256"}


def load_warehouse_provenance(path: Path) -> WarehouseProvenance:
    try:
        payload = read_json_object(path)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid warehouse provenance JSON at {path.as_posix()}: {exc}"
        ) from exc

    try:
        provenance_version = _as_non_empty_string(
            payload.get("provenance_version"),
            field_name="provenance_version",
        )
        if provenance_version != WAREHOUSE_PROVENANCE_VERSION:
            raise ValueError(
                "Unsupported provenance_version: "
                f"{provenance_version!r}. Supported value is "
                f"{WAREHOUSE_PROVENANCE_VERSION!r}."
            )

        warehouse_kind_value = _as_non_empty_string(
            payload.get("warehouse_kind"), field_name="warehouse_kind"
        )
        if warehouse_kind_value not in {"full", "deterministic_sample", "fixture"}:
            raise ValueError(
                "warehouse_kind must be one of: full, deterministic_sample, fixture"
            )
        warehouse_kind = cast(WarehouseKind, warehouse_kind_value)

        collection_id = _as_non_empty_string(
            payload.get("collection_id"), field_name="collection_id"
        )

        relations = _as_mapping(payload.get("relations"), field_name="relations")
        games_relation = _as_non_empty_string(
            relations.get("games_relation"), field_name="relations.games_relation"
        )
        move_context_relation = _as_non_empty_string(
            relations.get("move_context_relation"),
            field_name="relations.move_context_relation",
        )

        snapshot_counts = _parse_counts(
            payload.get("snapshot_counts"), field_name="snapshot_counts"
        )

        transformation_identity_payload = _as_mapping(
            payload.get("transformation_identity"),
            field_name="transformation_identity",
        )
        transformation_identity = WarehouseTransformationIdentity(
            identity_kind=_as_non_empty_string(
                transformation_identity_payload.get("identity_kind"),
                field_name="transformation_identity.identity_kind",
            ),
            identity_value=_as_non_empty_string(
                transformation_identity_payload.get("identity_value"),
                field_name="transformation_identity.identity_value",
            ),
        )

        raw_sampling = payload.get("sampling")
        sampling: WarehouseSamplingProvenance | None = None
        if raw_sampling is not None:
            sampling_payload = _as_mapping(raw_sampling, field_name="sampling")
            hash_modulus = _as_optional_non_negative_int(
                sampling_payload.get("hash_modulus"),
                field_name="sampling.hash_modulus",
            )
            hash_threshold = _as_optional_non_negative_int(
                sampling_payload.get("hash_threshold"),
                field_name="sampling.hash_threshold",
            )
            if hash_modulus is not None:
                if hash_modulus == 0:
                    raise ValueError("sampling.hash_modulus must be positive when provided")
                if hash_threshold is not None and hash_threshold > hash_modulus:
                    raise ValueError(
                        "sampling.hash_threshold must be <= sampling.hash_modulus"
                    )

            parent_full_counts = _parse_counts(
                sampling_payload.get("parent_full_counts"),
                field_name="sampling.parent_full_counts",
            )
            sampled_counts = _parse_counts(
                sampling_payload.get("sampled_counts"),
                field_name="sampling.sampled_counts",
            )
            if sampled_counts.games > parent_full_counts.games:
                raise ValueError(
                    "sampling.sampled_counts.games must be <= "
                    "sampling.parent_full_counts.games"
                )
            if sampled_counts.moves > parent_full_counts.moves:
                raise ValueError(
                    "sampling.sampled_counts.moves must be <= "
                    "sampling.parent_full_counts.moves"
                )

            sampling = WarehouseSamplingProvenance(
                parent_collection_id=_as_non_empty_string(
                    sampling_payload.get("parent_collection_id"),
                    field_name="sampling.parent_collection_id",
                ),
                sampling_rule_version=_as_non_empty_string(
                    sampling_payload.get("sampling_rule_version"),
                    field_name="sampling.sampling_rule_version",
                ),
                sampling_predicate=_as_non_empty_string(
                    sampling_payload.get("sampling_predicate"),
                    field_name="sampling.sampling_predicate",
                ),
                requested_rate_percent=_as_rate_percent(
                    sampling_payload.get("requested_rate_percent"),
                    field_name="sampling.requested_rate_percent",
                ),
                hash_modulus=hash_modulus,
                hash_threshold=hash_threshold,
                parent_full_counts=parent_full_counts,
                sampled_counts=sampled_counts,
                effective_rate_percent=_as_optional_rate_percent(
                    sampling_payload.get("effective_rate_percent"),
                    field_name="sampling.effective_rate_percent",
                ),
            )

            if sampling.sampled_counts != snapshot_counts:
                raise ValueError(
                    "sampling.sampled_counts must match snapshot_counts for sampled warehouses"
                )

        if warehouse_kind == "deterministic_sample" and sampling is None:
            raise ValueError("deterministic_sample warehouse_kind requires a sampling object")
        if warehouse_kind == "full" and sampling is not None:
            raise ValueError("full warehouse_kind must not declare a sampling object")

        declared_hash = _as_non_empty_string(
            payload.get("warehouse_provenance_sha256"),
            field_name="warehouse_provenance_sha256",
        )
        expected_hash = canonical_warehouse_provenance_sha256(_without_hash(payload))
        if declared_hash != expected_hash:
            raise ValueError(
                "warehouse_provenance_sha256 mismatch: "
                f"declared {declared_hash!r}, expected {expected_hash!r}"
            )

        return WarehouseProvenance(
            provenance_version=provenance_version,
            warehouse_kind=warehouse_kind,
            collection_id=collection_id,
            games_relation=games_relation,
            move_context_relation=move_context_relation,
            snapshot_counts=snapshot_counts,
            transformation_identity=transformation_identity,
            sampling=sampling,
            warehouse_provenance_sha256=declared_hash,
            payload=payload,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid warehouse provenance contract at {path.as_posix()}: {exc}"
        ) from exc


def warehouse_effective_rate_percent(provenance: WarehouseProvenance) -> float | None:
    if provenance.warehouse_kind == "full":
        return 100.0

    if provenance.warehouse_kind == "deterministic_sample":
        sampling = provenance.sampling
        if sampling is None:
            return None
        if sampling.effective_rate_percent is not None:
            return sampling.effective_rate_percent
        if sampling.parent_full_counts.games > 0:
            return (sampling.sampled_counts.games / sampling.parent_full_counts.games) * 100.0
        return sampling.requested_rate_percent

    return None