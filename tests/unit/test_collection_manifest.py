from __future__ import annotations

import pytest

from chesslens.ingestion.collection_manifest import (
    CollectionManifestError,
    CollectionShardEntry,
    build_collection_manifest_dict,
    derive_collection_id,
    parse_collection_entries,
    validate_collection_reconciliation,
)


def _entry(
    index: int, start: int, end: int, *, dataset_id: str | None = None
) -> CollectionShardEntry:
    accepted = end - start
    return CollectionShardEntry(
        shard_index=index,
        dataset_id=dataset_id or f"dataset-{index}",
        dataset_relpath=f"shards/shard-{index:05d}/datasets/dataset-{index}",
        start_global_index=start,
        end_global_index=end,
        scanned_games=accepted,
        accepted_games=accepted,
        rejected_games=0,
        emitted_moves=accepted * 10,
        error_records=0,
        output_bytes=1000,
        processing_active_seconds=1.5,
        peak_rss_bytes=2000,
    )


def _manifest(entries: list[CollectionShardEntry], *, status: str = "complete") -> dict:
    return build_collection_manifest_dict(
        collection_id="c0",
        status=status,
        parent_archive_filename="parent.pgn.zst",
        parent_archive_sha256="b" * 64,
        source_month="2017-01",
        shard_manifest_identity_hash="h0",
        games_per_shard=7,
        player_hmac_key_id=None,
        expected_raw_games=20,
        selected_shard_count=len(entries),
        completed_shard_count=len(entries),
        started_at_utc="t0",
        updated_at_utc="t1",
        entries=entries,
    )


def test_collection_id_is_deterministic_and_path_free() -> None:
    a = derive_collection_id(
        parent_archive_sha256="b" * 64,
        source_month="2017-01",
        shard_manifest_identity_hash="h0",
        games_per_shard=250000,
        player_hmac_key_id="key-1",
    )
    b = derive_collection_id(
        parent_archive_sha256="b" * 64,
        source_month="2017-01",
        shard_manifest_identity_hash="h0",
        games_per_shard=250000,
        player_hmac_key_id="key-1",
    )
    assert a == b
    other = derive_collection_id(
        parent_archive_sha256="b" * 64,
        source_month="2017-01",
        shard_manifest_identity_hash="h0",
        games_per_shard=250000,
        player_hmac_key_id="key-2",
    )
    assert other != a


def test_reconciliation_ok() -> None:
    manifest = _manifest([_entry(0, 0, 7), _entry(1, 7, 14), _entry(2, 14, 20)])
    validate_collection_reconciliation(manifest, require_complete=True)
    assert manifest["counts"]["accepted_games"] == 20


def test_reconciliation_rejects_gap() -> None:
    manifest = _manifest([_entry(0, 0, 7), _entry(1, 8, 14)])
    with pytest.raises(CollectionManifestError, match="gap/overlap"):
        validate_collection_reconciliation(manifest)


def test_reconciliation_rejects_duplicate_dataset_id() -> None:
    manifest = _manifest([_entry(0, 0, 7), _entry(1, 7, 14, dataset_id="dataset-0")])
    with pytest.raises(CollectionManifestError, match="Duplicate dataset id"):
        validate_collection_reconciliation(manifest)


def test_reconciliation_rejects_duplicate_shard_index() -> None:
    manifest = _manifest([_entry(0, 0, 7), _entry(0, 7, 14)])
    with pytest.raises(CollectionManifestError, match="Non-contiguous shard index"):
        validate_collection_reconciliation(manifest)


def test_require_complete_rejects_incomplete() -> None:
    manifest = _manifest([_entry(0, 0, 7)], status="incomplete")
    with pytest.raises(CollectionManifestError, match="not 'complete'"):
        validate_collection_reconciliation(manifest, require_complete=True)


def test_parse_entries_round_trip() -> None:
    manifest = _manifest([_entry(0, 0, 7), _entry(1, 7, 14)])
    assert len(parse_collection_entries(manifest)) == 2
