from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from chesslens.ingestion.date_enrichment import (
    DATE_ENRICHMENT_MANIFEST_FILENAME,
    DateEnrichmentError,
    run_date_enrichment,
)
from chesslens.ingestion.pgn_reader import compute_sha256, stable_game_id
from chesslens.ingestion.shard_manifest import (
    ShardEntry,
    atomic_write_json,
    build_shard_manifest_dict,
)
from tests.conftest import write_zst_text


def _write_shard_manifest(
    *,
    shard_dir: Path,
    source_month: str,
    parent_archive_sha256: str,
    entries: list[ShardEntry],
) -> Path:
    manifest = build_shard_manifest_dict(
        status="complete",
        parent_archive_filename=f"lichess_db_standard_rated_{source_month}.pgn.zst",
        parent_archive_sha256=parent_archive_sha256,
        parent_archive_size_bytes=123,
        source_month=source_month,
        expected_total_games=sum(item.game_count for item in entries),
        games_per_shard=2,
        parquet_compression="zstd",
        compression_level=3,
        max_game_bytes=1024 * 1024,
        started_at_utc="t0",
        finished_at_utc="t1",
        total_emitted_games=sum(item.game_count for item in entries),
        entries=entries,
        peak_rss_bytes=1,
        sharding_active_seconds=1.0,
    )
    manifest_path = shard_dir / "_shard_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _make_shard_file(path: Path, games: str) -> None:
    created = write_zst_text(path.parent, games, filename=path.name)
    assert created == path


def test_date_enrichment_builds_complete_sidecar(tmp_path: Path) -> None:
    source_month = "2017-01"
    parent_archive_sha256 = "a" * 64
    shard_dir = tmp_path / "raw_shards" / source_month
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard0_path = shard_dir / "shard-00000.pgn.zst"
    shard1_path = shard_dir / "shard-00001.pgn.zst"

    _make_shard_file(
        shard0_path,
        """
[Event "g0"]
[Date "????.??.??"]
[UTCDate "2017.01.01"]
[Result "1-0"]

1. e4 e5 1-0

[Event "g1"]
[Date "2017.01.02"]
[UTCDate "????.??.??"]
[Result "0-1"]

1. d4 d5 0-1
""".strip()
        + "\n",
    )
    _make_shard_file(
        shard1_path,
        """
[Event "g2"]
[Date "????.??.??"]
[UTCDate "2017.01.03"]
[Result "1/2-1/2"]

1. c4 c5 1/2-1/2

[Event "g3"]
[Date "????.??.??"]
[UTCDate "????.??.??"]
[Result "1-0"]

1. Nf3 Nf6 1-0
""".strip()
        + "\n",
    )

    entry0 = ShardEntry(
        shard_index=0,
        filename=shard0_path.name,
        start_global_index=0,
        end_global_index=2,
        game_count=2,
        compressed_bytes=shard0_path.stat().st_size,
        sha256=compute_sha256(shard0_path),
    )
    entry1 = ShardEntry(
        shard_index=1,
        filename=shard1_path.name,
        start_global_index=2,
        end_global_index=4,
        game_count=2,
        compressed_bytes=shard1_path.stat().st_size,
        sha256=compute_sha256(shard1_path),
    )

    _write_shard_manifest(
        shard_dir=shard_dir,
        source_month=source_month,
        parent_archive_sha256=parent_archive_sha256,
        entries=[entry0, entry1],
    )

    result = run_date_enrichment(
        shard_root=tmp_path / "raw_shards",
        source_month=source_month,
        output_root=tmp_path / "processed" / "date_enrichment",
    )

    assert result.status == "complete"
    assert result.reused_existing is False
    assert result.completed_shard_count == 2
    assert result.enriched_games == 4

    manifest_path = result.dataset_root / DATE_ENRICHMENT_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["counts"]["canonical_from_utc_date"] == 2
    assert manifest["counts"]["canonical_from_date"] == 1
    assert manifest["counts"]["canonical_missing_or_invalid"] == 1
    assert manifest["counts"]["invalid_date_but_valid_utc_date"] == 2

    sidecar_file = result.dataset_root / "shards" / "shard-00000.parquet"
    table = pq.read_table(sidecar_file)
    rows = table.to_pylist()
    assert len(rows) == 2
    assert rows[0]["game_id"] == stable_game_id(parent_archive_sha256, 0)
    assert rows[1]["game_id"] == stable_game_id(parent_archive_sha256, 1)
    assert rows[0]["canonical_date_source"] == "utc_date"
    assert rows[1]["canonical_date_source"] == "date"


def test_date_enrichment_resume_then_reuse(tmp_path: Path) -> None:
    source_month = "2017-01"
    parent_archive_sha256 = "b" * 64
    shard_dir = tmp_path / "raw_shards" / source_month
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard0_path = shard_dir / "shard-00000.pgn.zst"
    shard1_path = shard_dir / "shard-00001.pgn.zst"
    _make_shard_file(
        shard0_path,
        """
[Event "a"]
[Date "????.??.??"]
[UTCDate "2017.01.10"]
[Result "1-0"]

1. e4 e5 1-0
""".strip()
        + "\n",
    )
    _make_shard_file(
        shard1_path,
        """
[Event "b"]
[Date "2017.01.11"]
[UTCDate "????.??.??"]
[Result "0-1"]

1. d4 d5 0-1
""".strip()
        + "\n",
    )

    entry0 = ShardEntry(
        shard_index=0,
        filename=shard0_path.name,
        start_global_index=0,
        end_global_index=1,
        game_count=1,
        compressed_bytes=shard0_path.stat().st_size,
        sha256=compute_sha256(shard0_path),
    )
    entry1 = ShardEntry(
        shard_index=1,
        filename=shard1_path.name,
        start_global_index=1,
        end_global_index=2,
        game_count=1,
        compressed_bytes=shard1_path.stat().st_size,
        sha256=compute_sha256(shard1_path),
    )
    _write_shard_manifest(
        shard_dir=shard_dir,
        source_month=source_month,
        parent_archive_sha256=parent_archive_sha256,
        entries=[entry0, entry1],
    )

    output_root = tmp_path / "processed" / "date_enrichment"
    first = run_date_enrichment(
        shard_root=tmp_path / "raw_shards",
        source_month=source_month,
        output_root=output_root,
        max_new_shards=1,
    )
    assert first.status == "incomplete"
    assert first.completed_shard_count == 1

    with pytest.raises(DateEnrichmentError, match="incomplete"):
        run_date_enrichment(
            shard_root=tmp_path / "raw_shards",
            source_month=source_month,
            output_root=output_root,
        )

    resumed = run_date_enrichment(
        shard_root=tmp_path / "raw_shards",
        source_month=source_month,
        output_root=output_root,
        resume=True,
    )
    assert resumed.status == "complete"
    assert resumed.completed_shard_count == 2

    reused = run_date_enrichment(
        shard_root=tmp_path / "raw_shards",
        source_month=source_month,
        output_root=output_root,
    )
    assert reused.reused_existing is True
    assert reused.status == "complete"