from __future__ import annotations

import json
from pathlib import Path

from chesslens.warehouse.benchmark import build_benchmark_summary


def test_build_benchmark_summary_from_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": "abc",
        "status": "complete",
        "source": {
            "source_month": "2013-01",
            "archive_filename": "lichess_db_standard_rated_2013-01.pgn.zst",
            "archive_sha256": "e" * 64,
        },
        "counts": {
            "accepted_games": 12,
            "rejected_games": 0,
            "emitted_moves": 345,
            "error_records": 0,
        },
        "performance": {
            "total_games_per_second": 10.5,
            "total_moves_per_second": 201.25,
            "peak_rss_bytes": 2000000,
        },
        "versions": {
            "ingestion_pipeline_version": "parquet_etl_v1",
            "schema_version": "1.1.0",
            "position_normalization_version": "fen4_legal_ep_v1",
            "board_encoding_version": "board18_abs_v1",
            "action_encoding_version": "action8x8x73_v1",
        },
    }
    (dataset_root / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = build_benchmark_summary(dataset_root, commit_sha="1234567")

    assert summary["dataset"]["dataset_id"] == "abc"
    assert summary["counts"]["accepted_games"] == 12
    assert summary["performance"]["throughput_games_per_second"] == 10.5
    assert summary["versions"]["schema"] == "1.1.0"
    assert summary["git_commit"] == "1234567"
