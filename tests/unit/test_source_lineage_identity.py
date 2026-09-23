from __future__ import annotations

from pathlib import Path

import duckdb

from chesslens.ingestion.pgn_reader import compute_sha256, stable_game_id
from chesslens.ingestion.run_ingestion import run_ingestion
from tests.conftest import write_zst_text

_G0 = '[Event "g0"]\n[Result "1-0"]\n\n1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0\n'
_G1_INCOMPLETE = '[Event "g1"]\n[Result "*"]\n\n1. d4 d5 *\n'
_G2 = '[Event "g2"]\n[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n'


def _write_config(tmp_path: Path, archive: Path) -> Path:
    body = "\n".join(
        [
            f"input_path: {archive.as_posix()}",
            f"output_root: {(tmp_path / 'processed').as_posix()}",
            "source_month: 2013-01",
            "strict: false",
            "require_complete_games: true",
            "batch_games: 5",
            "max_buffered_records: 400",
            "parquet_compression: zstd",
            "parquet_row_group_size: 128",
            "player_hash_mode: fixture_placeholder",
        ]
    )
    path = tmp_path / "identity.yaml"
    path.write_text(body + "\n", encoding="utf-8")
    return path


def test_rejected_game_does_not_shift_later_global_indices(tmp_path: Path) -> None:
    archive = write_zst_text(
        tmp_path, "\n".join([_G0, _G1_INCOMPLETE, _G2]) + "\n", filename="synthetic.pgn.zst"
    )
    config_path = _write_config(tmp_path, archive)
    result = run_ingestion(config_path=config_path, overrides={"max_games": None})

    assert result.accepted_games == 2
    assert result.rejected_games == 1
    assert result.scanned_games == 3

    glob = (result.dataset_path / "games" / "source_month=*" / "part-*.parquet").as_posix()
    connection = duckdb.connect(":memory:")
    try:
        rows = connection.execute(
            f"SELECT source_game_index, game_id FROM read_parquet('{glob}') "
            "ORDER BY source_game_index"
        ).fetchall()
    finally:
        connection.close()

    indices = [int(r[0]) for r in rows]
    assert indices == [0, 2]

    parent_sha = compute_sha256(archive)
    game_ids = {int(r[0]): r[1] for r in rows}
    assert game_ids[2] == stable_game_id(parent_sha, 2)
