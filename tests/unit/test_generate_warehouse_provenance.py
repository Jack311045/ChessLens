from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pytest

import chesslens.modeling.generate_warehouse_provenance as provenance_module
from chesslens.modeling.generate_warehouse_provenance import generate_warehouse_provenance
from chesslens.modeling.provenance import load_warehouse_provenance


def _write_collection_manifest(collection_root: Path, *, games: int, moves: int) -> None:
    collection_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection_id": "collection-fixture-001",
        "status": "complete",
        "counts": {
            "accepted_games": games,
            "emitted_moves": moves,
        },
    }
    (collection_root / "_collection_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _create_duckdb(path: Path, *, games: int, moves: int) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute("CREATE TABLE main.stg_games (game_id VARCHAR)")
        connection.execute("CREATE TABLE main.int_move_context (game_id VARCHAR, ply INTEGER)")

        if games > 0:
            game_rows = [(f"g{idx}",) for idx in range(games)]
            connection.executemany("INSERT INTO main.stg_games (game_id) VALUES (?)", game_rows)

        if moves > 0:
            move_rows = [(f"g{idx}", idx) for idx in range(moves)]
            connection.executemany(
                "INSERT INTO main.int_move_context (game_id, ply) VALUES (?, ?)",
                move_rows,
            )
    finally:
        connection.close()


def _write_sampling_evidence(
    path: Path,
    *,
    full_games: int,
    full_moves: int,
    sampled_games: int,
    sampled_moves: int,
) -> None:
    payload: dict[str, Any] = {
        "sampling": {
            "parent_collection_id": "collection-fixture-001",
            "sampling_rule_version": "sha256_mod_v1",
            "sampling_predicate": "mod(hash(game_id), 10000) < 4000",
            "requested_rate_percent": 40.0,
            "hash_modulus": 10000,
            "hash_threshold": 4000,
            "parent_full_counts": {
                "games": full_games,
                "moves": full_moves,
            },
            "sampled_counts": {
                "games": sampled_games,
                "moves": sampled_moves,
            },
        }
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_generate_full_warehouse_provenance(tmp_path: Path) -> None:
    collection_root = tmp_path / "collection"
    duckdb_path = tmp_path / "warehouse.duckdb"
    output_path = tmp_path / "warehouse_provenance.json"

    _write_collection_manifest(collection_root, games=3, moves=5)
    _create_duckdb(duckdb_path, games=3, moves=5)

    payload = generate_warehouse_provenance(
        collection_root=collection_root,
        duckdb_path=duckdb_path,
        games_relation="main.stg_games",
        move_context_relation="main.int_move_context",
        warehouse_kind="full",
        sampling_evidence_path=None,
        output_path=output_path,
        dbt_project_root=Path("dbt"),
        allow_overwrite_incompatible=False,
    )

    assert output_path.exists()
    assert payload["warehouse_kind"] == "full"

    loaded = load_warehouse_provenance(output_path)
    assert loaded.collection_id == "collection-fixture-001"
    assert loaded.snapshot_counts.games == 3
    assert loaded.snapshot_counts.moves == 5
    assert loaded.transformation_identity.identity_kind == "dbt_project_fingerprint_v1"


def test_generate_sampled_warehouse_provenance(tmp_path: Path) -> None:
    collection_root = tmp_path / "collection"
    duckdb_path = tmp_path / "warehouse.duckdb"
    output_path = tmp_path / "warehouse_provenance.json"
    evidence_path = tmp_path / "sampling_evidence.json"

    _write_collection_manifest(collection_root, games=10, moves=20)
    _create_duckdb(duckdb_path, games=4, moves=8)
    _write_sampling_evidence(
        evidence_path,
        full_games=10,
        full_moves=20,
        sampled_games=4,
        sampled_moves=8,
    )

    payload = generate_warehouse_provenance(
        collection_root=collection_root,
        duckdb_path=duckdb_path,
        games_relation="main.stg_games",
        move_context_relation="main.int_move_context",
        warehouse_kind="deterministic_sample",
        sampling_evidence_path=evidence_path,
        output_path=output_path,
        dbt_project_root=Path("dbt"),
        allow_overwrite_incompatible=False,
    )

    assert payload["sampling"] is not None
    loaded = load_warehouse_provenance(output_path)
    assert loaded.warehouse_kind == "deterministic_sample"
    assert loaded.sampling is not None
    assert loaded.sampling.sampled_counts.games == 4
    assert loaded.sampling.sampled_counts.moves == 8


def test_generate_sampled_provenance_fails_for_sample_count_mismatch(tmp_path: Path) -> None:
    collection_root = tmp_path / "collection"
    duckdb_path = tmp_path / "warehouse.duckdb"
    output_path = tmp_path / "warehouse_provenance.json"
    evidence_path = tmp_path / "sampling_evidence.json"

    _write_collection_manifest(collection_root, games=10, moves=20)
    _create_duckdb(duckdb_path, games=4, moves=8)
    _write_sampling_evidence(
        evidence_path,
        full_games=10,
        full_moves=20,
        sampled_games=3,
        sampled_moves=8,
    )

    with pytest.raises(RuntimeError, match="sampled_counts do not match"):
        generate_warehouse_provenance(
            collection_root=collection_root,
            duckdb_path=duckdb_path,
            games_relation="main.stg_games",
            move_context_relation="main.int_move_context",
            warehouse_kind="deterministic_sample",
            sampling_evidence_path=evidence_path,
            output_path=output_path,
            dbt_project_root=Path("dbt"),
            allow_overwrite_incompatible=False,
        )


def test_generate_sampled_provenance_rejects_mixed_full_sample_population(
    tmp_path: Path,
) -> None:
    collection_root = tmp_path / "collection"
    duckdb_path = tmp_path / "warehouse.duckdb"
    output_path = tmp_path / "warehouse_provenance.json"
    evidence_path = tmp_path / "sampling_evidence.json"

    _write_collection_manifest(collection_root, games=10, moves=20)
    _create_duckdb(duckdb_path, games=4, moves=20)
    _write_sampling_evidence(
        evidence_path,
        full_games=10,
        full_moves=20,
        sampled_games=4,
        sampled_moves=8,
    )

    with pytest.raises(RuntimeError, match="sampled_counts do not match"):
        generate_warehouse_provenance(
            collection_root=collection_root,
            duckdb_path=duckdb_path,
            games_relation="main.stg_games",
            move_context_relation="main.int_move_context",
            warehouse_kind="deterministic_sample",
            sampling_evidence_path=evidence_path,
            output_path=output_path,
            dbt_project_root=Path("dbt"),
            allow_overwrite_incompatible=False,
        )


def test_generate_provenance_refuses_incompatible_overwrite_without_flag(tmp_path: Path) -> None:
    collection_root = tmp_path / "collection"
    duckdb_path = tmp_path / "warehouse.duckdb"
    output_path = tmp_path / "warehouse_provenance.json"

    _write_collection_manifest(collection_root, games=3, moves=5)
    _create_duckdb(duckdb_path, games=3, moves=5)

    first = generate_warehouse_provenance(
        collection_root=collection_root,
        duckdb_path=duckdb_path,
        games_relation="main.stg_games",
        move_context_relation="main.int_move_context",
        warehouse_kind="full",
        sampling_evidence_path=None,
        output_path=output_path,
        dbt_project_root=Path("dbt"),
        allow_overwrite_incompatible=False,
    )

    connection = duckdb.connect(str(duckdb_path))
    try:
        connection.execute("INSERT INTO main.stg_games (game_id) VALUES ('g-extra')")
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="Refusing to overwrite incompatible"):
        generate_warehouse_provenance(
            collection_root=collection_root,
            duckdb_path=duckdb_path,
            games_relation="main.stg_games",
            move_context_relation="main.int_move_context",
            warehouse_kind="fixture",
            sampling_evidence_path=None,
            output_path=output_path,
            dbt_project_root=Path("dbt"),
            allow_overwrite_incompatible=False,
        )

    second = generate_warehouse_provenance(
        collection_root=collection_root,
        duckdb_path=duckdb_path,
        games_relation="main.stg_games",
        move_context_relation="main.int_move_context",
        warehouse_kind="fixture",
        sampling_evidence_path=None,
        output_path=output_path,
        dbt_project_root=Path("dbt"),
        allow_overwrite_incompatible=True,
    )

    assert first["warehouse_provenance_sha256"] != second["warehouse_provenance_sha256"]


def test_generate_warehouse_provenance_main_uses_env_path_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    collection_root = tmp_path / "collection"
    duckdb_path = tmp_path / "warehouse.duckdb"
    output_path = tmp_path / "warehouse_provenance.json"

    observed: dict[str, object] = {}

    def _fake_generate_warehouse_provenance(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setenv("CHESSLENS_COLLECTION_ROOT", str(collection_root))
    monkeypatch.setenv("CHESSLENS_DUCKDB_PATH", str(duckdb_path))
    monkeypatch.setattr(
        provenance_module,
        "generate_warehouse_provenance",
        _fake_generate_warehouse_provenance,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_warehouse_provenance",
            "--warehouse-kind",
            "fixture",
            "--output",
            str(output_path),
        ],
    )

    provenance_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "ok"}
    assert observed["collection_root"] == collection_root
    assert observed["duckdb_path"] == duckdb_path
    assert observed["output_path"] == output_path
