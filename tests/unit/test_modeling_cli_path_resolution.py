from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import chesslens.modeling.build_dataset as build_dataset_module
import chesslens.modeling.run_baselines as run_baselines_module


def test_build_dataset_main_uses_env_path_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "external_data"
    collection_id = "collection-abc"
    expected_collection_root = data_root / "processed" / "collections" / collection_id
    duckdb_path = tmp_path / "warehouse.duckdb"
    provenance_path = tmp_path / "warehouse_provenance.json"
    enrichment_manifest = tmp_path / "date_enrichment_manifest.json"

    observed: dict[str, object] = {}

    def _fake_run_modeling_dataset_build(**kwargs: object) -> object:
        observed.update(kwargs)
        return object()

    monkeypatch.setenv("CHESSLENS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("CHESSLENS_COLLECTION_ID", collection_id)
    monkeypatch.setenv("CHESSLENS_DUCKDB_PATH", str(duckdb_path))
    monkeypatch.setenv("CHESSLENS_WAREHOUSE_PROVENANCE_PATH", str(provenance_path))
    monkeypatch.setenv(
        "CHESSLENS_DATE_ENRICHMENT_MANIFEST_PATH", str(enrichment_manifest)
    )
    monkeypatch.setattr(
        build_dataset_module,
        "run_modeling_dataset_build",
        _fake_run_modeling_dataset_build,
    )
    monkeypatch.setattr(
        build_dataset_module,
        "modeling_build_result_to_json",
        lambda _: {"status": "ok"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_dataset", "--config", "configs/modeling/fixture.yaml"],
    )

    build_dataset_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "ok"}
    assert observed["collection_root_override"] == expected_collection_root.as_posix()
    assert observed["duckdb_path_override"] == duckdb_path.as_posix()
    assert observed["warehouse_provenance_path_override"] == provenance_path.as_posix()
    assert (
        observed["date_enrichment_manifest_path_override"]
        == enrichment_manifest.as_posix()
    )


def test_run_baselines_main_uses_modeling_manifest_env_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    modeling_manifest = tmp_path / "modeling_manifest.json"

    observed: dict[str, object] = {}

    def _fake_run_baselines(**kwargs: object) -> object:
        observed.update(kwargs)
        return object()

    monkeypatch.setenv("CHESSLENS_MODELING_MANIFEST_PATH", str(modeling_manifest))
    monkeypatch.setattr(run_baselines_module, "run_baselines", _fake_run_baselines)
    monkeypatch.setattr(run_baselines_module, "_result_json", lambda _: {"status": "ok"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baselines",
            "--config",
            "configs/baselines/fixture_smoke.yaml",
            "--dry-run",
        ],
    )

    run_baselines_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "ok"}
    assert observed["modeling_manifest_override"] == modeling_manifest.as_posix()
    assert observed["dry_run"] is True
