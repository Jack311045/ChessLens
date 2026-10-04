from __future__ import annotations

from pathlib import Path

import pytest

from chesslens.modeling.config import apply_cli_overrides, load_modeling_config


def _write_config(path: Path, *, schema_version: str = "1.1.0") -> None:
    path.write_text(
        "\n".join(
            [
                "input:",
                "  collection_root: data/processed/collections/fixture_collection",
                "  duckdb_path: data/tmp/chesslens_ci_collection.duckdb",
                (
                    "  warehouse_provenance_path: "
                    "data/manifests/fixture_modeling_warehouse_provenance.json"
                ),
                "  expected_collection_id: null",
                "versions:",
                "  modeling_pipeline_version: modeling_dataset_v2",
                "  split_definition_version: temporal_game_split_v1",
                "  feature_schema_version: policy_value_features_v1",
                "  label_definition_version: policy_value_labels_v1",
                f"  schema_version: {schema_version}",
                "  position_normalization_version: fen4_legal_ep_v1",
                "  board_encoding_version: board18_abs_v1",
                "  action_encoding_version: action8x8x73_v1",
                "sampling:",
                "  rule_version: sha256_mod_v1",
                "  seed: fixture-sampling-v1",
                "  hash_modulus: 10000",
                "  hash_threshold: 10000",
                "  requested_rate_percent: 100.0",
                "  max_games: null",
                "splits:",
                "  train:",
                "    start_date: 2013-01-01",
                "    end_date: 2013-01-20",
                "  validation:",
                "    start_date: 2013-01-21",
                "    end_date: 2013-01-25",
                "  test:",
                "    start_date: 2013-01-26",
                "    end_date: 2013-01-31",
                "  missing_or_invalid_date_policy: assign_train",
                "player_holdout:",
                "  rule_version: sha256_mod_v1",
                "  seed: fixture-holdout-v1",
                "  hash_modulus: 10000",
                "  hash_threshold: 1000",
                "  requested_rate_percent: 10.0",
                "  missing_player_hash_policy: exclude_from_player_disjoint_training",
                "output:",
                "  output_root: data/modeling",
                "  batch_rows: 5000",
                "  max_examples: null",
                "  parquet_compression: zstd",
                "  parquet_row_group_size: 2048",
                "behavior:",
                "  strict: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_modeling_config_success(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    config = load_modeling_config(config_path)

    assert config.sampling.hash_threshold == 10000
    assert config.splits.train.start_date.isoformat() == "2013-01-01"
    assert config.output.batch_rows == 5000
    assert (
        config.input.warehouse_provenance_path.name
        == "fixture_modeling_warehouse_provenance.json"
    )


def test_load_modeling_config_rejects_unsupported_version(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-version.yaml"
    _write_config(config_path, schema_version="9.9.9")

    with pytest.raises(ValueError, match="Unsupported versions.schema_version"):
        load_modeling_config(config_path)


def test_apply_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    config = load_modeling_config(config_path)

    overridden = apply_cli_overrides(
        config,
        collection_root="data/other_collection",
        output_root="data/other_modeling",
        duckdb_path="data/tmp/other.duckdb",
        warehouse_provenance_path="data/manifests/other_provenance.json",
        max_games=123,
        max_examples=456,
    )

    assert overridden.input.collection_root.name == "other_collection"
    assert overridden.output.output_root.name == "other_modeling"
    assert overridden.input.duckdb_path.name == "other.duckdb"
    assert overridden.input.warehouse_provenance_path.name == "other_provenance.json"
    assert overridden.sampling.max_games == 123
    assert overridden.output.max_examples == 456


def test_load_modeling_config_requires_provenance_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    lines = config_path.read_text(encoding="utf-8").splitlines()
    filtered = [line for line in lines if "warehouse_provenance_path" not in line]
    config_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input.warehouse_provenance_path"):
        load_modeling_config(config_path)
