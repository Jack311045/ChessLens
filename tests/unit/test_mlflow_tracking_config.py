from __future__ import annotations

from pathlib import Path

import pytest

from chesslens.modeling.mlflow_tracking_config import load_mlflow_tracking_config


def test_load_mlflow_tracking_config_success(tmp_path: Path) -> None:
    config_path = tmp_path / "mlflow.yaml"
    config_path.write_text(
        "\n".join(
            [
                "tracking:",
                "  experiment_name: ChessLensLocal",
                "  sqlite_path: data/mlflow/local.db",
                "  artifact_root: data/mlflow/artifacts",
                "  strict: true",
                "  default_include_model_weights: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_mlflow_tracking_config(config_path)

    assert config.experiment_name == "ChessLensLocal"
    assert config.sqlite_path.name == "local.db"
    assert config.artifact_root.name == "artifacts"
    assert config.tracking_uri.startswith("sqlite:///")


def test_load_mlflow_tracking_config_rejects_non_sqlite_path(tmp_path: Path) -> None:
    config_path = tmp_path / "mlflow.yaml"
    config_path.write_text(
        "\n".join(
            [
                "tracking:",
                "  experiment_name: ChessLensLocal",
                "  sqlite_path: data/mlflow/not_a_db.txt",
                "  artifact_root: data/mlflow/artifacts",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sqlite database file"):
        load_mlflow_tracking_config(config_path)
