from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from chesslens.modeling.mlflow_tracking import import_manifest_to_mlflow
from chesslens.modeling.mlflow_tracking_config import load_mlflow_tracking_config
from chesslens.modeling.neural_training import run_neural
from chesslens.modeling.run_baselines import run_baselines
from tests.integration.test_run_baselines import _write_baseline_config, _write_modeling_fixture
from tests.integration.test_run_neural import _write_neural_config


def _write_tracking_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "tracking:",
                "  experiment_name: ChessLensFixture",
                f"  sqlite_path: {(path.parent / 'tracking.db').as_posix()}",
                f"  artifact_root: {(path.parent / 'artifacts').as_posix()}",
                "  strict: true",
                "  default_include_model_weights: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _count_runs(*, tracking_config_path: Path, pipeline_kind: str, experiment_id: str) -> int:
    config = load_mlflow_tracking_config(tracking_config_path)
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    assert experiment is not None
    runs = mlflow.search_runs(
        experiment_ids=[str(experiment.experiment_id)],
        filter_string=(
            f"tags.chesslens_pipeline_kind = '{pipeline_kind}' "
            f"and tags.chesslens_experiment_id = '{experiment_id}'"
        ),
        output_format="list",
    )
    return len(runs)


def _metric_history_len(
    *,
    tracking_config_path: Path,
    run_id: str,
    metric_key: str,
) -> int:
    config = load_mlflow_tracking_config(tracking_config_path)
    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient(tracking_uri=config.tracking_uri)
    return len(client.get_metric_history(run_id, metric_key))


def _seed_partial_import_run(
    *,
    tracking_config_path: Path,
    pipeline_kind: str,
    experiment_id: str,
    import_status: str,
) -> str:
    config = load_mlflow_tracking_config(tracking_config_path)
    mlflow.set_tracking_uri(config.tracking_uri)

    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    mlflow_experiment_id = (
        str(experiment.experiment_id)
        if experiment is not None
        else str(
            mlflow.create_experiment(
                name=config.experiment_name,
                artifact_location=config.artifact_root.as_posix(),
            )
        )
    )

    with mlflow.start_run(
        experiment_id=mlflow_experiment_id,
        run_name=f"partial-{pipeline_kind}-{experiment_id[:8]}",
    ) as run:
        mlflow.set_tags(
            {
                "chesslens_pipeline_kind": pipeline_kind,
                "chesslens_experiment_id": experiment_id,
                "chesslens_import_status": import_status,
                "chesslens_run_origin": "historical_import",
            }
        )
        return str(run.info.run_id)


def test_mlflow_import_baseline_idempotent(tmp_path: Path) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    baseline_output_root = tmp_path / "baseline_output"
    baseline_config_path = tmp_path / "baseline.yaml"
    _write_baseline_config(
        baseline_config_path,
        manifest_path=modeling_manifest,
        output_root=baseline_output_root,
    )

    baseline_result = run_baselines(config_path=baseline_config_path)

    tracking_config_path = tmp_path / "tracking.yaml"
    _write_tracking_config(tracking_config_path)

    first = import_manifest_to_mlflow(
        manifest_path=baseline_result.manifest_path,
        pipeline_kind="baseline",
        tracking_config_path=tracking_config_path,
        mode="historical",
    )
    second = import_manifest_to_mlflow(
        manifest_path=baseline_result.manifest_path,
        pipeline_kind="baseline",
        tracking_config_path=tracking_config_path,
        mode="historical",
    )

    assert first.reused_existing is False
    assert second.reused_existing is True
    assert first.mlflow_run_id == second.mlflow_run_id
    assert (
        _count_runs(
            tracking_config_path=tracking_config_path,
            pipeline_kind="baseline",
            experiment_id=baseline_result.experiment_id,
        )
        == 1
    )


def test_mlflow_import_rejects_tampered_artifact(tmp_path: Path) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    baseline_output_root = tmp_path / "baseline_output"
    baseline_config_path = tmp_path / "baseline.yaml"
    _write_baseline_config(
        baseline_config_path,
        manifest_path=modeling_manifest,
        output_root=baseline_output_root,
    )

    baseline_result = run_baselines(config_path=baseline_config_path)
    tracking_config_path = tmp_path / "tracking.yaml"
    _write_tracking_config(tracking_config_path)

    tampered_path = baseline_result.output_path / "frequency_model.json"
    tampered_path.write_text(tampered_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact .* mismatch"):
        import_manifest_to_mlflow(
            manifest_path=baseline_result.manifest_path,
            pipeline_kind="baseline",
            tracking_config_path=tracking_config_path,
            mode="historical",
        )


def test_mlflow_import_neural_idempotent(tmp_path: Path) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)

    neural_output_root = tmp_path / "neural_output"
    neural_config_path = tmp_path / "neural.yaml"
    _write_neural_config(
        neural_config_path,
        manifest_path=modeling_manifest,
        output_root=neural_output_root,
        max_epochs=2,
    )

    neural_result = run_neural(config_path=neural_config_path)
    assert neural_result.completed is True

    tracking_config_path = tmp_path / "tracking.yaml"
    _write_tracking_config(tracking_config_path)

    first = import_manifest_to_mlflow(
        manifest_path=neural_result.manifest_path,
        pipeline_kind="neural",
        tracking_config_path=tracking_config_path,
        mode="historical",
    )
    first_history_len = _metric_history_len(
        tracking_config_path=tracking_config_path,
        run_id=first.mlflow_run_id,
        metric_key="validation.policy.best_mrr",
    )
    second = import_manifest_to_mlflow(
        manifest_path=neural_result.manifest_path,
        pipeline_kind="neural",
        tracking_config_path=tracking_config_path,
        mode="historical",
    )
    second_history_len = _metric_history_len(
        tracking_config_path=tracking_config_path,
        run_id=second.mlflow_run_id,
        metric_key="validation.policy.best_mrr",
    )

    assert first.reused_existing is False
    assert second.reused_existing is True
    assert first.mlflow_run_id == second.mlflow_run_id
    assert first_history_len == second_history_len
    assert (
        _count_runs(
            tracking_config_path=tracking_config_path,
            pipeline_kind="neural",
            experiment_id=neural_result.experiment_id,
        )
        == 1
    )


def test_mlflow_import_rejects_partial_existing_run(tmp_path: Path) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    baseline_output_root = tmp_path / "baseline_output"
    baseline_config_path = tmp_path / "baseline.yaml"
    _write_baseline_config(
        baseline_config_path,
        manifest_path=modeling_manifest,
        output_root=baseline_output_root,
    )

    baseline_result = run_baselines(config_path=baseline_config_path)

    tracking_config_path = tmp_path / "tracking.yaml"
    _write_tracking_config(tracking_config_path)

    seeded_run_id = _seed_partial_import_run(
        tracking_config_path=tracking_config_path,
        pipeline_kind="baseline",
        experiment_id=baseline_result.experiment_id,
        import_status="in_progress",
    )
    assert seeded_run_id

    with pytest.raises(RuntimeError, match="incomplete/failed"):
        import_manifest_to_mlflow(
            manifest_path=baseline_result.manifest_path,
            pipeline_kind="baseline",
            tracking_config_path=tracking_config_path,
            mode="historical",
        )
