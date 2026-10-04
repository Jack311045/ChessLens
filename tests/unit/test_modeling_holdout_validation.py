from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pytest

from chesslens.modeling.build_dataset import (
    _append_assignment_candidate,
    _validate_split_and_holdout_constraints,
)
from chesslens.modeling.config import (
    ModelingBehaviorConfig,
    ModelingConfig,
    ModelingInputConfig,
    ModelingOutputConfig,
    ModelingSamplingConfig,
    ModelingVersionConfig,
    PlayerHoldoutConfig,
    TemporalSplitConfig,
)
from chesslens.modeling.contracts import (
    FEATURE_SCHEMA_VERSION,
    LABEL_DEFINITION_VERSION,
    MODELING_PIPELINE_VERSION,
    SPLIT_DEFINITION_VERSION,
)
from chesslens.modeling.splits import TemporalRange


def _make_config(
    *,
    holdout_threshold: int,
    missing_player_hash_policy: str = "exclude_from_player_disjoint_training",
) -> ModelingConfig:
    return ModelingConfig(
        input=ModelingInputConfig(
            collection_root=Path("data/processed/collections/test"),
            duckdb_path=Path("data/tmp/test.duckdb"),
            warehouse_provenance_path=Path("reports/local/test_warehouse_provenance.json"),
            date_enrichment_manifest_path=None,
            require_date_enrichment=False,
            expected_collection_id=None,
            move_context_relation="main.int_move_context",
            games_relation="main.stg_games",
        ),
        versions=ModelingVersionConfig(
            modeling_pipeline_version=MODELING_PIPELINE_VERSION,
            split_definition_version=SPLIT_DEFINITION_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            label_definition_version=LABEL_DEFINITION_VERSION,
            schema_version="1.1.0",
            position_normalization_version="fen4_legal_ep_v1",
            board_encoding_version="board18_abs_v1",
            action_encoding_version="action8x8x73_v1",
        ),
        sampling=ModelingSamplingConfig(
            rule_version="sha256_mod_v1",
            seed="test-game-sampling-seed",
            hash_modulus=10000,
            hash_threshold=10000,
            requested_rate_percent=100.0,
            max_games=None,
        ),
        splits=TemporalSplitConfig(
            train=TemporalRange(start_date=date(2017, 1, 1), end_date=date(2017, 1, 20)),
            validation=TemporalRange(start_date=date(2017, 1, 21), end_date=date(2017, 1, 25)),
            test=TemporalRange(start_date=date(2017, 1, 26), end_date=date(2017, 1, 31)),
            missing_or_invalid_date_policy="assign_train",
        ),
        player_holdout=PlayerHoldoutConfig(
            rule_version="sha256_mod_v1",
            seed="test-player-holdout-seed",
            hash_modulus=10000,
            hash_threshold=holdout_threshold,
            requested_rate_percent=10.0,
            missing_player_hash_policy=missing_player_hash_policy,  # type: ignore[arg-type]
        ),
        output=ModelingOutputConfig(
            output_root=Path("data/modeling"),
            batch_rows=128,
            max_examples=None,
            parquet_compression="zstd",
            parquet_row_group_size=256,
        ),
        behavior=ModelingBehaviorConfig(strict=True),
    )


def _append_one_assignment(
    *,
    config: ModelingConfig,
    game_id: str,
    white_player_hash: str | None,
    black_player_hash: str | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    _append_assignment_candidate(
        candidate={
            "game_id": game_id,
            "source_month": "2017-01",
            "played_date_raw": "2017.01.05",
            "played_date_for_split": "2017-01-05",
            "played_date_source": "date",
            "result": "1-0",
            "white_player_hash": white_player_hash,
            "black_player_hash": black_player_hash,
            "white_rating": 1600,
            "black_rating": 1500,
            "time_control_raw": "300+0",
            "eco": "C20",
            "opening": "King Pawn Game",
            "ply_count": 40,
            "sample_score_u64": 123,
        },
        config=config,
        out_buffer=rows,
        game_rejections=Counter(),
        sampling_stats=Counter(),
        date_source_counts=Counter(),
    )
    assert len(rows) == 1
    return rows[0]


def _create_assignment_tables(
    connection: duckdb.DuckDBPyConnection,
    assignments: list[tuple[str, str | None, str | None, bool, bool, bool, bool]],
) -> None:
    connection.execute(
        """
        CREATE TABLE assignments (
            game_id VARCHAR,
            white_player_hash VARCHAR,
            black_player_hash VARCHAR,
            white_player_is_holdout BOOLEAN,
            black_player_is_holdout BOOLEAN,
            is_player_holdout_game BOOLEAN,
            player_disjoint_training_eligible BOOLEAN
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        assignments,
    )

    connection.execute(
        """
        CREATE TABLE policy_examples (
            game_id VARCHAR,
            temporal_split VARCHAR,
            policy_target_action_index INTEGER
        )
        """
    )
    policy_rows = [(row[0], "train", 120) for row in assignments]
    connection.executemany(
        "INSERT INTO policy_examples VALUES (?, ?, ?)",
        policy_rows,
    )


def test_false_positive_opponent_is_not_treated_as_holdout_leak() -> None:
    connection = duckdb.connect()
    try:
        _create_assignment_tables(
            connection,
            assignments=[
                ("g_ab_holdout", "player_a", "player_b", True, False, True, False),
                ("g_bc_train", "player_b", "player_c", False, False, False, True),
            ],
        )
        _validate_split_and_holdout_constraints(
            connection=connection,
            assignments_sql="SELECT * FROM assignments",
            primary_policy_sql="SELECT * FROM policy_examples",
        )
    finally:
        connection.close()


def test_true_holdout_leak_still_fails_validation() -> None:
    connection = duckdb.connect()
    try:
        _create_assignment_tables(
            connection,
            assignments=[
                ("g_ab_holdout", "player_a", "player_b", True, False, True, False),
                ("g_ad_illegal", "player_a", "player_d", True, False, True, True),
            ],
        )
        with pytest.raises(
            RuntimeError,
            match="held-out player hash leaked into player-disjoint training population",
        ):
            _validate_split_and_holdout_constraints(
                connection=connection,
                assignments_sql="SELECT * FROM assignments",
                primary_policy_sql="SELECT * FROM policy_examples",
            )
    finally:
        connection.close()


def test_assignment_flags_when_both_players_are_holdout() -> None:
    config = _make_config(holdout_threshold=10000)
    row = _append_one_assignment(
        config=config,
        game_id="g_both_holdout",
        white_player_hash="player_a",
        black_player_hash="player_b",
    )

    assert row["white_player_is_holdout"] is True
    assert row["black_player_is_holdout"] is True
    assert row["is_player_holdout_game"] is True
    assert row["player_disjoint_training_eligible"] is False


def test_assignment_flags_when_neither_player_is_holdout() -> None:
    config = _make_config(holdout_threshold=0)
    row = _append_one_assignment(
        config=config,
        game_id="g_neither_holdout",
        white_player_hash="player_x",
        black_player_hash="player_y",
    )

    assert row["white_player_is_holdout"] is False
    assert row["black_player_is_holdout"] is False
    assert row["is_player_holdout_game"] is False
    assert row["player_disjoint_training_eligible"] is True


def test_missing_player_hash_exclusion_policy_marks_row_ineligible() -> None:
    config = _make_config(
        holdout_threshold=0,
        missing_player_hash_policy="exclude_from_player_disjoint_training",
    )
    row = _append_one_assignment(
        config=config,
        game_id="g_missing_black_hash",
        white_player_hash="player_x",
        black_player_hash=None,
    )

    assert row["white_player_is_holdout"] is False
    assert row["black_player_is_holdout"] is False
    assert row["is_player_holdout_game"] is False
    assert row["player_disjoint_training_eligible"] is False


def test_assignment_holdout_flags_are_deterministic_for_identical_inputs() -> None:
    config = _make_config(holdout_threshold=1000)

    first = _append_one_assignment(
        config=config,
        game_id="g_deterministic",
        white_player_hash="player_white",
        black_player_hash="player_black",
    )
    second = _append_one_assignment(
        config=config,
        game_id="g_deterministic",
        white_player_hash="player_white",
        black_player_hash="player_black",
    )

    assert first["white_player_is_holdout"] == second["white_player_is_holdout"]
    assert first["black_player_is_holdout"] == second["black_player_is_holdout"]
    assert first["is_player_holdout_game"] == second["is_player_holdout_game"]
    assert (
        first["player_disjoint_training_eligible"]
        == second["player_disjoint_training_eligible"]
    )
