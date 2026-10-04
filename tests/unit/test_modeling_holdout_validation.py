from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from chesslens.modeling.build_dataset import (
    _is_holdout_player,
    _validate_split_and_holdout_constraints,
)
from chesslens.modeling.config import (
    MissingPlayerHashPolicy,
    ModelingConfig,
    load_modeling_config,
)


def _holdout_config(
    *,
    missing_player_hash_policy: MissingPlayerHashPolicy,
) -> ModelingConfig:
    config = load_modeling_config(Path("configs/modeling/fixture.yaml"))
    player_holdout = replace(
        config.player_holdout,
        seed="unit-holdout-seed",
        hash_modulus=10,
        hash_threshold=3,
        requested_rate_percent=30.0,
        missing_player_hash_policy=missing_player_hash_policy,
    )
    return replace(config, player_holdout=player_holdout)


def _find_player_hashes(config: ModelingConfig) -> tuple[str, str, str]:
    holdout_player: str | None = None
    non_holdout_players: list[str] = []

    for idx in range(10000):
        player_hash = f"player_{idx}"
        if _is_holdout_player(config, player_hash):
            if holdout_player is None:
                holdout_player = player_hash
        elif len(non_holdout_players) < 2:
            non_holdout_players.append(player_hash)

        if holdout_player is not None and len(non_holdout_players) == 2:
            break

    if holdout_player is None or len(non_holdout_players) < 2:
        raise AssertionError("Unable to find deterministic holdout/non-holdout players")

    return holdout_player, non_holdout_players[0], non_holdout_players[1]


def _run_validator(
    *,
    config: ModelingConfig,
    assignments: Sequence[tuple[str, str, bool, bool, str | None, str | None]],
) -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            """
            CREATE TABLE assignments (
                game_id VARCHAR,
                temporal_split VARCHAR,
                player_disjoint_training_eligible BOOLEAN,
                is_player_holdout_game BOOLEAN,
                white_player_hash VARCHAR,
                black_player_hash VARCHAR
            )
            """
        )
        connection.executemany(
            "INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?)",
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
        policy_rows = [(row[0], row[1], 7) for row in assignments]
        connection.executemany(
            "INSERT INTO policy_examples VALUES (?, ?, ?)",
            policy_rows,
        )

        _validate_split_and_holdout_constraints(
            connection=connection,
            assignments_sql="SELECT * FROM assignments",
            primary_policy_sql="SELECT * FROM policy_examples",
            config=config,
        )
    finally:
        connection.close()


def test_holdout_vs_non_holdout_and_non_holdout_vs_non_holdout_passes() -> None:
    config = _holdout_config(
        missing_player_hash_policy="exclude_from_player_disjoint_training"
    )
    holdout_player, normal_b, normal_c = _find_player_hashes(config)

    assignments = [
        ("g_holdout_vs_normal", "train", False, True, holdout_player, normal_b),
        ("g_normal_vs_normal", "train", True, False, normal_b, normal_c),
    ]

    _run_validator(config=config, assignments=assignments)


def test_validator_fails_when_holdout_player_is_marked_eligible_in_train() -> None:
    config = _holdout_config(
        missing_player_hash_policy="exclude_from_player_disjoint_training"
    )
    holdout_player, normal_b, _ = _find_player_hashes(config)

    assignments = [
        ("g_leak", "train", True, True, holdout_player, normal_b),
    ]

    with pytest.raises(
        RuntimeError,
        match="held-out player hash leaked into player-disjoint training population",
    ):
        _run_validator(config=config, assignments=assignments)


def test_validator_fails_when_game_holdout_flag_mismatches_players() -> None:
    config = _holdout_config(
        missing_player_hash_policy="exclude_from_player_disjoint_training"
    )
    holdout_player, normal_b, _ = _find_player_hashes(config)

    assignments = [
        ("g_bad_holdout_flag", "train", False, False, holdout_player, normal_b),
    ]

    with pytest.raises(
        RuntimeError,
        match="is_player_holdout_game does not match per-player holdout eligibility",
    ):
        _run_validator(config=config, assignments=assignments)


def test_validator_respects_missing_player_policy() -> None:
    exclude_config = _holdout_config(
        missing_player_hash_policy="exclude_from_player_disjoint_training"
    )
    _, normal_b, _ = _find_player_hashes(exclude_config)

    invalid_assignments = [
        ("g_missing_player", "train", True, False, normal_b, None),
    ]
    with pytest.raises(
        RuntimeError,
        match="player_disjoint_training_eligible does not match recomputed player-holdout policy",
    ):
        _run_validator(config=exclude_config, assignments=invalid_assignments)

    include_config = _holdout_config(
        missing_player_hash_policy="include_in_player_disjoint_training"
    )
    _, include_normal_b, _ = _find_player_hashes(include_config)
    valid_assignments = [
        ("g_missing_player", "train", True, False, include_normal_b, None),
    ]
    _run_validator(config=include_config, assignments=valid_assignments)
