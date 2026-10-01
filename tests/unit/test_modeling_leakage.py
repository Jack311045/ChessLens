from __future__ import annotations

import duckdb

from chesslens.modeling.leakage import (
    compute_novel_position_test_count,
    compute_position_overlap_metrics,
)


def test_compute_position_overlap_metrics() -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            """
            CREATE TABLE examples AS
            SELECT * FROM (
                VALUES
                    ('train', 'pos_a', 0),
                    ('train', 'pos_b', 1),
                    ('validation', 'pos_b', 2),
                    ('test', 'pos_b', 3),
                    ('test', 'pos_c', 4)
            ) AS t(temporal_split, position_id, ply)
            """
        )

        relation_sql = "SELECT * FROM examples"
        metrics = compute_position_overlap_metrics(connection=connection, relation_sql=relation_sql)
        novel_count = compute_novel_position_test_count(
            connection=connection, relation_sql=relation_sql
        )

        assert metrics["split_counts"]["train"]["rows"] == 2
        assert metrics["split_counts"]["test"]["rows"] == 2
        assert metrics["position_overlaps"]["train_test"]["distinct_position_overlap"] == 1
        assert novel_count == 1
    finally:
        connection.close()
