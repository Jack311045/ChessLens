# Phase 1.2b DuckDB + dbt Walkthrough

Audience: developers who already understand Phase 1.2a ingestion and want to reason clearly about SQL lineage, grain, and leakage-safe marts.

## 1. What Phase 1.2b Adds

Phase 1.2b turns published parquet bronze outputs into reproducible analytics/training tables with dbt.

Core additions:

- deterministic preflight checks before SQL runs;
- source registration into DuckDB (`bronze_*` views);
- staged model layers (staging -> intermediate -> marts);
- SQL tests for keys, grain, reconciliation, and leakage boundaries.

## 2. Why dbt Instead of Ad-hoc SQL Scripts

dbt gives:

- explicit model lineage (`ref` and `source`);
- first-class tests and docs;
- repeatable DAG execution in CI;
- SQL transformations that stay reviewable in Git.

## 3. Preflight: Fail Fast Before Warehouse Build

Before dbt, run:

```bash
uv run python -m chesslens.warehouse.preflight
```

Preflight checks manifest completeness and row-count reconciliation against parquet, then registers DuckDB views.

## 4. Source Layer: Bronze Views

The bronze source is registered in DuckDB, then referenced in dbt as real `source()` relations.

```sql
select count(*) as bronze_games_rows
from {{ source('bronze', 'bronze_games') }}
```

```sql
select count(*) as bronze_moves_rows
from {{ source('bronze', 'bronze_moves') }}
```

```sql
select status, counts.accepted_games
from {{ source('bronze', 'bronze_manifest') }}
```

## 5. Staging Models: Type-Safe, Explicit Columns

`stg_games` keeps one row per game and normalizes types.

```sql
select
    cast(game_id as varchar) as game_id,
    cast(source_month as varchar) as source_month,
    cast(result as varchar) as result,
    cast(ply_count as integer) as ply_count
from {{ source('bronze', 'bronze_games') }}
```

`stg_moves` keeps one row per move at (`game_id`, `ply`) grain.

```sql
select
    cast(game_id as varchar) as game_id,
    cast(ply as integer) as ply,
    cast(position_id as varchar) as position_id,
    cast(side_to_move as varchar) as side_to_move,
    cast(played_move_uci as varchar) as played_move_uci,
    cast(played_move_san as varchar) as played_move_san
from {{ source('bronze', 'bronze_moves') }}
```

`stg_manifest` flattens nested JSON fields so tests can reconcile counts.

```sql
select
    cast(dataset_id as varchar) as dataset_id,
    cast(counts.accepted_games as bigint) as accepted_games,
    cast(counts.emitted_moves as bigint) as emitted_moves,
    cast(counts.error_records as bigint) as error_records
from {{ source('bronze', 'bronze_manifest') }}
```

## 6. Intermediate Models: Global Position Semantics + Move Context

`int_positions` deduplicates positions globally and derives FEN features.

```sql
select
    position_id,
    min(normalized_pre_move_fen) as normalized_fen,
    min(side_to_move) as side_to_move,
    split_part(min(normalized_pre_move_fen), ' ', 3) as castling_rights,
    nullif(split_part(min(normalized_pre_move_fen), ' ', 4), '-') as en_passant_square,
    count(*) as occurrence_count
from {{ ref('stg_moves') }}
group by 1
```

`int_move_context` joins move rows with game metadata without changing grain.

```sql
select
    m.game_id,
    m.ply,
    m.played_move_uci,
    g.result,
    g.termination,
    g.white_rating,
    g.black_rating
from {{ ref('stg_moves') }} m
join {{ ref('stg_games') }} g on m.game_id = g.game_id
```

## 7. Mart Models: Separate Analytics Labels from Training Inputs

`fct_move_events` is analytics-focused and includes outcome labels.

```sql
select
    game_id,
    ply,
    played_move_uci,
    result as final_result_label,
    termination as termination_label
from {{ ref('int_move_context') }}
```

`mart_policy_examples` is training-safe and intentionally excludes post-outcome columns.

```sql
select
    game_id,
    ply,
    position_id,
    normalized_pre_move_fen,
    side_to_move,
    case when side_to_move = 'w' then white_rating else black_rating end as mover_rating,
    case when side_to_move = 'w' then black_rating else white_rating end as opponent_rating,
    played_move_uci
from {{ ref('int_move_context') }}
```

## 8. Key Tests and Why They Matter

Uniqueness at move grain:

```sql
select game_id, ply, count(*)
from {{ ref('stg_moves') }}
group by 1, 2
having count(*) > 1
```

Contiguous zero-based plies:

```sql
with per_game as (
  select game_id, min(ply) as min_ply, max(ply) as max_ply, count(*) as rows
  from {{ ref('stg_moves') }}
  group by 1
)
select *
from per_game
where min_ply <> 0 or max_ply + 1 <> rows
```

Manifest reconciliation:

```sql
with observed as (
  select
    (select count(*) from {{ ref('stg_games') }}) as game_rows,
    (select count(*) from {{ ref('stg_moves') }}) as move_rows,
    (select count(*) from {{ ref('stg_ingestion_errors') }}) as error_rows
)
select *
from {{ ref('stg_manifest') }} m
cross join observed o
where m.accepted_games <> o.game_rows
   or m.emitted_moves <> o.move_rows
   or m.error_records <> o.error_rows
```

Leakage guard test (forbidden columns absent in policy mart):

```sql
select column_name
from information_schema.columns
where table_name = '{{ ref('mart_policy_examples').identifier }}'
  and lower(column_name) in (
      'result',
      'final_result_label',
      'termination',
      'termination_label',
      'ply_count',
      'game_ply_count'
  )
```

## 9. Typical Local Run Sequence

Linux/macOS:

```bash
export CHESSLENS_DATASET_ROOT=data/processed/datasets/<dataset_id>
uv run python -m chesslens.warehouse.preflight
uv run dbt debug --project-dir dbt --profiles-dir dbt
uv run dbt compile --project-dir dbt --profiles-dir dbt
uv run dbt build --project-dir dbt --profiles-dir dbt
uv run dbt docs generate --project-dir dbt --profiles-dir dbt
```

PowerShell:

```powershell
$env:CHESSLENS_DATASET_ROOT = "data/processed/datasets/<dataset_id>"
python -m uv run python -m chesslens.warehouse.preflight
python -m uv run dbt debug --project-dir dbt --profiles-dir dbt
python -m uv run dbt compile --project-dir dbt --profiles-dir dbt
python -m uv run dbt build --project-dir dbt --profiles-dir dbt
python -m uv run dbt docs generate --project-dir dbt --profiles-dir dbt
```

## 10. Progression to Phase 2 Modeling

With this warehouse layer in place:

- `fct_move_events` supports exploratory analysis and feature diagnostics;
- `mart_policy_examples` is ready for supervised move-policy datasets;
- tests enforce reproducibility and leakage boundaries before model training begins.
