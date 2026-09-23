with observed as (
    select
        (select count(*) from {{ ref('stg_games') }}) as game_rows,
        (select count(*) from {{ ref('stg_moves') }}) as move_rows,
        (select count(*) from {{ ref('stg_ingestion_errors') }}) as error_rows
),
manifest as (
    select
        accepted_games,
        emitted_moves,
        error_records,
        games_dataset_row_count,
        moves_dataset_row_count,
        ingestion_errors_dataset_row_count
    from {{ ref('stg_manifest') }}
)

select
    m.accepted_games,
    m.emitted_moves,
    m.error_records,
    m.games_dataset_row_count,
    m.moves_dataset_row_count,
    m.ingestion_errors_dataset_row_count,
    o.game_rows,
    o.move_rows,
    o.error_rows
from manifest as m
cross join observed as o
where m.accepted_games <> o.game_rows
   or m.emitted_moves <> o.move_rows
   or m.error_records <> o.error_rows
   or m.games_dataset_row_count <> o.game_rows
   or m.moves_dataset_row_count <> o.move_rows
   or m.ingestion_errors_dataset_row_count <> o.error_rows
