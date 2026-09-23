with source as (
    select *
    from {{ source('bronze', 'bronze_moves') }}
)

select
    cast(game_id as varchar) as game_id,
    cast(source_month as varchar) as source_month,
    cast(ply as integer) as ply,
    cast(pre_move_fen as varchar) as pre_move_fen,
    cast(normalized_pre_move_fen as varchar) as normalized_pre_move_fen,
    cast(position_id as varchar) as position_id,
    cast(side_to_move as varchar) as side_to_move,
    cast(played_move_uci as varchar) as played_move_uci,
    cast(played_move_san as varchar) as played_move_san,
    cast(halfmove_clock as integer) as halfmove_clock,
    cast(fullmove_number as integer) as fullmove_number,
    cast(clock_annotation_raw as varchar) as clock_annotation_raw,
    cast(schema_version as varchar) as schema_version
from source
