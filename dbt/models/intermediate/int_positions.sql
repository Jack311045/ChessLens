with move_events as (
    select
        m.position_id,
        m.normalized_pre_move_fen as normalized_fen,
        m.side_to_move,
        g.source_month
    from {{ ref('stg_moves') }} as m
    inner join {{ ref('stg_games') }} as g on m.game_id = g.game_id
),
parsed as (
    select
        position_id,
        normalized_fen,
        side_to_move,
        split_part(normalized_fen, ' ', 3) as castling_rights,
        nullif(split_part(normalized_fen, ' ', 4), '-') as en_passant_square,
        source_month
    from move_events
)

select
    position_id,
    min(normalized_fen) as normalized_fen,
    min(side_to_move) as side_to_move,
    min(castling_rights) as castling_rights,
    min(en_passant_square) as en_passant_square,
    count(*) as occurrence_count,
    min(source_month) as first_seen_source_month,
    max(source_month) as last_seen_source_month
from parsed
group by 1
