select
    m.game_id,
    m.ply,
    m.source_month,
    m.position_id,
    m.pre_move_fen,
    m.normalized_pre_move_fen,
    m.side_to_move,
    m.played_move_uci,
    m.played_move_san,
    m.halfmove_clock,
    m.fullmove_number,
    m.clock_annotation_raw,
    g.white_rating,
    g.black_rating,
    g.time_control_raw,
    g.eco,
    g.opening,
    g.result,
    g.termination,
    g.ply_count as game_ply_count,
    g.schema_version
from {{ ref('stg_moves') }} as m
inner join {{ ref('stg_games') }} as g on m.game_id = g.game_id
