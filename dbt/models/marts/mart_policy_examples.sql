select
    game_id,
    ply,
    source_month,
    position_id,
    pre_move_fen,
    normalized_pre_move_fen,
    side_to_move,
    time_control_raw,
    case
        when side_to_move = 'w' then white_rating
        when side_to_move = 'b' then black_rating
        else null
    end as mover_rating,
    case
        when side_to_move = 'w' then black_rating
        when side_to_move = 'b' then white_rating
        else null
    end as opponent_rating,
    case
        when side_to_move = 'w' then white_rating - black_rating
        when side_to_move = 'b' then black_rating - white_rating
        else null
    end as rating_difference,
    played_move_uci
from {{ ref('int_move_context') }}
