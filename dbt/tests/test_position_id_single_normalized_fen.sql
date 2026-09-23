select
    position_id,
    count(distinct normalized_pre_move_fen) as distinct_fen_count
from {{ ref('stg_moves') }}
group by 1
having count(distinct normalized_pre_move_fen) > 1
