select
    game_id,
    ply,
    count(*) as row_count
from {{ ref('fct_move_events') }}
group by 1, 2
having count(*) > 1
