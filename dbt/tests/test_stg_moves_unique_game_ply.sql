select
    game_id,
    ply,
    count(*) as row_count
from {{ ref('stg_moves') }}
group by 1, 2
having count(*) > 1
