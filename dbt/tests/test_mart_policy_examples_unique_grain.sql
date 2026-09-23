select
    game_id,
    ply,
    count(*) as row_count
from {{ ref('mart_policy_examples') }}
group by 1, 2
having count(*) > 1
