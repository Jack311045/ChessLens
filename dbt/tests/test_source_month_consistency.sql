select
    m.game_id,
    g.source_month as game_source_month,
    m.source_month as move_source_month
from {{ ref('stg_moves') }} as m
inner join {{ ref('stg_games') }} as g on m.game_id = g.game_id
where m.source_month <> g.source_month
