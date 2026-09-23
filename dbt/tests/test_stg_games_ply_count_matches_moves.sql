with observed as (
    select
        game_id,
        count(*) as observed_ply_count
    from {{ ref('stg_moves') }}
    group by 1
)

select
    g.game_id,
    g.ply_count as expected_ply_count,
    o.observed_ply_count
from {{ ref('stg_games') }} as g
inner join observed as o on g.game_id = o.game_id
where g.ply_count <> o.observed_ply_count
