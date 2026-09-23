with per_game as (
    select
        game_id,
        min(ply) as min_ply,
        max(ply) as max_ply,
        count(*) as move_rows,
        count(distinct ply) as distinct_ply_rows
    from {{ ref('stg_moves') }}
    group by 1
)

select
    game_id,
    min_ply,
    max_ply,
    move_rows,
    distinct_ply_rows
from per_game
where min_ply <> 0
   or max_ply + 1 <> move_rows
   or distinct_ply_rows <> move_rows
