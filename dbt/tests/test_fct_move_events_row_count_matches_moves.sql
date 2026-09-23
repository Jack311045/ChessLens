with counts as (
    select
        (select count(*) from {{ ref('fct_move_events') }}) as fact_rows,
        (select count(*) from {{ ref('stg_moves') }}) as move_rows
)

select *
from counts
where fact_rows <> move_rows
