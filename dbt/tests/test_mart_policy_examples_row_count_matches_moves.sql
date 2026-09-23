with counts as (
    select
        (select count(*) from {{ ref('mart_policy_examples') }}) as mart_rows,
        (select count(*) from {{ ref('stg_moves') }}) as move_rows
)

select *
from counts
where mart_rows <> move_rows
