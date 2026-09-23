with comparisons as (
    select
        'games' as dataset_name,
        (select count(*) from {{ source('bronze', 'bronze_games') }}) as bronze_rows,
        (select count(*) from {{ ref('stg_games') }}) as staging_rows

    union all

    select
        'moves' as dataset_name,
        (select count(*) from {{ source('bronze', 'bronze_moves') }}) as bronze_rows,
        (select count(*) from {{ ref('stg_moves') }}) as staging_rows

    union all

    select
        'ingestion_errors' as dataset_name,
        (select count(*) from {{ source('bronze', 'bronze_ingestion_errors') }}) as bronze_rows,
        (select count(*) from {{ ref('stg_ingestion_errors') }}) as staging_rows
)

select *
from comparisons
where bronze_rows <> staging_rows
