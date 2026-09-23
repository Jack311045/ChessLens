with source as (
    select *
    from {{ source('bronze', 'bronze_ingestion_errors') }}
)

select
    cast(source_archive as varchar) as source_archive,
    cast(source_month as varchar) as source_month,
    cast(source_game_index as bigint) as source_game_index,
    cast(error_code as varchar) as error_code,
    cast(error_message as varchar) as error_message,
    cast(handling_decision as varchar) as handling_decision,
    cast(recoverability as varchar) as recoverability,
    cast(run_id as varchar) as run_id,
    cast(schema_version as varchar) as schema_version
from source
