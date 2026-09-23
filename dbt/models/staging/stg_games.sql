with source as (
    select *
    from {{ source('bronze', 'bronze_games') }}
)

select
    cast(game_id as varchar) as game_id,
    cast(source_archive as varchar) as source_archive,
    cast(source_archive_sha256 as varchar) as source_archive_sha256,
    cast(source_month as varchar) as source_month,
    cast(source_game_index as bigint) as source_game_index,
    cast(event as varchar) as event,
    cast(played_date as varchar) as played_date,
    cast(round as varchar) as round,
    cast(result as varchar) as result,
    cast(white_rating as integer) as white_rating,
    cast(black_rating as integer) as black_rating,
    cast(time_control_raw as varchar) as time_control_raw,
    cast(eco as varchar) as eco,
    cast(opening as varchar) as opening,
    cast(termination as varchar) as termination,
    cast(white_player_hash as varchar) as white_player_hash,
    cast(black_player_hash as varchar) as black_player_hash,
    cast(ply_count as integer) as ply_count,
    cast(schema_version as varchar) as schema_version
from source
