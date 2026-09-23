with source as (
    select *
    from {{ source('bronze', 'bronze_manifest') }}
)

select
    cast(dataset_id as varchar) as dataset_id,
    cast(configuration_hash as varchar) as configuration_hash,
    cast(status as varchar) as status,
    cast(manifest_version as varchar) as manifest_version,
    cast(run_id as varchar) as run_id,
    cast(started_at_utc as varchar) as started_at_utc,
    cast(finished_at_utc as varchar) as finished_at_utc,
    cast(source.source_month as varchar) as source_month,
    cast(source.archive_filename as varchar) as source_archive_filename,
    cast(source.archive_sha256 as varchar) as source_archive_sha256,
    cast(counts.accepted_games as bigint) as accepted_games,
    cast(counts.rejected_games as bigint) as rejected_games,
    cast(counts.emitted_moves as bigint) as emitted_moves,
    cast(counts.error_records as bigint) as error_records,
    cast(datasets.games.row_count as bigint) as games_dataset_row_count,
    cast(datasets.moves.row_count as bigint) as moves_dataset_row_count,
    cast(datasets.ingestion_errors.row_count as bigint) as ingestion_errors_dataset_row_count
from source
