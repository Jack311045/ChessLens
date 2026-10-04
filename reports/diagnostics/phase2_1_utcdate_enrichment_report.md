# Phase 2.1 UTCDate Enrichment Report

## Scope
- Root-cause diagnosis for temporal split failures caused by invalid PGN `Date` headers.
- UTCDate-first canonical date resolver implementation.
- Resumable, header-only sidecar enrichment pipeline (no move re-ingestion, no bronze rewrite).
- Phase 2.1 modeling integration with explicit enrichment manifest input and fail-fast guards.

## Phase A Evidence
- Diagnostic artifact: `reports/diagnostics/phase2_1_utcdate_header_diagnostic_2017_sample.json`
- Real-data sampling method: first, middle, and last 2017 shard; 3000 games per shard.
- Result summary:
  - `Date` header coverage: 100% present but 0% valid in sampled rows.
  - `UTCDate` header coverage: 100% present and 100% valid in sampled rows.
  - `Date invalid AND UTCDate valid`: 100% in all sampled shards.
- Representative sampled values showed `Date="????.??.??"` with valid `UTCDate` calendar dates.

## Implemented Changes
- Canonical resolver module:
  - `src/chesslens/ingestion/date_resolution.py`
  - policy version: `utc_date_first_v1`
  - rule: valid `UTCDate` -> valid `Date` -> null.
- Resumable sidecar pipeline:
  - `src/chesslens/ingestion/date_enrichment.py`
  - command: `python -m chesslens.ingestion.date_enrichment`
  - atomic manifest publication and checksum-validated artifact reuse.
- Modeling integration:
  - explicit input path support (`input.date_enrichment_manifest_path`, CLI/env override).
  - optional strict requirement flag (`input.require_date_enrichment`).
  - sidecar join by `game_id` before split assignment.
  - fail-fast checks for duplicate enrichment keys and missing enrichment coverage.
  - audit-preserved upstream `played_date_raw`, plus canonical `played_date_iso` and `played_date_source`.
  - identity payload includes enrichment identity/hash/version when enabled.

## Quality Gates
- `uv sync --frozen --dev`
- `uv run ruff check .`
- `uv run mypy src tests scripts`
- `uv run pytest -q`

All quality gates completed successfully.

## Notes
- Existing published bronze/collection datasets are not rewritten.
- Stable `game_id`, player hash, and encoding lineage are preserved.
