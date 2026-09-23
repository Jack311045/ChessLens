# Architecture Decisions (Phase 0 and 1.1)

This file records early decisions using a lightweight ADR style: context, decision, consequences, alternatives.

## ADR-001: src package layout

Context:
- We need reusable code for ingestion, validation, and feature encoding that can scale to later ETL/training phases.

Decision:
- Use `src/chesslens/` package layout with subpackages for `domain`, `ingestion`, `validation`, and `features`.

Consequences:
- Import behavior in tests mirrors installed package behavior.
- Clear boundaries reduce accidental notebook-style coupling.

Alternatives considered:
- Flat top-level modules.
- Notebook-first implementation.

Why not chosen:
- Harder to scale, test, and type-check reliably.

## ADR-002: uv-managed reproducible environment

Context:
- Reproducibility and deterministic dependency resolution are mandatory for portfolio credibility.

Decision:
- Use `uv` with `pyproject.toml` and committed `uv.lock`.

Consequences:
- Fast, deterministic installs across machines and CI.
- Dependency provenance is explicit.

Alternatives considered:
- Hand-maintained `requirements.txt` only.
- Poetry/Pipenv.

Why not chosen:
- `requirements.txt` alone does not hold full project metadata or dependency groups.

## ADR-003: raw data excluded from Git

Context:
- Lichess archives are large and not suitable for source control.

Decision:
- Ignore `data/raw/` and broad large artifact patterns in `.gitignore`.

Consequences:
- Repository remains lightweight and reviewable.
- CI never depends on external large downloads.

Alternatives considered:
- Commit raw archive.

Why not chosen:
- Violates repository hygiene and practical collaboration constraints.

## ADR-004: staged local archive + streaming decompression

Context:
- The source file is compressed and potentially very large.

Decision:
- Standardize local raw archive path under `data/raw/` and stream with `zstandard.ZstdDecompressor().stream_reader(...)` + `io.TextIOWrapper`.

Consequences:
- Memory usage is bounded by stream buffers, not full file size.
- Same code path supports both tiny and large archives.

Alternatives considered:
- Full decompression to disk first.
- `read()` entire decompressed content into RAM.

Why not chosen:
- Wastes I/O and memory, and does not scale to later phases.

## ADR-005: FEN normalization version `fen4_legal_ep_v1`

Context:
- Position identity must be stable and match legal en-passant semantics.

Decision:
- Normalize to first four FEN fields with legal en-passant representation from `python-chess`.

Consequences:
- Stable cross-game position identity suitable for joins.
- Halfmove/fullmove counters kept separately when needed.

Alternatives considered:
- Keep full six-field FEN identity.
- Ignore en-passant square fully.

Why not chosen:
- Six fields over-distinguish identical board states for many analytics; removing en-passant entirely can merge positions that differ legally.

## ADR-006: game and position identity strategy

Context:
- IDs must be deterministic and source-stable.

Decision:
- `game_id = sha256(source_archive_sha256 + source_game_index)`.
- `position_id = sha256(normalization_version + normalized_fen)`.

Consequences:
- IDs are reproducible and independent of local filenames.
- Supports idempotent ingestion and re-runs.

Alternatives considered:
- Use player names or site URLs.
- Use random UUIDs.

Why not chosen:
- Player/site can expose PII and are not stable IDs.
- Random IDs break determinism and reproducibility.

## ADR-007: zero-based pre-move ply convention

Context:
- Prediction examples represent the board before the played move.

Decision:
- Use zero-based `ply` where `ply=0` is White move 1.
- Capture board/FEN before `board.push(move)`.

Consequences:
- Label/feature separation is explicit.
- Leakage from post-move state is easier to avoid.

Alternatives considered:
- One-based ply indexing.
- Post-move state as primary row state.

Why not chosen:
- Harder to reason about prediction point and leakage boundaries.

## ADR-008: absolute orientation board encoding

Context:
- First model version should maximize interpretability and debugging ease.

Decision:
- Board tensor uses absolute orientation with file `a` index 0 and rank `1` index 0.

Consequences:
- Direct map from FEN to tensor indices for debugging.
- Later experiments can compare with side-relative orientation.

Alternatives considered:
- Side-to-move-relative orientation.

Why not chosen:
- Less intuitive for early validation and documentation.

## ADR-009: fixed `8 x 8 x 73` action encoding

Context:
- Policy heads require fixed output shape across positions.

Decision:
- Use AlphaZero-style plane encoding with 4672 global actions and legal masking.

Consequences:
- Neural policy outputs can be dense tensors with sparse legality masks.
- Round-trip tests and collision checks are required for trustworthiness.

Alternatives considered:
- Variable-length legal move lists.

Why not chosen:
- Harder batching and model-output alignment for supervised policy training.

## ADR-010: thin orchestration around tested library code

Context:
- Airflow is valuable later but orchestration should not hold core game logic.

Decision:
- Keep parsing/encoding/validation in importable library code; future Airflow tasks will call these functions.

Consequences:
- Core logic remains testable without scheduler runtime.
- Lower coupling and better local development velocity.

Alternatives considered:
- Place most logic directly in DAG task callables.

Why not chosen:
- Harder to test, reuse, and type-check.

## ADR-011: defer substantive dbt until real ingestion outputs exist

Context:
- dbt models need actual staged sources and known grain.

Decision:
- Create dbt skeleton and documentation only in this PR.

Consequences:
- Avoid fake SQL transformations disconnected from produced data.
- Next ETL PR can add real lineage-backed models.

Alternatives considered:
- Create placeholder SQL with invented sources.

Why not chosen:
- Creates misleading project evidence and maintenance burden.

## ADR-012: publish only bronze games/moves/errors in Phase 1.2a

Context:
- We need stable physical Parquet outputs now, but globally deduplicated positions require SQL-level global deduplication logic.

Decision:
- Publish only `games`, `moves`, and `ingestion_errors` as bronze datasets in Phase 1.2a.
- Defer globally deduplicated `int_positions` to Phase 1.2b dbt models built from bronze `moves`.

Consequences:
- Physical outputs remain truthful about grain and deduplication scope.
- Phase 1.2b can implement and test true global deduplication in SQL.

Alternatives considered:
- Publish a Phase 1.2a `positions` table deduplicated only inside games or batches.

Why not chosen:
- It would imply global uniqueness that is not actually guaranteed.

## ADR-013: stage-validate-publish for safe idempotent datasets

Context:
- A failed ingestion run must never look like a complete dataset to downstream readers.

Decision:
- Write to a unique staging path first.
- Validate row counts, keys, contiguity, foreign keys, partition values, and schemas.
- Write complete manifest only after validation passes.
- Publish by renaming staging dataset into deterministic final dataset path.
- Reuse an existing completed dataset with the same dataset_id after validating it.

Consequences:
- Failed runs do not modify completed datasets.
- Repeated identical runs avoid duplicate writes.
- Downstream readers can safely read only `datasets/<dataset_id>` outputs.

Alternatives considered:
- Write directly into final path and append parts as processing proceeds.

Why not chosen:
- Leaves partially written datasets visible and increases corruption risk.

## ADR-014: explicit player hash modes with required HMAC env inputs

Context:
- Fixture data can use deterministic placeholders, but real archives require keyed anonymization.

Decision:
- Support explicit `player_hash_mode` values:
	- `fixture_placeholder` for sanitized fixture workflows.
	- `hmac_sha256` for real-data ingestion.
- Require HMAC secret/key-id environment variables for HMAC mode and fail early if missing.

Consequences:
- Real-data ingestion enforces keyed anonymization policy.
- Secrets are never serialized into manifests, dataset IDs, or paths.

Alternatives considered:
- Infer security mode from file names or environment heuristics.

Why not chosen:
- Hidden policy switching is brittle and unsafe.

## ADR-015: dataset identity is logical and pipeline-versioned

Context:
- Output-affecting ingestion logic can change without Arrow schema changes.
- Git commits include documentation/test-only changes that must not force new dataset IDs.

Decision:
- Add `INGESTION_PIPELINE_VERSION` (currently `parquet_etl_v1`).
- Include pipeline version in effective config, configuration hash inputs, and dataset ID derivation.
- Keep `git_commit` in manifest for auditability but exclude it from dataset identity.

Consequences:
- Logical transformation changes can intentionally force new dataset IDs.
- Non-output changes (e.g., docs-only commits) do not perturb dataset identity.
- Reproducibility claims remain about logical transformation identity, not guaranteed byte-for-byte file equality.

Alternatives considered:
- Include Git commit in dataset ID.

Why not chosen:
- Over-couples identity to repository activity unrelated to output bytes or logic.

## ADR-016: strict manifest identity checks before dataset reuse

Context:
- Reusing an existing dataset path is safe only if manifest identity fields match the current execution intent.

Decision:
- Before reuse, require identity agreement on completion status, dataset ID, configuration hash, source SHA/month, pipeline version, schema/encoding versions, and player key ID.
- Missing or inconsistent identity fields fail reuse loudly.
- Run schema + DuckDB validation only after identity checks pass.

Consequences:
- Prevents accidental reuse of stale/tampered datasets.
- Preserves idempotency without silent corruption or identity drift.

Alternatives considered:
- Reuse based only on final dataset path and completion status.

Why not chosen:
- Path-only checks are insufficient against tampered or incompatible manifests.

## ADR-017: explicit timing boundaries and throughput semantics

Context:
- Single-duration metrics are ambiguous when checksum cost is mixed with processing and validation.

Decision:
- Record separate durations:
	- `checksum_duration_seconds`
	- `processing_and_validation_duration_seconds`
	- `total_duration_seconds`
- Publish separate throughput metrics for processing and total time.

Consequences:
- Performance reporting is honest and interpretable.
- End-to-end throughput and processing-only throughput are distinguishable.

Alternatives considered:
- Keep a single `duration_seconds` and one throughput value.

Why not chosen:
- Conflates phases and can mislead comparisons.