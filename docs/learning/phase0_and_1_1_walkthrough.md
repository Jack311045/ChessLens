# Phase 0 and Phase 1.1 Walkthrough

Audience: undergraduate mathematics/computer science student who knows Python and wants to explain design decisions in interviews.

## 1. What These Phases Accomplish

Phase 0 and Phase 1.1 build the engineering foundation before large-scale ETL or model training.

Phase 0 gives you:

- reproducible environment and dependency locking;
- stable project layout;
- deterministic encoding contracts;
- quality gates (tests, lint, type checks, CI).

Phase 1.1 gives you:

- explicit logical data contracts with versioning;
- deterministic game/position identity rules;
- bounded-memory real-data streaming validation;
- privacy-safe fixture generation for CI.

## 2. Why a Repo Skeleton Alone Is Not Enough

A folder tree with empty files does not prove correctness. Production-style evidence needs:

- contract definitions with schema and invariants;
- deterministic IDs and encoding behavior;
- failure handling under malformed/truncated inputs;
- reproducible local and CI execution;
- tests that verify properties, not just happy paths.

## 3. Directory Roles

- `src/`: importable library code used by scripts/tests and future orchestration.
- `tests/`: unit/property/integration checks with fixture-only CI data.
- `configs/`: parameterized YAML configs instead of hardcoded paths.
- `data/`: local raw source, tiny fixtures, and manifests.
- `dbt/`: skeleton for future SQL lineage once ingestion outputs exist.
- `docs/`: architecture decisions, contracts, encoding spec, learning notes.
- `.github/`: CI workflow automation.

## 4. Why `src/chesslens/` Has Two Levels

- `src/` controls Python import behavior so tests mimic installed package behavior.
- `chesslens/` is the package namespace (`import chesslens...`).

Without this split, path leaks and accidental local-import behavior become common.

## 5. `pyproject.toml`, `uv.lock`, `.venv`, `requirements.txt`

- `pyproject.toml`: project metadata + dependency constraints + tool config.
- `uv.lock`: exact pinned dependency graph for deterministic installs.
- `.venv`: machine-local environment directory (ignored).
- `requirements.txt`: compatibility export format for platforms that require it.

Why `uv.lock` is committed:

- CI and collaborators install the same transitive versions.
- Debugging is reproducible.

Why `.venv` is ignored:

- not portable across machines/OS.
- includes binaries and local paths.

## 6. Package-by-Package Purpose

Runtime:

- `python-chess`: parse PGN, replay legal moves, board/FEN handling.
- `zstandard`: stream `.zst` compression/decompression.
- `numpy`: tensor representation for board/action encodings.
- `polars`: fast local frame checks now, larger ETL later.
- `pyarrow`: explicit schema definitions and Parquet round-trips.
- `duckdb`: future local analytical store compatibility.
- `PyYAML`: config parsing.

Development:

- `pytest`: test runner.
- `hypothesis`: property tests over randomized legal positions.
- `ruff`: lint and formatting checks.
- `mypy`: static type checks.
- `pre-commit`: automated local gate runner.

## 7. Compressed Streaming: How It Works

Pipeline used by ingestion:

1. open compressed file as binary stream;
2. wrap with `zstandard.ZstdDecompressor().stream_reader(...)`;
3. wrap decompressed bytes with `io.TextIOWrapper` for text decoding;
4. call `chess.pgn.read_game()` repeatedly to parse one game at a time.

Key point: this reads incrementally. It does not load entire decompressed archive into RAM.

## 8. Downloading a File vs Loading It Into RAM

- Downloading: data is stored on disk.
- Loading into RAM: bytes are read into process memory.

A 30 GB archive on disk is not automatically 30 GB in RAM. Streaming keeps memory bounded to chunk buffers.

## 9. What a Generator and `yield` Do

A generator returns one item at a time and resumes where it left off.

Why this matters:

- lower memory usage;
- can stop early (`max_games`);
- natural fit for long sequential archives.

`iter_raw_games` and `iter_parsed_games` are generator-based for exactly this reason.

## 10. Data Contracts Vocabulary

- grain: one row means exactly one thing.
- schema: field names, types, nullability.
- primary key: unique row identity.
- foreign key: relation to another table.
- nullability: whether missing values are allowed.
- invariant: condition that must always hold.

## 11. FEN vs Normalized FEN vs `position_id` vs Neural Encoding

- FEN (full): includes board + side + castling + en passant + halfmove + fullmove.
- normalized FEN (`fen4_legal_ep_v1`): first four fields only, legal-en-passant convention.
- `position_id`: SHA-256 hash of normalization version + normalized FEN.
- neural encoding: dense numeric tensor `(18, 8, 8)` derived from board state.

Normalized FEN is a board-state identity, not a full history identity (repetition history is not encoded).

## 12. The 18-Plane Board Encoding

Channel order:

0..5 white pieces (P,N,B,R,Q,K)
6..11 black pieces (P,N,B,R,Q,K)
12 side-to-move plane
13..16 castling rights planes (WK,WQ,BK,BQ)
17 legal en-passant square plane

Orientation is absolute:

- file index 0 = `a`
- rank index 0 = `1`
- tensor index order `[channel, rank, file]`

## 13. Why Policy Needs a Fixed Action Space

Neural policy heads output fixed-size vectors/tensors. Chess legal moves vary by position. Fixed `4672` actions allows:

- consistent model output shape;
- batching;
- legal masking to zero invalid actions.

## 14. The `8 x 8 x 73` Action Encoding

- 56 planes: sliding moves (8 directions x 7 distances)
- 8 planes: knight offsets
- 9 planes: underpromotions (left/forward/right x N/B/R)

Index formula:

`action_index = from_square * 73 + move_plane`

Special cases:

- castling: king two-square horizontal move.
- en passant: pawn diagonal move.
- queen promotion: represented by standard sliding plane.

## 15. Why Pre-Move vs Post-Move Matters

The supervised target move is generated from the pre-move board.

If you capture post-move state as input, the model sees outcome information that would be unavailable at decision time.

## 16. Leakage Examples

- `result` as feature: leaks game outcome into earlier plies.
- final `termination`: future information.
- post-move clocks embedded in comments: may encode later events.
- opening labels at very early plies: may include future-move naming effects.

## 17. Testing Layers

- fixture tests: confirm tiny committed sample integrity.
- unit tests: deterministic behavior of individual functions.
- property tests: broad randomized invariants across many legal positions.
- integration tests: multi-module behavior with real streaming fixture.
- end-to-end tests: full pipeline (deferred to future PR).

## 18. Why Raw Archive Is Excluded from Git

- very large file size;
- repository performance and storage concerns;
- unnecessary for deterministic CI once tiny fixture exists.

## 19. Git Terms in This Project

- repository: version-controlled project folder.
- branch: parallel line of work (`phase-0-foundation-and-data-contracts`).
- commit: snapshot with message.
- remote: hosted repository endpoint (GitHub, etc.).
- push: send local commits to remote.
- CI: automated checks per push/PR.
- pull request: proposed merge from branch to main.

## 20. Important Functions Explained

### `load_ingestion_config`

- input: path to YAML config + optional override mapping.
- output: typed `IngestionConfig`.
- invariant: `max_games >= 0`; input path required.
- failure behavior: raises `ValueError`/`FileNotFoundError` on invalid config.
- design reason: separate runtime parameters from code.

### `compute_sha256`

- input: file path and chunk size.
- output: hex SHA-256 string.
- invariant: same file bytes => same hash.
- failure behavior: I/O errors propagate.
- design reason: integrity and stable source identity.

### `iter_raw_games`

- input: `.pgn.zst` path, optional max games.
- output: generator of `(source_game_index, chess.pgn.Game)`.
- invariant: reads incrementally; never unbounded full-file read.
- failure behavior: raises parse/decompression exceptions.
- design reason: bounded-memory scaling path.

### `parse_game_to_records`

- input: one parsed PGN game + source metadata.
- output: `ParsedGame` containing `GameRecord`, `MoveRecord[]`, `PositionRecord[]`.
- invariant: each move is legal on pre-move board, zero-based pre-move `ply`.
- failure behavior: raises `GameParseException` with typed error record.
- design reason: enforce core semantic correctness in one place.

### `iter_parsed_games`

- input: archive path + strict/tolerant mode + optional error callback.
- output: generator of valid parsed game bundles.
- invariant: deterministic processing order by source ordinal.
- failure behavior: strict mode raises; tolerant mode reports and skips invalid.
- design reason: single API for both tests and profiling.

### `normalize_fen`

- input: FEN string + normalization version.
- output: normalized 4-field FEN.
- invariant: uses legal en-passant convention.
- failure behavior: invalid FEN/version raises `ValueError`.
- design reason: stable cross-run position identity.

### `position_id_from_normalized_fen`

- input: normalized FEN + version.
- output: SHA-256 position ID.
- invariant: deterministic and version-scoped.
- failure behavior: none expected for valid strings.
- design reason: explicit compatibility boundary for future normalization changes.

### `encode_board_18x8x8`

- input: `chess.Board`.
- output: `numpy.ndarray` of shape `(18,8,8)`.
- invariant: deterministic, non-mutating, fixed channel order.
- failure behavior: none for valid board object.
- design reason: clear neural-ready representation.

### `encode_move_to_index`

- input: `chess.Move` + board context.
- output: integer in `[0,4671]`.
- invariant: follows fixed plane mapping.
- failure behavior: raises if move cannot be represented.
- design reason: fixed policy target index.

### `decode_index_to_move`

- input: action index + board context.
- output: `chess.Move`.
- invariant: inverse mapping for legal moves.
- failure behavior: raises on out-of-range, impossible, or illegal (when required).
- design reason: debugging, analysis, and round-trip validation.

### `legal_move_mask`

- input: board.
- output: bool vector `(4672,)`.
- invariant: true-count equals number of legal moves.
- failure behavior: raises on collisions.
- design reason: support masked policy inference/training.

### `create_fixture`

- input: fixture config path + optional overrides.
- output: manifest dict.
- invariant: selects first N complete valid games; sanitizes identifiers.
- failure behavior: raises when insufficient valid games or config invalid.
- design reason: deterministic tiny dataset for CI.

### `profile_archive`

- input: smoke config and optional overrides.
- output: machine-readable profile dict + JSON report.
- invariant: bounded by `max_games` and excludes PII in output.
- failure behavior: strict mode can raise on parse errors.
- design reason: validate assumptions safely before large ETL.

### `records_to_arrow_table`

- input: dataclass records + explicit Arrow schema.
- output: typed Arrow table.
- invariant: output schema is explicit and versioned.
- failure behavior: type mismatch errors propagate.
- design reason: contract-checkable analytical interoperability.

## 21. File-by-File Explanation (What Was Added)

Top-level:

- `pyproject.toml`: package metadata, dependencies, tool config.
- `uv.lock`: exact resolved dependency graph (generated during setup).
- `.python-version`: target interpreter version.
- `.gitignore`: excludes raw archives, local envs, large artifacts, secrets.
- `Makefile`: convenient reproducible commands.
- `.pre-commit-config.yaml`: local quality hooks.

Config and CI:

- `configs/ingestion/fixture.yaml`: fixture generation configuration.
- `configs/ingestion/smoke.yaml`: sample profiler configuration.
- `.github/workflows/ci.yml`: lint + mypy + tests in CI.

Source code:

- `src/chesslens/domain/records.py`: typed logical contracts.
- `src/chesslens/validation/schemas.py`: explicit Arrow/Polars schemas.
- `src/chesslens/features/position_encoding.py`: normalization, IDs, board encoding.
- `src/chesslens/features/action_encoding.py`: fixed action space and legal masks.
- `src/chesslens/ingestion/config.py`: typed YAML config loader.
- `src/chesslens/ingestion/pgn_reader.py`: streaming reader and record extraction.
- `src/chesslens/ingestion/sample_profiler.py`: bounded real-data profile command.

Scripts:

- `scripts/create_fixture.py`: deterministic, sanitized fixture generator.

Tests:

- `tests/unit/*`: deterministic correctness and edge behavior checks.
- `tests/property/*`: randomized invariant/property checks.
- `tests/integration/*`: fixture streaming, legality replay, schema round-trips.

Docs:

- `docs/ChessLens_Project_Blueprint.md`: preserved source blueprint copy.
- `docs/architecture-decisions.md`: ADR log and alternatives.
- `docs/data-contract.md`: formal semantic contract tables.
- `docs/dependency-plan.md`: environment and dependency rationale.
- `docs/encoding-specification.md`: exact board/action mapping.
- `docs/glossary.md`: key terms.

## 22. Alternatives Considered and Rejected

- full in-memory PGN parsing: rejected due scaling risk.
- random UUID game IDs: rejected due non-determinism.
- storing full 6-field FEN as position identity: rejected for over-distinguishing states.
- variable-length action targets: rejected due model output mismatch and batching complexity.
- immediate dbt SQL implementation: rejected because ingestion outputs are not yet materialized.

## 23. Known Limitations and Exact Next PR Scope

Current limitations:

- no full monthly ETL partition writer;
- no engine-eval archive parser implementation;
- no training dataset marts;
- no model training or inference APIs.

Next PR should implement exactly:

1. Phase 1.2 scalable ETL writer for game/move/position Parquet outputs.
2. Run manifests persisted per ingestion run with artifact URIs.
3. Deterministic partitioning and idempotent rebuild behavior.
4. First dbt staging models and SQL data tests over produced Parquet/DuckDB sources.
5. Throughput and memory instrumentation for larger bounded runs.

## 24. Interview Questions with Model Answers

1. Why normalize FEN to four fields?
- It preserves board, side-to-move, castling, and legal en-passant identity while excluding move counters that are not always needed for position-level joins.

2. Why include normalization version in `position_id` hashing?
- So future normalization changes do not silently collide with old IDs.

3. Why use zero-based pre-move ply indexing?
- It makes prediction-time semantics explicit and avoids post-move leakage.

4. Why fixed 4672-action space instead of variable legal-move lists?
- Fixed outputs are required for standard dense policy heads and efficient batching.

5. How do you prevent illegal action predictions from being selected?
- Compute legal mask from board and apply it before ranking/selecting policy outputs.

6. Why stream `.pgn.zst` instead of decompressing the full file first?
- Streaming keeps memory bounded and scales to very large archives.

7. Why commit `uv.lock` but ignore `.venv`?
- Lockfile is reproducibility metadata; `.venv` is machine-specific runtime state.

8. Why have both dataclasses and Arrow schemas?
- Dataclasses define Python-level contracts; Arrow schemas define analytical storage types and enable strict table validation.

9. How do you handle malformed PGN data?
- Strict mode fails fast; tolerant mode records typed ingestion errors and continues for profiling.

10. Why sanitize fixture player names and site URLs?
- To avoid publishing identifying information in committed derived data while preserving structural realism.

11. Why is dbt deferred in this PR?
- Real dbt models need stable physical sources; creating fake transformations now would be misleading.

12. How do property tests add value beyond unit tests?
- They stress invariants across many randomized legal positions, catching edge interactions that hand-picked examples miss.
