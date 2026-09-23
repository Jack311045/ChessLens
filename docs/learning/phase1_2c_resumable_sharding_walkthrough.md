# Phase 1.2c: Resumable, Game-Boundary-Aware Sharding (Beginner Walkthrough)

Audience: someone new to data engineering. This explains, from first principles,
how ChessLens splits one huge chess archive into many small pieces ("shards") that
can be processed over several sessions — so you never need to leave your computer
running for ~70 hours straight.

Read this alongside the Phase 1.2a (ingestion) and Phase 1.2b (dbt warehouse)
walkthroughs. Nothing from those phases was removed.

---

## 1. The problem in plain words

The 2017-01 Lichess archive holds **10,680,708 games** in a compressed
`~1.9 GB` file. Ingesting it end-to-end takes many hours. If your laptop sleeps,
loses power, or you simply want to stop, you would lose all progress.

The fix: cut the archive into ~43 independent **shards** of 250,000 games each,
then ingest a few shards per session. Stop any time; resume later.

---

## 2. Why you cannot just cut the file in half

### 2.1 Compressed vs decompressed streams

A `.pgn.zst` file is **compressed**. The bytes on disk are not the chess text —
they are a zstd-encoded representation of it. Think of it like a vacuum-sealed bag:
you cannot read the contents without opening (decompressing) it.

- **Compressed stream**: the bytes in the `.zst` file.
- **Decompressed stream**: the readable PGN text you get after unzipping.

### 2.2 Why arbitrary byte splitting of `.zst` is unsafe

If you chopped the compressed `.zst` file at byte 900,000,000, neither half would
decompress. zstd data is only decodable from the start of a **frame**; a random
offset lands in the middle of compressed symbols and is meaningless. A zstd
**frame/stream** is one self-contained compressed unit that begins with a header —
you must start decoding at a frame boundary, not anywhere.

### 2.3 Why arbitrary text splitting is also unsafe

Even after decompressing, cutting the text at a random line could slice a game in
half — you would get half of game 7 in one shard and half in the next. Both would
be corrupt.

So we must split **on game boundaries**, and we must write each shard as its **own
independent zstd stream** so it can be decompressed by itself later.

---

## 3. What is a PGN game boundary?

A PGN game looks like this:

```
[Event "Rated Blitz game"]
[Site "..."]
[Result "1-0"]

1. e4 e5 2. Nf3 ... 1-0
```

In Lichess "standard rated" exports, **every game starts with a line beginning
`[Event `**, and game movetext never contains a line that starts with `[Event `.
So the boundary between games is the next line that starts with `[Event `.

We stream the decompressed text and cut exactly at those boundaries. This is a
**boundary scan**, not a full chess-rules parser and not board reconstruction — we
only find where each game starts and ends. That assumption is documented and tested
(comments and header values containing brackets do not fool it).

---

## 4. What is a shard?

A **shard** is a small `.pgn.zst` file containing a contiguous group of complete
games (default 250,000). Each shard:

- is **independently compressed** (its own zstd stream), so you can decompress it
  alone without the parent file;
- keeps games in their original order;
- preserves all PGN content (headers, moves, comments, `[%clk]` and `[%eval]`
  annotations, promotions, castling, en passant) because we copy the exact
  decompressed **bytes** of each game — the text is byte-preserved, not
  re-serialized.

Physical layout:

```
data/raw_shards/2017-01/
  shard-00000.pgn.zst
  shard-00001.pgn.zst
  ...
  _shard_manifest.json
```

---

## 5. Global vs local game indices

Two different "position numbers" exist:

- **Global source game index**: the game's position in the *original parent
  archive*, counting **every** raw game (0, 1, 2, ... 10,680,707).
- **Local shard game index**: the game's position *within one shard* (0, 1, 2, ...).

Shard 1 (offset 250,000) local game 0 is **global** game 250,000.

This matters because a game's identity must never depend on which shard it lives in.

---

## 6. Parent checksum vs shard checksum

- **Parent archive SHA-256**: fingerprint of the original 2017-01 file. Anchors game
  identity and proves you are resuming against the same source.
- **Shard SHA-256**: fingerprint of one shard file. Proves a specific shard was not
  changed or corrupted.

They are different on purpose and both are recorded.

---

## 7. The four identifiers you must not confuse

| Identifier | Scope | Determined by |
| --- | --- | --- |
| `game_id` | one game | parent archive SHA-256 + **global** game index |
| `dataset_id` | one processed shard's Parquet dataset | shard bytes + effective config + versions |
| `collection_id` | the whole month's set of shard datasets | parent SHA + month + shard-plan identity + versions + key id |
| `run_id` | one execution attempt | timestamp + random (always unique) |

The key invariant:

```
game_id = sha256(parent_archive_sha256 + "|" + global_source_game_index)
```

It is **never** based on the shard's SHA or the local index. That is why a game has
the **same** `game_id` whether ingested directly from the parent or from a shard.
The global index counts rejected games too, so rejecting one game does not shift the
identities of the games after it.

---

## 8. Atomic writes, `.partial` files, and half-open ranges

### 8.1 Atomic write and rename

Writing a file is not instant. If the power dies mid-write you could get a
half-written file. To avoid this we write to a temporary name first, then **rename**
it — a rename is atomic on the filesystem (it either fully happens or not at all).

- A shard is written as `shard-00012.pgn.zst.partial`.
- Only after it is closed, independently decompressed, counted, and checksummed do we
  rename it to `shard-00012.pgn.zst`.
- The manifest is written the same way (temp file, then `os.replace`).

### 8.2 Why `.partial` files exist

A `.partial` file is an in-progress shard. If you see one, it means a shard was being
written when the process stopped. On resume we delete **only** the exact expected
`.partial` for the next shard and rewrite it. Completed shards are never touched.

### 8.3 Half-open index ranges

Each shard records `[start, end)` — start included, end excluded. Shard 0 is
`[0, 7)` (games 0–6), shard 1 is `[7, 14)`, etc. Half-open ranges make contiguity
trivial to check: the next shard's start must equal the previous shard's end, with
no gaps and no overlaps, first start 0, final end equal to total games.

---

## 9. Checkpoint and resume behavior

The **shard manifest** (`_shard_manifest.json`) is the checkpoint. It records the
parent identity, the sharding config, and one entry per completed shard.

On resume the sharder:

1. Re-checksums the parent and confirms it matches the manifest (same source).
2. Loads the manifest and validates shard ranges are contiguous.
3. Re-checksums every completed shard file (detects tampering/corruption).
4. Rejects a changed source, changed shard size, changed splitter version, changed
   compression, a missing shard, a gap/overlap, or a tampered shard — loudly.
5. Deletes only the known incomplete `.partial` (and any unreferenced orphan shard at
   the next index left by a crash between publish and manifest write).
6. Resumes at the next global game index.

### Why resume may need to rescan the parent stream

zstd cannot jump to game 250,000 directly (no random access to a game boundary). So
to resume, we re-open the parent and **boundary-scan** past the games we already
emitted, discarding them, then continue. This is cheap scanning of boundaries, not
re-doing any chess work.

---

## 10. Idempotency

**Idempotent** means "running it again changes nothing." If you re-run a completed
shard plan, every shard is reused; file contents, checksums, and modification times
stay identical. If you re-run sharded ingestion, already-processed shards are skipped
and counts do not double.

---

## 11. Manifest-driven lineage

Nothing is discovered by scanning folders with wildcards. Every step reads an
explicit manifest:

- the **shard manifest** lists exactly which shards exist and their ranges/hashes;
- the **collection manifest** lists exactly which processed shard datasets belong to
  the collection.

This prevents accidentally mixing in old datasets, fixtures, reruns, or other months.

---

## 12. HMAC consistency across shards

Player names are anonymized with a keyed HMAC. All shards use the **same secret and
the same key id**, so the same player gets the same pseudonym in every shard. The
secret is read from an environment variable and **never** written to any manifest or
log; only the non-secret key id is recorded.

---

## 13. How collection dbt input works

Phase 1.2b ran dbt on **one** dataset root. Phase 1.2c adds a **collection** mode:

- `CHESSLENS_COLLECTION_ROOT` points at a completed collection.
- Preflight reads the collection manifest and registers the DuckDB `bronze_*` views
  as the **union of only the dataset roots listed in the manifest** (an explicit file
  list, never a broad glob).
- A synthesized `bronze_manifest` exposes the aggregate counts in the same shape the
  single-dataset models expect, so **the exact same dbt models and tests** run over
  the whole month — including global `game_id` uniqueness and `(game_id, ply)`
  uniqueness across all shards.

The old single-dataset commands still work unchanged.

---

## 14. How aggregate metrics avoid double-counting

Throughput is reported using **active compute time** (sum of each shard's processing
seconds), never the multi-day wall-clock gap between sessions. Because the collection
manifest stores one entry per shard and aggregates are recomputed from those unique
entries, reusing a completed shard never double-counts its games, moves, or seconds.
Peak RSS is the **maximum** observed across shards, not a sum.

---

## 15. What happens if... (failure scenarios)

| Event | Result |
| --- | --- |
| Internet disconnects | No effect; everything is local. |
| Terminal closes / Python crashes | Completed shards are safe; rerun with `--resume`. |
| Computer sleeps or loses power | Same; the `.partial` in progress is discarded and rewritten on resume. |
| One shard is corrupted | Resume re-checksums shards and fails loudly; delete that shard and resume to rebuild it. |
| Parent archive changes | Identity check fails loudly; you must not resume against a different source. |
| You rerun a completed command | Idempotent: nothing changes. |

---

## 16. Inspecting progress and resuming safely

```powershell
# How far has sharding gotten?
python -m uv run python -m chesslens.ingestion.run_sharding `
  --config configs/sharding/2017_01.yaml --status

# How many shards are ingested into the collection?
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml --status
```

Resume is always the same command again (with `--resume`). It is safe to run any
number of times.

---

## 17. Verifying final counts

```powershell
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml --verify-only
```

This checks shard-index contiguity, global range contiguity, physical shard
checksums, per-dataset manifest reconciliation, and that aggregate counts add up.
The **10-million evidence** flag (`portfolio_10m_satisfied`) is only `true` when at
least 10,000,000 games were actually accepted and published — never on a prefix.

---

## 18. Which generated files are safe to delete

Safe to delete (regenerable, and git-ignored):

- `data/tmp/*` (DuckDB scratch)
- `data/raw_shards/<month>/*.pgn.zst.partial` (in-progress shards)

Delete only if you accept re-doing work:

- `data/raw_shards/<month>/` completed shards (you can re-shard the parent)
- `data/processed/collections/<id>/` processed datasets (you can re-ingest)

Never delete:

- `data/raw/*.pgn.zst` (the parent archive — large, not committed, hard to re-fetch)
- any `_shard_manifest.json` / `_collection_manifest.json` you still need to resume

---

## 19. Expected disk usage and checking free space

Use the dry run (it processes nothing):

```powershell
python -m uv run python -m chesslens.ingestion.run_sharding `
  --config configs/sharding/2017_01.yaml --dry-run
```

It prints estimated raw-shard, bronze-Parquet, and DuckDB sizes with a safety margin,
plus your free disk space. These are **estimates** derived from the measured 2013-01
bytes-per-game — they are clearly labeled and are **not** a measured 2017 result.

---

## 20. The core user workflow

```powershell
# 1. Look before you leap (no processing)
python -m uv run python -m chesslens.ingestion.run_sharding `
  --config configs/sharding/2017_01.yaml --dry-run

# 2. Create or resume raw shards (safe to stop/rerun)
python -m uv run python -m chesslens.ingestion.run_sharding `
  --config configs/sharding/2017_01.yaml --resume

# 3. Process ONE new shard this session, then shut down if you want
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml --max-new-shards 1 --resume

# 4. Check progress any time
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml --status

# 5. When all shards are done, validate the whole collection
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml --verify-only
```

Real-archive ingestion needs the HMAC environment variables set first:

```powershell
$env:CHESSLENS_PLAYER_HMAC_KEY = "<your-secret>"
$env:CHESSLENS_PLAYER_HMAC_KEY_ID = "prod-key-2026-09"
```

---

## 21. How the 10M evidence can honestly be described

After a complete 2017-01 run you can say, truthfully:

> "Built a resumable ETL that split a 1.9 GB, 10.68-million-game archive into 43
> game-boundary-aware shards and ingested them across multiple sessions into a
> validated DuckDB/dbt warehouse, with global game-identity preservation and
> row-level uniqueness enforced across all shards."

Only claim the 10M milestone if `portfolio_10m_satisfied` is `true` (≥ 10,000,000
games actually accepted and published).

---

## 22. Interview questions you should be able to answer

1. Why is splitting a `.zst` at an arbitrary byte offset unsafe?
2. What is the difference between a compressed stream and a decompressed stream?
3. How do you detect a PGN game boundary in a Lichess archive, and what is the
   documented assumption?
4. Why is each shard compressed independently?
5. What is a zstd frame, and why does it force resume to rescan?
6. Explain global vs local game indices and why identity uses the global index.
7. Why does a rejected game not shift later game identities?
8. What are `game_id`, `dataset_id`, `collection_id`, and `run_id`?
9. What does atomic write-and-rename protect against, and why do `.partial` files
   exist?
10. What is a half-open range and why is it convenient?
11. What does idempotency mean here, and how is it enforced on resume?
12. How does collection-level dbt input avoid accidentally including the wrong data?
13. Why are aggregate metrics based on active compute time instead of wall clock?
14. How is the HMAC secret kept out of manifests and logs while staying consistent
    across shards?
15. What exactly must be true before you claim the 10-million-game milestone?
