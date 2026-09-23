"""Disk and size estimates for sharding/ingestion, grounded in measured 2013 data.

All figures are ESTIMATES derived from the measured 2013-01 run and multiplied by a
documented safety margin. They must never be presented as measured 2017 results.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

# Measured from the published 2013-01 dataset manifest.
MEASURED_2013_GAMES = 121_332
MEASURED_2013_BRONZE_OUTPUT_BYTES = 477_746_376
MEASURED_2013_PARENT_ARCHIVE_BYTES = 17_761_302

# Bytes-per-game derived from the measured 2013-01 run (estimates only).
BRONZE_BYTES_PER_GAME = MEASURED_2013_BRONZE_OUTPUT_BYTES / MEASURED_2013_GAMES
RAW_ARCHIVE_BYTES_PER_GAME = MEASURED_2013_PARENT_ARCHIVE_BYTES / MEASURED_2013_GAMES

# Safety margin applied to every estimate.
SAFETY_MARGIN = 1.30


@dataclass(frozen=True)
class SizeEstimate:
    expected_games: int
    estimated_raw_shards_bytes: int
    estimated_bronze_parquet_bytes: int
    estimated_duckdb_bytes: int
    estimated_total_bytes: int
    free_disk_bytes: int
    safety_margin: float

    def as_dict(self) -> dict[str, object]:
        return {
            "expected_games": self.expected_games,
            "safety_margin": self.safety_margin,
            "estimated_raw_shards_bytes": self.estimated_raw_shards_bytes,
            "estimated_bronze_parquet_bytes": self.estimated_bronze_parquet_bytes,
            "estimated_duckdb_bytes": self.estimated_duckdb_bytes,
            "estimated_total_bytes": self.estimated_total_bytes,
            "estimated_total_gib": round(self.estimated_total_bytes / (1024**3), 3),
            "free_disk_bytes": self.free_disk_bytes,
            "free_disk_gib": round(self.free_disk_bytes / (1024**3), 3),
            "fits_with_margin": self.free_disk_bytes >= self.estimated_total_bytes,
            "note": "estimates from measured 2013-01 bytes-per-game; not a 2017 measurement",
        }


def estimate_sizes(expected_games: int, *, disk_probe_path: Path) -> SizeEstimate:
    raw = int(expected_games * RAW_ARCHIVE_BYTES_PER_GAME * SAFETY_MARGIN)
    bronze = int(expected_games * BRONZE_BYTES_PER_GAME * SAFETY_MARGIN)
    # DuckDB/dbt intermediate + mart tables roughly duplicate bronze move-level data.
    duckdb_bytes = int(bronze * 1.5)
    total = raw + bronze + duckdb_bytes
    probe = disk_probe_path if disk_probe_path.exists() else disk_probe_path.anchor or Path.cwd()
    free = shutil.disk_usage(probe).free
    return SizeEstimate(
        expected_games=expected_games,
        estimated_raw_shards_bytes=raw,
        estimated_bronze_parquet_bytes=bronze,
        estimated_duckdb_bytes=duckdb_bytes,
        estimated_total_bytes=total,
        free_disk_bytes=free,
        safety_margin=SAFETY_MARGIN,
    )
