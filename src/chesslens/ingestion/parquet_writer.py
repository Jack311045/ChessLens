"""Bounded batch-to-Parquet writer for ingestion datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from chesslens.validation.schemas import (
    GAME_ARROW_SCHEMA,
    INGESTION_ERROR_ARROW_SCHEMA,
    MOVE_ARROW_SCHEMA,
    records_to_arrow_table,
)

DATASET_ORDER = ("games", "moves", "ingestion_errors")

DATASET_SCHEMAS = {
    "games": GAME_ARROW_SCHEMA,
    "moves": MOVE_ARROW_SCHEMA,
    "ingestion_errors": INGESTION_ERROR_ARROW_SCHEMA,
}


@dataclass(frozen=True)
class DatasetWriteStats:
    dataset_name: str
    part_count: int
    row_count: int
    total_bytes: int
    relative_files: list[str]


class PartitionedParquetWriter:
    """Writes deterministic Parquet parts into source_month Hive partitions."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        source_month: str,
        compression: str,
        row_group_size: int,
    ) -> None:
        self.dataset_root = dataset_root
        self.source_month = source_month
        self.compression = compression
        self.row_group_size = row_group_size

        self._part_index: dict[str, int] = {name: 0 for name in DATASET_ORDER}
        self._row_count: dict[str, int] = {name: 0 for name in DATASET_ORDER}
        self._bytes_written: dict[str, int] = {name: 0 for name in DATASET_ORDER}
        self._relative_files: dict[str, list[str]] = {name: [] for name in DATASET_ORDER}

    def _partition_dir(self, dataset_name: str) -> Path:
        path = self.dataset_root / dataset_name / f"source_month={self.source_month}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_table(self, dataset_name: str, table: Any) -> Path:
        partition_dir = self._partition_dir(dataset_name)
        part_number = self._part_index[dataset_name]
        output_file = partition_dir / f"part-{part_number:06d}.parquet"
        pq.write_table(
            table,
            output_file,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )

        self._part_index[dataset_name] += 1
        self._row_count[dataset_name] += int(table.num_rows)
        file_size = output_file.stat().st_size
        self._bytes_written[dataset_name] += file_size
        self._relative_files[dataset_name].append(output_file.relative_to(self.dataset_root).as_posix())
        return output_file

    def write_records(self, dataset_name: str, records: list[Any]) -> None:
        if dataset_name not in DATASET_SCHEMAS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if not records:
            return
        table = records_to_arrow_table(records, DATASET_SCHEMAS[dataset_name])
        self._write_table(dataset_name, table)

    def ensure_schema_safe_empty_parts(self) -> None:
        for dataset_name in DATASET_ORDER:
            if self._part_index[dataset_name] > 0:
                continue
            empty_table = records_to_arrow_table([], DATASET_SCHEMAS[dataset_name])
            self._write_table(dataset_name, empty_table)

    def stats(self) -> dict[str, DatasetWriteStats]:
        return {
            dataset_name: DatasetWriteStats(
                dataset_name=dataset_name,
                part_count=self._part_index[dataset_name],
                row_count=self._row_count[dataset_name],
                total_bytes=self._bytes_written[dataset_name],
                relative_files=list(self._relative_files[dataset_name]),
            )
            for dataset_name in DATASET_ORDER
        }

    @property
    def total_output_bytes(self) -> int:
        return sum(self._bytes_written.values())
