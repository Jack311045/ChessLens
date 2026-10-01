"""Bounded deterministic Parquet writer for modeling datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class PartitionWriteStats:
    dataset_name: str
    partition_value: str
    part_count: int
    row_count: int
    total_bytes: int
    relative_files: list[str]


class ModelingParquetWriter:
    """Writes deterministic Parquet part files, optionally partitioned by split."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        compression: str,
        row_group_size: int,
    ) -> None:
        self.dataset_root = dataset_root
        self.compression = compression
        self.row_group_size = row_group_size

        self._part_index: dict[tuple[str, str], int] = {}
        self._row_count: dict[tuple[str, str], int] = {}
        self._bytes_written: dict[tuple[str, str], int] = {}
        self._relative_files: dict[tuple[str, str], list[str]] = {}

    def _dataset_partition_dir(self, *, dataset_name: str, partition_value: str) -> Path:
        base = self.dataset_root / dataset_name
        if partition_value == "_all":
            path = base
        else:
            path = base / f"temporal_split={partition_value}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_table(self, *, dataset_name: str, partition_value: str, table: pa.Table) -> None:
        key = (dataset_name, partition_value)
        part_number = self._part_index.get(key, 0)
        partition_dir = self._dataset_partition_dir(
            dataset_name=dataset_name,
            partition_value=partition_value,
        )
        out_path = partition_dir / f"part-{part_number:06d}.parquet"
        pq.write_table(
            table,
            out_path,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )

        self._part_index[key] = part_number + 1
        self._row_count[key] = self._row_count.get(key, 0) + int(table.num_rows)
        size = int(out_path.stat().st_size)
        self._bytes_written[key] = self._bytes_written.get(key, 0) + size
        self._relative_files.setdefault(key, []).append(
            out_path.relative_to(self.dataset_root).as_posix()
        )

    def write_rows(
        self,
        *,
        dataset_name: str,
        rows: list[dict[str, Any]],
        schema: pa.Schema,
        partition_column: str | None = None,
    ) -> None:
        if not rows:
            return

        if partition_column is None:
            table = pa.Table.from_pylist(rows, schema=schema)
            self._write_table(dataset_name=dataset_name, partition_value="_all", table=table)
            return

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            partition_value = str(row[partition_column])
            grouped.setdefault(partition_value, []).append(row)

        for partition_value in sorted(grouped):
            table = pa.Table.from_pylist(grouped[partition_value], schema=schema)
            self._write_table(
                dataset_name=dataset_name,
                partition_value=partition_value,
                table=table,
            )

    def stats(self) -> dict[str, dict[str, PartitionWriteStats]]:
        result: dict[str, dict[str, PartitionWriteStats]] = {}
        for (dataset_name, partition_value), part_count in self._part_index.items():
            result.setdefault(dataset_name, {})[partition_value] = PartitionWriteStats(
                dataset_name=dataset_name,
                partition_value=partition_value,
                part_count=part_count,
                row_count=self._row_count[(dataset_name, partition_value)],
                total_bytes=self._bytes_written[(dataset_name, partition_value)],
                relative_files=list(self._relative_files[(dataset_name, partition_value)]),
            )
        return result

    @property
    def total_output_bytes(self) -> int:
        return sum(self._bytes_written.values())
