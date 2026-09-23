"""Typed source lineage for game identity across direct and sharded ingestion.

The logical identity of a game must never depend on which physical shard file it
was read from. Instead, identity is anchored to the original *parent* archive
identity plus a *global* game index that counts every raw game boundary in the
parent archive (including games later rejected by tolerant parsing).

`SourceLineage` carries the identity anchor (parent archive name + SHA-256) and a
global index offset so that a shard can be ingested with the exact same
`game_id` values it would have received from a direct parent ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLineage:
    """Identity anchor used to derive stable global game identity.

    Attributes:
        identity_archive_name: Parent archive filename recorded on every
            ``GameRecord.source_archive`` so shard-derived rows match direct rows.
        identity_archive_sha256: Parent archive SHA-256 used to derive ``game_id``.
        global_game_index_offset: Added to each local (in-shard) game index to
            produce the global source game index.
        shard_index: Deterministic shard number (physical lineage only).
        shard_filename: Physical shard filename (physical lineage only).
        shard_sha256: Physical shard SHA-256 (physical lineage only).
    """

    identity_archive_name: str
    identity_archive_sha256: str
    global_game_index_offset: int
    shard_index: int | None = None
    shard_filename: str | None = None
    shard_sha256: str | None = None

    def global_index(self, local_index: int) -> int:
        return self.global_game_index_offset + local_index

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "parent_archive_filename": self.identity_archive_name,
            "parent_archive_sha256": self.identity_archive_sha256,
            "global_game_index_offset": self.global_game_index_offset,
            "shard_index": self.shard_index,
            "shard_filename": self.shard_filename,
            "shard_sha256": self.shard_sha256,
        }
