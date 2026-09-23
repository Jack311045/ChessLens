"""Streaming, byte-preserving PGN game-boundary detection for Lichess archives.

Why bytes and not text: splitting a compressed ``.pgn.zst`` at arbitrary byte
offsets is unsafe because zstd frames are not independently decodable at random
positions, and splitting decompressed text at arbitrary offsets can cut a game in
half. This module streams the *decompressed* byte stream and yields the exact
bytes of each complete PGN game, so each shard we later write is a valid,
independently decompressible archive whose games re-parse identically.

Format assumption (documented and tested): in Lichess "standard rated" exports a
new game always begins with a line starting with ``[Event `` and game movetext
never contains a line that starts with ``[Event ``. This is a boundary scan, not
a full PGN grammar parser and not a chess-board reconstruction.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import zstandard

_MARKER = b"[Event "
_LINE_MARKER = b"\n" + _MARKER
_UTF8_BOM = b"\xef\xbb\xbf"

DEFAULT_CHUNK_SIZE = 1 << 20
DEFAULT_MAX_GAME_BYTES = 8 * 1024 * 1024


class PgnBoundaryError(RuntimeError):
    """Raised when the stream cannot be split into complete PGN games safely."""


def _first_game_start(buffer: bytearray) -> int | None:
    if buffer.startswith(_MARKER):
        return 0
    idx = buffer.find(_LINE_MARKER)
    if idx == -1:
        return None
    return idx + 1


def iter_game_bytes(
    stream: BinaryIO,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_game_bytes: int = DEFAULT_MAX_GAME_BYTES,
) -> Iterator[bytes]:
    """Yield the exact bytes of each complete PGN game from a decompressed stream.

    Memory is bounded to at most one in-flight game plus one read chunk. A single
    game larger than ``max_game_bytes`` raises :class:`PgnBoundaryError` so one
    malformed game cannot grow memory without bound.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_game_bytes <= 0:
        raise ValueError("max_game_bytes must be positive")

    buffer = bytearray()
    game_started = False
    stripped_bom = False
    eof = False

    while not eof:
        chunk = stream.read(chunk_size)
        if not chunk:
            eof = True
        else:
            buffer += chunk

        if not stripped_bom:
            if buffer.startswith(_UTF8_BOM):
                del buffer[: len(_UTF8_BOM)]
            if len(buffer) >= len(_UTF8_BOM) or eof:
                stripped_bom = True

        if not game_started:
            start = _first_game_start(buffer)
            if start is None:
                if eof:
                    return
                if len(buffer) > max_game_bytes:
                    raise PgnBoundaryError(
                        "No PGN game header found within max_game_bytes; "
                        "input does not look like a Lichess standard archive"
                    )
                continue
            if start > 0:
                del buffer[:start]
            game_started = True

        while True:
            idx = buffer.find(_LINE_MARKER)
            if idx == -1:
                break
            game = bytes(buffer[: idx + 1])
            del buffer[: idx + 1]
            yield game

        if not eof and len(buffer) > max_game_bytes:
            raise PgnBoundaryError(
                f"Single PGN game exceeds max_game_bytes={max_game_bytes}"
            )

    if game_started and buffer.strip():
        if len(buffer) > max_game_bytes:
            raise PgnBoundaryError(
                f"Final PGN game exceeds max_game_bytes={max_game_bytes}"
            )
        yield bytes(buffer)


def iter_game_bytes_from_zst(
    archive_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_game_bytes: int = DEFAULT_MAX_GAME_BYTES,
) -> Iterator[bytes]:
    """Stream-decompress a ``.pgn.zst`` archive and yield each complete game."""

    with archive_path.open("rb") as compressed:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(compressed) as binary_stream:
            yield from iter_game_bytes(
                binary_stream,
                chunk_size=chunk_size,
                max_game_bytes=max_game_bytes,
            )


def count_games_in_zst(
    archive_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_game_bytes: int = DEFAULT_MAX_GAME_BYTES,
) -> int:
    """Independently decompress an archive and count complete games."""

    return sum(
        1
        for _ in iter_game_bytes_from_zst(
            archive_path,
            chunk_size=chunk_size,
            max_game_bytes=max_game_bytes,
        )
    )
