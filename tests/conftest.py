from __future__ import annotations

from pathlib import Path

import zstandard


def write_zst_text(tmp_path: Path, content: str, filename: str = "sample.pgn.zst") -> Path:
    output_path = tmp_path / filename
    compressor = zstandard.ZstdCompressor(level=3)
    with output_path.open("wb") as stream:
        with compressor.stream_writer(stream) as writer:
            writer.write(content.encode("utf-8"))
    return output_path