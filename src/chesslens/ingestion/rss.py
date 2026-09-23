"""Cross-platform periodic peak-RSS sampler shared by ingestion and sharding."""

from __future__ import annotations

import os
import threading

import psutil


class PeakRssSampler:
    """Samples process RSS on a background thread and reports the peak."""

    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0

    def start(self) -> None:
        process = psutil.Process(os.getpid())

        def sampler() -> None:
            while not self._stop.is_set():
                try:
                    rss = int(process.memory_info().rss)
                    if rss > self._peak_rss_bytes:
                        self._peak_rss_bytes = rss
                finally:
                    self._stop.wait(self._interval_seconds)

        self._thread = threading.Thread(target=sampler, name="peak-rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self._peak_rss_bytes

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak_rss_bytes
