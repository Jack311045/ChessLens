"""Canonical played-date resolution for PGN headers.

Policy: use UTCDate when valid, otherwise Date when valid, otherwise null.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

CanonicalDateSource = Literal["utc_date", "date", "missing_or_invalid"]

CANONICAL_PLAYED_DATE_RESOLVER_VERSION = "utc_date_first_v1"

_DATE_PATTERN = re.compile(r"^\d{4}[.-]\d{2}[.-]\d{2}$")


@dataclass(frozen=True)
class CanonicalPlayedDate:
    date_header_raw: str | None
    utc_date_header_raw: str | None
    date_header_valid: bool
    utc_date_header_valid: bool
    canonical_played_date_iso: str | None
    canonical_date_source: CanonicalDateSource
    resolver_version: str = CANONICAL_PLAYED_DATE_RESOLVER_VERSION


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _parse_pgn_date_to_iso(value: str | None) -> str | None:
    text = _normalize_optional_text(value)
    if text is None:
        return None
    if "?" in text:
        return None
    if _DATE_PATTERN.fullmatch(text) is None:
        return None

    normalized = text.replace(".", "-")
    try:
        parsed = datetime.strptime(normalized, "%Y-%m-%d").date()
    except ValueError:
        return None
    return parsed.isoformat()


def resolve_canonical_played_date(
    *,
    date_header: str | None,
    utc_date_header: str | None,
) -> CanonicalPlayedDate:
    utc_iso = _parse_pgn_date_to_iso(utc_date_header)
    date_iso = _parse_pgn_date_to_iso(date_header)

    if utc_iso is not None:
        return CanonicalPlayedDate(
            date_header_raw=_normalize_optional_text(date_header),
            utc_date_header_raw=_normalize_optional_text(utc_date_header),
            date_header_valid=date_iso is not None,
            utc_date_header_valid=True,
            canonical_played_date_iso=utc_iso,
            canonical_date_source="utc_date",
        )

    if date_iso is not None:
        return CanonicalPlayedDate(
            date_header_raw=_normalize_optional_text(date_header),
            utc_date_header_raw=_normalize_optional_text(utc_date_header),
            date_header_valid=True,
            utc_date_header_valid=False,
            canonical_played_date_iso=date_iso,
            canonical_date_source="date",
        )

    return CanonicalPlayedDate(
        date_header_raw=_normalize_optional_text(date_header),
        utc_date_header_raw=_normalize_optional_text(utc_date_header),
        date_header_valid=False,
        utc_date_header_valid=False,
        canonical_played_date_iso=None,
        canonical_date_source="missing_or_invalid",
    )