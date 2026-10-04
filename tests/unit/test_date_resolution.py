from __future__ import annotations

from chesslens.ingestion.date_resolution import (
    CANONICAL_PLAYED_DATE_RESOLVER_VERSION,
    resolve_canonical_played_date,
)


def test_resolver_prefers_utc_date_when_both_present() -> None:
    resolved = resolve_canonical_played_date(
        date_header="2017.01.15",
        utc_date_header="2017.01.14",
    )
    assert resolved.canonical_played_date_iso == "2017-01-14"
    assert resolved.canonical_date_source == "utc_date"
    assert resolved.date_header_valid is True
    assert resolved.utc_date_header_valid is True
    assert resolved.resolver_version == CANONICAL_PLAYED_DATE_RESOLVER_VERSION


def test_resolver_falls_back_to_date_when_utc_invalid() -> None:
    resolved = resolve_canonical_played_date(
        date_header="2017.01.15",
        utc_date_header="????.??.??",
    )
    assert resolved.canonical_played_date_iso == "2017-01-15"
    assert resolved.canonical_date_source == "date"
    assert resolved.date_header_valid is True
    assert resolved.utc_date_header_valid is False


def test_resolver_returns_missing_when_both_invalid() -> None:
    resolved = resolve_canonical_played_date(
        date_header="????.??.??",
        utc_date_header="2017.02.31",
    )
    assert resolved.canonical_played_date_iso is None
    assert resolved.canonical_date_source == "missing_or_invalid"
    assert resolved.date_header_valid is False
    assert resolved.utc_date_header_valid is False


def test_resolver_accepts_dash_separated_date() -> None:
    resolved = resolve_canonical_played_date(
        date_header="2017-01-15",
        utc_date_header=None,
    )
    assert resolved.canonical_played_date_iso == "2017-01-15"
    assert resolved.canonical_date_source == "date"