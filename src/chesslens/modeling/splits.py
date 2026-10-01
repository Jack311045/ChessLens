"""Temporal split parsing and assignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

DatePolicy = Literal["reject", "assign_train", "assign_validation", "assign_test"]


@dataclass(frozen=True)
class TemporalRange:
    start_date: date
    end_date: date


@dataclass(frozen=True)
class PlayedDateParseResult:
    played_date_raw: str | None
    played_date_iso: str | None
    parsed_date: date | None
    status: Literal["ok", "missing", "invalid"]


def parse_played_date(value: str | None) -> PlayedDateParseResult:
    if value is None:
        return PlayedDateParseResult(
            played_date_raw=None,
            played_date_iso=None,
            parsed_date=None,
            status="missing",
        )

    text = value.strip()
    if not text:
        return PlayedDateParseResult(
            played_date_raw=value,
            played_date_iso=None,
            parsed_date=None,
            status="missing",
        )

    if "?" in text:
        return PlayedDateParseResult(
            played_date_raw=text,
            played_date_iso=None,
            parsed_date=None,
            status="invalid",
        )

    formats = ("%Y.%m.%d", "%Y-%m-%d")
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        return PlayedDateParseResult(
            played_date_raw=text,
            played_date_iso=parsed.isoformat(),
            parsed_date=parsed,
            status="ok",
        )

    return PlayedDateParseResult(
        played_date_raw=text,
        played_date_iso=None,
        parsed_date=None,
        status="invalid",
    )


def validate_temporal_ranges(
    *,
    train: TemporalRange,
    validation: TemporalRange,
    test: TemporalRange,
) -> None:
    if train.start_date > train.end_date:
        raise ValueError("train split start_date must be <= end_date")
    if validation.start_date > validation.end_date:
        raise ValueError("validation split start_date must be <= end_date")
    if test.start_date > test.end_date:
        raise ValueError("test split start_date must be <= end_date")

    if train.end_date >= validation.start_date:
        raise ValueError("train and validation ranges must not overlap")
    if validation.end_date >= test.start_date:
        raise ValueError("validation and test ranges must not overlap")


def assign_temporal_split(
    *,
    parsed_date: date | None,
    date_status: Literal["ok", "missing", "invalid"],
    train: TemporalRange,
    validation: TemporalRange,
    test: TemporalRange,
    missing_or_invalid_date_policy: DatePolicy,
) -> tuple[str | None, str]:
    if parsed_date is not None:
        if train.start_date <= parsed_date <= train.end_date:
            return "train", "date_interval"
        if validation.start_date <= parsed_date <= validation.end_date:
            return "validation", "date_interval"
        if test.start_date <= parsed_date <= test.end_date:
            return "test", "date_interval"
        return None, "outside_declared_ranges"

    if missing_or_invalid_date_policy == "reject":
        if date_status == "missing":
            return None, "missing_played_date"
        return None, "invalid_played_date"

    if missing_or_invalid_date_policy == "assign_train":
        return "train", "policy_assigned_missing_or_invalid"
    if missing_or_invalid_date_policy == "assign_validation":
        return "validation", "policy_assigned_missing_or_invalid"
    if missing_or_invalid_date_policy == "assign_test":
        return "test", "policy_assigned_missing_or_invalid"

    raise ValueError(
        f"Unsupported missing_or_invalid_date_policy: {missing_or_invalid_date_policy!r}"
    )
