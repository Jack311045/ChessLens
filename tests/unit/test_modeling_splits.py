from __future__ import annotations

from datetime import date

import pytest

from chesslens.modeling.splits import (
    TemporalRange,
    assign_temporal_split,
    parse_played_date,
    validate_temporal_ranges,
)


def test_parse_played_date_variants() -> None:
    parsed_dot = parse_played_date("2013.01.07")
    assert parsed_dot.status == "ok"
    assert parsed_dot.played_date_iso == "2013-01-07"

    parsed_dash = parse_played_date("2013-01-08")
    assert parsed_dash.status == "ok"
    assert parsed_dash.played_date_iso == "2013-01-08"

    parsed_missing = parse_played_date("????.??.??")
    assert parsed_missing.status == "invalid"


def test_assign_temporal_split_by_date() -> None:
    train = TemporalRange(start_date=date(2013, 1, 1), end_date=date(2013, 1, 20))
    validation = TemporalRange(start_date=date(2013, 1, 21), end_date=date(2013, 1, 25))
    test = TemporalRange(start_date=date(2013, 1, 26), end_date=date(2013, 1, 31))

    split, reason = assign_temporal_split(
        parsed_date=date(2013, 1, 22),
        date_status="ok",
        train=train,
        validation=validation,
        test=test,
        missing_or_invalid_date_policy="reject",
    )

    assert split == "validation"
    assert reason == "date_interval"


def test_assign_temporal_split_policy_assignment() -> None:
    train = TemporalRange(start_date=date(2013, 1, 1), end_date=date(2013, 1, 20))
    validation = TemporalRange(start_date=date(2013, 1, 21), end_date=date(2013, 1, 25))
    test = TemporalRange(start_date=date(2013, 1, 26), end_date=date(2013, 1, 31))

    split, reason = assign_temporal_split(
        parsed_date=None,
        date_status="missing",
        train=train,
        validation=validation,
        test=test,
        missing_or_invalid_date_policy="assign_train",
    )

    assert split == "train"
    assert reason == "policy_assigned_missing_or_invalid"


def test_validate_temporal_ranges_rejects_overlap() -> None:
    train = TemporalRange(start_date=date(2013, 1, 1), end_date=date(2013, 1, 20))
    validation = TemporalRange(start_date=date(2013, 1, 20), end_date=date(2013, 1, 25))
    test = TemporalRange(start_date=date(2013, 1, 26), end_date=date(2013, 1, 31))

    with pytest.raises(ValueError, match="must not overlap"):
        validate_temporal_ranges(train=train, validation=validation, test=test)
