"""Executable UTC checks for the single triplet timestamp.

Draft 2020-12 `format` is an annotation. A type check, and a shape pattern
that still matches month 13, both accept values that are not UTC instants.
`classify_ts_utc` parses the value. There is one `ts_utc` on the triplet,
not a timestamp inside each frame.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "frame_triplet.schema.json"

_CANONICAL = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?(Z|\+00:00)$"
)
_NAIVE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?$"
)
_OFFSET = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(\.[0-9]{1,6})?(Z|[+-][0-9]{2}:[0-9]{2}|[+-][0-9]{4})$"
)


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def classify_ts_utc(value: object) -> str:
    """Return ok, naive, non_utc, malformed, or wrong_type."""
    if type(value) is not str:
        return "wrong_type"
    if _CANONICAL.fullmatch(value):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return "malformed"
        if parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0):
            return "ok"
        return "malformed"
    if _NAIVE.fullmatch(value):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return "malformed"
        if parsed.tzinfo is None:
            return "naive"
        return "malformed"
    if _OFFSET.fullmatch(value):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return "malformed"
        offset = parsed.utcoffset()
        if offset is None:
            return "naive"
        if offset != timedelta(0):
            return "non_utc"
        return "malformed"
    return "malformed"


def annotation_only_accepts(value: object, prop: dict) -> bool:
    """Apply type and pattern. Ignore `format`, as draft 2020-12 does by default."""
    if prop.get("type") == "string" and type(value) is not str:
        return False
    pattern = prop.get("pattern")
    if pattern and type(value) is str and re.search(pattern, value) is None:
        return False
    if prop.get("type") == "string":
        return type(value) is str
    return True


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("2026-11-07T23:00:00+00:00", "ok"),
        ("2026-11-07T23:00:00.123456+00:00", "ok"),
        ("2026-11-07T23:00:00Z", "ok"),
        ("2026-11-07T23:00:00.000001Z", "ok"),
        ("2024-02-29T00:00:00Z", "ok"),
        ("2026-11-07T23:00:00", "naive"),
        ("2026-11-07T23:00:00.123456", "naive"),
        ("2026-11-07T23:00:00+05:00", "non_utc"),
        ("2026-11-07T23:00:00-05:00", "non_utc"),
        ("2026-13-01T00:00:00Z", "malformed"),
        ("2026-11-31T00:00:00+00:00", "malformed"),
        ("2026-02-29T00:00:00Z", "malformed"),
        ("2026-11-07T23:00:00+0000", "malformed"),
        ("2026-11-07 23:00:00+00:00", "malformed"),
        ("2026-11-07T23:00:00z", "malformed"),
        ("2026-11-07", "malformed"),
        ("yesterday", "malformed"),
        ("", "malformed"),
        (None, "wrong_type"),
        (0, "wrong_type"),
        (True, "wrong_type"),
    ],
)
def test_timestamp_class_is_executed(value: object, kind: str) -> None:
    assert classify_ts_utc(value) == kind


def test_hook_isoformat_is_ok() -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    assert classify_ts_utc(stamp) == "ok"
    assert _CANONICAL.fullmatch(stamp)


def test_canonical_pattern_does_not_reject_impossible_dates() -> None:
    value = "2026-13-01T00:00:00Z"
    assert _CANONICAL.fullmatch(value)
    assert classify_ts_utc(value) == "malformed"


def test_format_annotation_accepts_impossible_calendar_date() -> None:
    prop = load_schema()["properties"]["ts_utc"]
    value = "2026-13-01T00:00:00Z"
    assert prop.get("type") == "string"
    assert prop.get("format") in (None, "date-time")
    assert annotation_only_accepts(value, prop) is True
    assert classify_ts_utc(value) == "malformed"


def test_type_string_alone_accepts_a_naive_timestamp() -> None:
    prop = load_schema()["properties"]["ts_utc"]
    naive = "2026-11-07T23:00:00"
    assert prop.get("type") == "string"
    assert isinstance(naive, str)
    assert classify_ts_utc(naive) == "naive"


def test_schema_declares_one_triplet_timestamp() -> None:
    schema = load_schema()
    assert "ts_utc" in schema["properties"]
    required = schema.get("required", [])
    assert "ts_utc" in required
    for name in ("frame_claim", "frame_act", "frame_humility"):
        frame = schema["properties"].get(name, {})
        assert "ts_utc" not in frame.get("properties", {})
