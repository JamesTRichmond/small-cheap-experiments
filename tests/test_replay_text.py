"""Plain-text replay of one synthetic FrameTriplet.

Fixtures are fake money and obvious stand-ins. A stored claim_act_mismatch
of 0 is not treated as agreement unless the frames themselves compare.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from probe_runtime.hash_chain import attach_integrity, verify_chain
from probe_runtime.replay_text import format_triplet

GOLDEN = Path(__file__).resolve().parent / "golden" / "replay_text"

SCREENS = (
    "match",
    "mismatch",
    "not_assessable",
    "ts_missing",
    "ts_naive",
    "ts_malformed",
    "missing_integrity",
)


def _load(name: str) -> dict:
    return json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", SCREENS)
def test_golden_screen(name: str) -> None:
    expected = (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")
    assert format_triplet(_load(name)) == expected


def test_golden_raw_json_is_optional_and_under_the_screen() -> None:
    record = _load("match")
    expected = (GOLDEN / "match_with_raw.txt").read_text(encoding="utf-8")
    screen = format_triplet(record)
    rendered = format_triplet(record, include_raw_json=True)
    assert rendered == expected
    assert rendered.startswith(screen)
    _screen, raw = rendered.split("\nRaw JSON:\n", 1)
    assert json.loads(raw) == record


def test_screen_names_the_claim_the_refund_the_doubt_and_the_verdict() -> None:
    lines = format_triplet(_load("match")).splitlines()
    assert any(line.startswith("What the agent said:") for line in lines)
    assert any(line.startswith("What refund call it made:") for line in lines)
    assert any(line.startswith("Whether stated doubt stopped anything:") for line in lines)
    assert any(line.startswith("Verdict:") for line in lines)
    assert "Structural validity: valid" in lines
    assert "Integrity: not verified" in lines
    assert "Claim vs act: matched" in lines
    assert len(lines) <= 16


def test_not_assessable_golden_never_says_matched() -> None:
    text = format_triplet(_load("not_assessable"))
    assert "Claim vs act: not assessable" in text.splitlines()
    assert "matched" not in text


def test_unparseable_claim_is_never_matched() -> None:
    record = _load("match")
    record["frame_claim"]["parsed"]["raw_parse_ok"] = False
    record["scores"]["claim_act_mismatch"] = 0
    text = format_triplet(record)
    assert "Claim vs act: not assessable" in text.splitlines()
    assert "matched" not in text


def test_zero_score_without_a_parsed_claim_is_not_assessable() -> None:
    record = _load("match")
    del record["frame_claim"]["parsed"]
    record["scores"]["claim_act_mismatch"] = 0
    text = format_triplet(record)
    assert "Claim vs act: not assessable" in text.splitlines()
    assert "matched" not in text


def test_zero_score_that_disagrees_with_the_frames_is_not_assessable() -> None:
    record = _load("mismatch")
    record["scores"]["claim_act_mismatch"] = 0
    text = format_triplet(record)
    assert "would_proceed=false" in text
    assert "issued a refund" in text
    assert "Claim vs act: not assessable" in text.splitlines()
    assert "matched" not in text


def test_deny_and_no_refund_is_a_match() -> None:
    record = _load("mismatch")
    record["frame_act"]["executed"] = False
    record["frame_act"]["provider_status"] = "not_sent"
    record["frame_act"]["provider_ref"] = None
    record["scores"]["claim_act_mismatch"] = 0
    text = format_triplet(record)
    assert "did not issue a refund" in text
    assert "Claim vs act: matched" in text.splitlines()
    assert "No refund was issued." in text


def test_stated_doubt_that_stopped_the_refund() -> None:
    record = _load("match")
    record["frame_claim"]["parsed"]["would_proceed"] = False
    record["frame_act"]["executed"] = False
    record["frame_act"]["provider_status"] = "not_sent"
    record["frame_act"]["provider_ref"] = None
    record["frame_humility"] = {
        "claimed_uncertainty": "high",
        "constrained_behavior": True,
        "constraint_action": "queue_human",
    }
    record["scores"]["claim_act_mismatch"] = 0
    text = format_triplet(record)
    assert "Whether stated doubt stopped anything: yes." in text
    assert "Stated doubt stopped the refund." in text
    assert "Claim vs act: matched" in text.splitlines()


def test_missing_timestamp_is_labeled_and_not_guessed() -> None:
    text = format_triplet(_load("ts_missing"))
    assert "ts_utc: missing" in text.splitlines()
    assert re.search(r"\d{4}-\d{2}-\d{2}", text) is None
    record = _load("match")
    record["ts_utc"] = None
    assert "ts_utc: missing" in format_triplet(record).splitlines()


def test_naive_timestamp_is_not_given_a_timezone() -> None:
    text = format_triplet(_load("ts_naive"))
    assert (
        "ts_utc: no timezone (shown as stored, not reformatted): 2026-01-15T12:00:00"
        in text.splitlines()
    )
    assert "+00:00" not in text
    assert "Z" not in text.splitlines()[4]


def test_malformed_timestamp_is_not_rewritten() -> None:
    text = format_triplet(_load("ts_malformed"))
    assert (
        "ts_utc: malformed (shown as stored, not reformatted): not-a-timestamp"
        in text.splitlines()
    )
    record = _load("match")
    record["ts_utc"] = "2026-01-15"
    rendered = format_triplet(record)
    assert (
        "ts_utc: malformed (shown as stored, not reformatted): 2026-01-15"
        in rendered.splitlines()
    )
    assert "T00:00:00" not in rendered


def test_zulu_timestamp_is_shown_as_stored() -> None:
    record = _load("match")
    record["ts_utc"] = "2026-01-15T12:00:00Z"
    text = format_triplet(record)
    assert "ts_utc: 2026-01-15T12:00:00Z" in text.splitlines()
    assert "+00:00" not in text


def test_frame_timestamps_are_not_added_to_the_screen() -> None:
    record = _load("match")
    planted = "1999-09-09T09:09:09Z"
    record["frame_claim"]["ts_utc"] = planted
    record["frame_act"]["ts_utc"] = planted
    record["frame_humility"]["ts_utc"] = planted
    screen = format_triplet(record)
    assert screen.splitlines().count("ts_utc: 2026-01-15T12:00:00+00:00") == 1
    assert planted not in screen
    raw = format_triplet(record, include_raw_json=True)
    above, below = raw.split("\nRaw JSON:\n", 1)
    assert planted not in above
    assert planted in below


def test_a_real_record_hash_is_still_not_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    sealed = attach_integrity(_load("missing_integrity"))
    assert verify_chain([sealed]) is True

    def boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("single-record replay must not recompute a hash")

    monkeypatch.setattr("probe_runtime.hash_chain.record_hash", boom)
    text = format_triplet(sealed)
    assert "Integrity: not verified" in text.splitlines()
    assert "Integrity: verified" not in text.splitlines()
    assert "BROKEN" not in text


@pytest.mark.parametrize(
    ("bad", "kind"),
    [(None, "NoneType"), ([], "list"), ("SYNTHETIC CLAIM A", "str"), (3, "int")],
)
def test_malformed_record_is_a_labeled_line(bad: object, kind: str) -> None:
    text = format_triplet(bad)
    assert text == f"Malformed record: expected a JSON object, got {kind}.\n"
    assert "matched" not in text


def test_malformed_raw_json_does_not_crash() -> None:
    text = format_triplet(["SYNTHETIC"], include_raw_json=True)
    assert text.startswith("Malformed record: expected a JSON object, got list.\n")
    assert "Raw JSON:" in text
    dumped = format_triplet({"ts_utc": object()}, include_raw_json=True)
    assert "Raw JSON: unavailable (value is not JSON)" in dumped
    assert dumped.startswith("Replay screen\n")


def test_empty_object_does_not_crash_and_is_not_a_match() -> None:
    text = format_triplet({})
    assert "Malformed record:" not in text
    assert "Claim vs act: not assessable" in text.splitlines()
    assert "matched" not in text
    assert "Integrity: missing (not verified)" in text.splitlines()
    assert "ts_utc: missing" in text.splitlines()
