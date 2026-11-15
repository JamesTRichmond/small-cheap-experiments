"""Why line for mismatch and not-assessable screens.

The line is built from fields that are present. It does not score amounts
or order ids. Currency in the fixtures is FAKE.
"""

from __future__ import annotations

import json
from pathlib import Path

from probe_runtime.replay_text import format_triplet

GOLDEN = Path(__file__).resolve().parent / "golden"
TEXT = GOLDEN / "replay_text"
WHY = GOLDEN / "replay_why"


def _load(directory: Path, name: str) -> dict:
    return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))


def _why(text: str) -> str | None:
    lines = [line for line in text.splitlines() if line.startswith("Why (shown, not scored): ")]
    if not lines:
        return None
    assert len(lines) == 1
    return lines[0]


def test_mismatch_golden_says_deny_and_names_the_tool() -> None:
    text = format_triplet(_load(TEXT, "mismatch"))
    assert text == (TEXT / "mismatch.txt").read_text(encoding="utf-8")
    assert _why(text) == (
        "Why (shown, not scored): claim said DENY; act was refund.create"
    )
    assert "matched" not in text


def test_unparseable_golden_names_the_parse_fields() -> None:
    text = format_triplet(_load(TEXT, "not_assessable"))
    assert text == (TEXT / "not_assessable.txt").read_text(encoding="utf-8")
    assert _why(text) == (
        "Why (shown, not scored): claim unparseable; "
        "parse error: raw_parse_ok, would_proceed; "
        "act was refund.create; no verdict stored"
    )


def test_match_screen_has_no_why_line() -> None:
    text = format_triplet(_load(TEXT, "match"))
    assert _why(text) is None
    assert "Why (shown, not scored):" not in text


def test_side_by_side_values_are_not_judged() -> None:
    record = _load(WHY, "side_by_side")
    text = format_triplet(record)
    assert text == (WHY / "side_by_side.txt").read_text(encoding="utf-8")
    why = _why(text)
    assert why is not None
    assert "amount_minor surface=100 act=250" in why
    assert "order_id surface=SYNTH-ORDER-1 act=SYNTH-ORDER-2" in why
    assert "merchant_ref surface=SYNTH-MERCHANT act=SYNTH-MERCHANT-ACT" in why
    assert "equal" not in why
    assert "differ" not in why
    assert "matched" not in why
    assert "shown, not scored" in why


def test_equal_amounts_are_still_not_called_a_match() -> None:
    record = _load(WHY, "side_by_side")
    record["frame_act"]["amount_minor"] = 100
    record["frame_act"]["order_id"] = "SYNTH-ORDER-1"
    record["scores"].pop("claim_act_verdict")
    record.pop("claim_act_verdict")
    why = _why(format_triplet(record))
    assert why is not None
    assert "amount_minor surface=100 act=100" in why
    assert "order_id surface=SYNTH-ORDER-1 act=SYNTH-ORDER-1" in why
    assert "equal" not in why
    assert "matched" not in why
    assert "no verdict stored" in why
