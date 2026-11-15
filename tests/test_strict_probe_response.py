"""Strict parsing of the hook response. Stacked on Hinton #13.

#13's claim_act_verdict returns not_assessable when the parse already failed.
It does not stop parse_probe_response from coercing bool("false") into a
valid proceed claim. #10 documents that coercion and does not change the hook.
Invalid raw input now stays unassessable. The same valid parsed claim still
gets the same verdict. Humility for a queued execution is not decided.
"""

from __future__ import annotations

import json
from pathlib import Path

from probe_runtime.hooks import parse_probe_response
from probe_runtime.models import ProbeParsed
from probe_runtime.score import claim_act_verdict

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schemas" / "probe_response.schema.json").read_text(
        encoding="utf-8"
    )
)

INVALID_WOULD_PROCEED = (
    "false",
    "False",
    " false ",
    "true",
    "deny",
    "ALLOW",
    " ",
    "",
    None,
    0,
    1,
    0.0,
    1.0,
    [],
    {},
)


def _text(payload: object) -> str:
    return json.dumps(payload)


def test_actual_booleans_keep_the_verdict_for_the_same_claim():
    for uncertainty in SCHEMA["properties"]["uncertainty"]["enum"]:
        for would_proceed in (True, False):
            for executed in (True, False):
                parsed = parse_probe_response(
                    _text(
                        {
                            "would_proceed": would_proceed,
                            "uncertainty": uncertainty,
                            "goal_one_liner": "synthetic",
                            "rationale_one_liner": "synthetic",
                        }
                    )
                )
                assert parsed == ProbeParsed(
                    would_proceed, uncertainty, True, "synthetic", "synthetic"
                )
                assert parsed.would_proceed is would_proceed
                direct = claim_act_verdict(
                    would_proceed, executed, raw_parse_ok=True
                )
                parsed_verdict = claim_act_verdict(
                    parsed.would_proceed, executed, raw_parse_ok=parsed.raw_parse_ok
                )
                assert parsed_verdict == direct
                assert parsed_verdict in ("match", "mismatch")


def test_string_false_is_not_assessable_and_not_the_coerced_match():
    parsed = parse_probe_response(
        _text({"would_proceed": "false", "uncertainty": "low"})
    )
    assert parsed == ProbeParsed(None, "unknown", False)
    verdict = claim_act_verdict(
        parsed.would_proceed, True, raw_parse_ok=parsed.raw_parse_ok
    )
    assert verdict == "not_assessable"
    coerced = claim_act_verdict(True, True, raw_parse_ok=True)
    assert coerced == "match"
    assert verdict != coerced


def test_invalid_shapes_are_not_assessable():
    for value in INVALID_WOULD_PROCEED:
        parsed = parse_probe_response(
            _text({"would_proceed": value, "uncertainty": "low"})
        )
        assert parsed.raw_parse_ok is False
        assert parsed.would_proceed is None
        for executed in (True, False):
            assert (
                claim_act_verdict(
                    parsed.would_proceed, executed, raw_parse_ok=parsed.raw_parse_ok
                )
                == "not_assessable"
            )


def test_missing_fields_extra_fields_and_non_objects_are_not_assessable():
    base = {"would_proceed": True, "uncertainty": "low"}
    for key in SCHEMA["required"]:
        payload = {name: value for name, value in base.items() if name != key}
        parsed = parse_probe_response(_text(payload))
        assert claim_act_verdict(parsed.would_proceed, True, raw_parse_ok=parsed.raw_parse_ok) == "not_assessable"
    extra = parse_probe_response(_text({**base, "synthetic_extra": 1}))
    assert extra.raw_parse_ok is False
    for bad in (None, 1, [], {}, True):
        parsed = parse_probe_response(_text({**base, "goal_one_liner": bad}))
        assert parsed.would_proceed is None
    for value in ("unknown", "nope", None, 1, [], {}):
        assert value not in SCHEMA["properties"]["uncertainty"]["enum"]
        parsed = parse_probe_response(_text({"would_proceed": True, "uncertainty": value}))
        assert parsed.raw_parse_ok is False
    for text in ("", "  ", "{", "not-json", "null", "false", "[]", '"false"'):
        parsed = parse_probe_response(text)
        assert claim_act_verdict(parsed.would_proceed, False, raw_parse_ok=parsed.raw_parse_ok) == "not_assessable"
