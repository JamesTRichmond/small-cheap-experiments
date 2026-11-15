"""Strict probe-response parsing.

Invalid raw inputs used to be coerced with bool() or accepted with a default
uncertainty. Those rows are now unassessable. The same valid parsed claim and
executed flag still produce the same score_frames output. Score functions are
not modified here. raw_parse_ok is parse validity, not probe/response pairing.
"""

from __future__ import annotations

import json
from pathlib import Path

from probe_runtime.gate import decide
from probe_runtime.hooks import parse_probe_response
from probe_runtime.models import ProbeParsed
from probe_runtime.score import resolve_constraint_action, score_frames

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "probe_response.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

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


def _label(parsed: ProbeParsed) -> str:
    if not parsed.raw_parse_ok or parsed.would_proceed is None:
        return "not assessable"
    return "assessable"


def test_schema_file_is_the_public_object_contract():
    assert SCHEMA["type"] == "object"
    assert SCHEMA["additionalProperties"] is False
    assert SCHEMA["required"] == ["would_proceed", "uncertainty"]
    assert SCHEMA["properties"]["would_proceed"]["type"] == "boolean"
    assert SCHEMA["properties"]["uncertainty"]["type"] == "string"
    assert SCHEMA["properties"]["uncertainty"]["enum"] == ["none", "low", "high"]
    assert SCHEMA["properties"]["goal_one_liner"]["type"] == "string"
    assert SCHEMA["properties"]["rationale_one_liner"]["type"] == "string"


def test_actual_booleans_and_enum_uncertainty_parse():
    for uncertainty in SCHEMA["properties"]["uncertainty"]["enum"]:
        for would_proceed in (True, False):
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
            direct = ProbeParsed(
                would_proceed=would_proceed,
                uncertainty=uncertainty,
                raw_parse_ok=True,
                goal_one_liner="synthetic",
                rationale_one_liner="synthetic",
            )
            assert parsed == direct
            assert parsed.would_proceed is would_proceed
            assert _label(parsed) == "assessable"


def test_omitted_optional_strings_stay_empty_without_coercion():
    parsed = parse_probe_response(
        _text({"would_proceed": False, "uncertainty": "low"})
    )
    assert parsed == ProbeParsed(False, "low", True, "", "")
    assert parse_probe_response(
        _text(
            {
                "would_proceed": True,
                "uncertainty": "none",
                "goal_one_liner": "",
                "rationale_one_liner": "",
            }
        )
    ).goal_one_liner == ""


def test_valid_parsed_claims_keep_exact_score_outputs():
    """Same valid claim and executed flag: score_frames is unchanged."""
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
                direct = ProbeParsed(
                    would_proceed, uncertainty, True, "synthetic", "synthetic"
                )
                assert parsed == direct
                decision = decide(parsed)
                _, action = resolve_constraint_action(
                    decision=decision, executed=executed, parsed=parsed
                )
                _, direct_action = resolve_constraint_action(
                    decision=decide(direct), executed=executed, parsed=direct
                )
                assert action == direct_action
                assert score_frames(parsed, executed, action) == score_frames(
                    direct, executed, direct_action
                )


def test_string_false_is_unassessable_and_not_the_coerced_green_score():
    """bool('false') used to become True and could score as green. It must not."""
    parsed = parse_probe_response(
        _text({"would_proceed": "false", "uncertainty": "low"})
    )
    assert parsed == ProbeParsed(None, "unknown", False)
    assert _label(parsed) == "not assessable"
    assert decide(parsed) != "allow"
    executed_scores = score_frames(parsed, True, "none")
    assert executed_scores["claim_act_mismatch"] == 0
    assert executed_scores["green_then_execute"] == 0
    coerced = ProbeParsed(True, "low", True)
    coerced_scores = score_frames(coerced, True, "none")
    assert coerced_scores["green_then_execute"] == 1
    assert executed_scores != coerced_scores
    assert _label(parsed) != "assessable"


def test_invalid_would_proceed_values_are_unassessable():
    for value in INVALID_WOULD_PROCEED:
        parsed = parse_probe_response(
            _text({"would_proceed": value, "uncertainty": "low"})
        )
        assert parsed.raw_parse_ok is False
        assert parsed.would_proceed is None
        assert _label(parsed) == "not assessable"
        assert decide(parsed) == "queue_human"
        assert decide(parsed) != "allow"


def test_missing_required_fields_are_unassessable():
    required = SCHEMA["required"]
    base = {"would_proceed": True, "uncertainty": "low"}
    for key in required:
        payload = {name: value for name, value in base.items() if name != key}
        parsed = parse_probe_response(_text(payload))
        assert parsed == ProbeParsed(None, "unknown", False)
        assert _label(parsed) == "not assessable"
    assert _label(parse_probe_response(_text({}))) == "not assessable"


def test_extra_fields_and_bad_optional_types_are_unassessable():
    extra = {
        "would_proceed": True,
        "uncertainty": "none",
        "synthetic_extra": 1,
    }
    assert parse_probe_response(_text(extra)).raw_parse_ok is False
    for bad in (None, 1, [], {}, True):
        parsed = parse_probe_response(
            _text(
                {
                    "would_proceed": True,
                    "uncertainty": "low",
                    "goal_one_liner": bad,
                }
            )
        )
        assert parsed.raw_parse_ok is False
        assert parsed.would_proceed is None


def test_uncertainty_outside_schema_enum_is_unassessable():
    enum = SCHEMA["properties"]["uncertainty"]["enum"]
    for value in ("unknown", "nope", "HIGH", None, 1, [], {}):
        assert value not in enum
        parsed = parse_probe_response(
            _text({"would_proceed": True, "uncertainty": value})
        )
        assert parsed.raw_parse_ok is False
        assert _label(parsed) == "not assessable"
        assert decide(parsed) != "allow"


def test_non_objects_and_malformed_text_are_unassessable():
    for text in (
        "",
        "   ",
        "{",
        "not-json",
        "null",
        "true",
        "false",
        "0",
        "[]",
        '"false"',
    ):
        parsed = parse_probe_response(text)
        assert parsed == ProbeParsed(None, "unknown", False)
        assert _label(parsed) == "not assessable"
        assert decide(parsed) != "allow"


def test_non_string_input_is_unassessable():
    parsed = parse_probe_response(b'{"would_proceed": true, "uncertainty": "low"}')  # type: ignore[arg-type]
    assert parsed.raw_parse_ok is False
    assert parsed.would_proceed is None
