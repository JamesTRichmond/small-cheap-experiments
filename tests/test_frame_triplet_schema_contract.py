"""Structural contract for a sealed sce.replay.v1 FrameTriplet.

The checker applies the schema file. Draft 2020-12 treats `format` as an
annotation, and so does this checker: a timestamp can match `ts_utc`'s
pattern and still be an impossible calendar date. Hash correctness and
claim-versus-act meaning are not structural questions.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from probe_runtime.hash_chain import GENESIS, attach_integrity
from probe_runtime.hooks import post_tool, pre_tool
from probe_runtime.models import FrameTriplet
from probe_runtime.store import ReplayStore

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "frame_triplet.schema.json"
FIXTURES = ROOT / "tests" / "fixtures" / "replay_contract"

_ANNOTATIONS = {"$schema", "$id", "$comment", "title", "description", "examples", "default"}
_ASSERTIONS = {
    "type",
    "const",
    "enum",
    "required",
    "properties",
    "additionalProperties",
    "pattern",
    "format",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
}
_TYPE_NAMES = {"object", "array", "string", "integer", "number", "boolean", "null"}


class SchemaUnsupported(Exception):
    """The schema uses an assertion this checker would otherwise skip."""


def _sha256_obj(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _strict_eq(left: object, right: object) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is bool and type(right) is bool and left == right
    return left == right


def _type_label(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _matches_type(value: object, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return type(value) is int
    if type_name == "number":
        return type(value) in (int, float)
    if type_name == "boolean":
        return type(value) is bool
    if type_name == "null":
        return value is None
    raise SchemaUnsupported(f"unsupported type {type_name}")


def _declared_types(schema: dict) -> list[str]:
    declared = schema["type"]
    names = [declared] if isinstance(declared, str) else declared
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise SchemaUnsupported("type must be a string or a non-empty list of strings")
    return names


def _check_schema(schema: object, path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise SchemaUnsupported(f"{path}: schema must be an object")
    unknown = set(schema) - _ANNOTATIONS - _ASSERTIONS
    if unknown:
        raise SchemaUnsupported(f"{path}: unsupported keywords {sorted(unknown)}")
    if "format" in schema and schema["format"] != "date-time":
        raise SchemaUnsupported(f"{path}: unsupported format {schema['format']!r}")
    if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
        raise SchemaUnsupported(f"{path}: additionalProperties must be a boolean")
    if "type" in schema:
        for name in _declared_types(schema):
            if name not in _TYPE_NAMES:
                raise SchemaUnsupported(f"{path}: unsupported type {name}")
    if "properties" in schema:
        if not isinstance(schema["properties"], dict):
            raise SchemaUnsupported(f"{path}: properties must be an object")
        for key, sub in schema["properties"].items():
            _check_schema(sub, f"{path}.{key}")
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
            raise SchemaUnsupported(f"{path}: required must be a list of strings")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise SchemaUnsupported(f"{path}: enum must be a list")
    if "pattern" in schema and not isinstance(schema["pattern"], str):
        raise SchemaUnsupported(f"{path}: pattern must be a string")


def validate(instance: object, schema: dict) -> list[str]:
    """Return structural errors. `format` is not applied."""
    _check_schema(schema)
    return _validate(instance, schema, "$")


def _validate(instance: object, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    if "const" in schema and not _strict_eq(instance, schema["const"]):
        errors.append(f"{path}: expected const {schema['const']!r}")
    if "enum" in schema and not any(_strict_eq(instance, item) for item in schema["enum"]):
        errors.append(f"{path}: {instance!r} is not in the enum")
    if "type" in schema and not any(_matches_type(instance, name) for name in _declared_types(schema)):
        errors.append(f"{path}: expected {_declared_types(schema)}, got {_type_label(instance)}")
        return errors

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: pattern mismatch")

    if type(instance) in (int, float) and ("minimum" in schema or "maximum" in schema):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")

    describes_object = (
        "properties" in schema or "required" in schema or "additionalProperties" in schema
    )
    if describes_object:
        if not isinstance(instance, dict):
            errors.append(f"{path}: expected object, got {_type_label(instance)}")
            return errors
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}.{key}: required")
        properties = schema.get("properties", {})
        allow_additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                errors.extend(_validate(value, properties[key], f"{path}.{key}"))
            elif allow_additional is False:
                errors.append(f"{path}.{key}: additional property")
    return errors


def _assert_valid(record: dict) -> None:
    assert validate(record, load_schema()) == []


def _assert_invalid(record: dict, fragment: str) -> None:
    found = validate(record, load_schema())
    assert found, "expected schema errors"
    assert any(fragment in item for item in found), found


def test_shipped_schema_is_fully_asserted_except_format_annotation() -> None:
    schema = load_schema()
    _check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    timestamp = schema["properties"]["ts_utc"]
    assert timestamp["format"] == "date-time"
    assert "pattern" in timestamp


def test_validator_rejects_keywords_it_would_skip() -> None:
    with pytest.raises(SchemaUnsupported):
        validate({}, {"type": "object", "unevaluatedProperties": False})


def test_schema_properties_match_frame_triplet_fields() -> None:
    schema = load_schema()
    model_fields = {item.name for item in fields(FrameTriplet)}
    assert set(schema["properties"]) == model_fields
    assert set(schema["required"]) == model_fields


def test_one_triplet_timestamp_and_no_per_frame_timestamps() -> None:
    schema = load_schema()
    assert "ts_utc" in schema["properties"]
    for name in ("frame_claim", "frame_act", "frame_humility"):
        assert "ts_utc" not in schema["properties"][name]["properties"]


def test_prev_hash_pattern_accepts_the_chain_genesis_constant() -> None:
    pattern = load_schema()["properties"]["integrity"]["properties"]["prev_hash"]["pattern"]
    assert re.search(pattern, GENESIS)


@pytest.mark.parametrize(
    "name",
    ["valid_matched.json", "valid_mismatched.json", "valid_unparsed.json"],
)
def test_synthetic_fixtures_are_structurally_valid(name: str) -> None:
    record = load_fixture(name)
    _assert_valid(record)
    assert record["frame_claim"]["probe_text_hash"] == _sha256_obj("SYNTHETIC_NON_PROBE")
    assert "SYNTHETIC_NON_PROBE" not in json.dumps(record)


def test_schema_accepts_disagreement_and_unparsed_claims() -> None:
    """A sealed disagreement is still a valid record. So is an unparsed claim."""
    mismatched = load_fixture("valid_mismatched.json")
    assert mismatched["frame_claim"]["parsed"]["would_proceed"] is False
    assert mismatched["frame_act"]["executed"] is True
    _assert_valid(mismatched)

    unparsed = load_fixture("valid_unparsed.json")
    assert unparsed["frame_claim"]["parsed"]["raw_parse_ok"] is False
    assert unparsed["frame_claim"]["parsed"]["would_proceed"] is None
    assert unparsed["scores"]["claim_act_mismatch"] == 0
    _assert_valid(unparsed)


def test_fixtures_round_trip_through_to_dict() -> None:
    for name in ("valid_matched.json", "valid_mismatched.json", "valid_unparsed.json"):
        raw = load_fixture(name)
        assert FrameTriplet(**raw).to_dict() == raw


def test_fixtures_do_not_carry_holdout_probes() -> None:
    real_ids: list[str] = []
    real_hashes: list[str] = []
    for path in sorted((ROOT / "probes" / "holdout").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            real_ids.append(row["probe_id"])
            real_hashes.append(_sha256_obj(row["text"]))
    blob = "\n".join(path.read_text(encoding="utf-8") for path in sorted(FIXTURES.glob("*.json")))
    assert [probe_id for probe_id in real_ids if probe_id in blob] == []
    assert [digest for digest in real_hashes if digest in blob] == []


def test_empty_object_fails_required_frames() -> None:
    found = validate({}, load_schema())
    for key in ("frame_claim", "frame_act", "frame_humility", "scores", "integrity"):
        assert any(f"$.{key}: required" == item for item in found), found


@pytest.mark.parametrize("frame", ["frame_claim", "frame_act", "frame_humility"])
def test_each_frame_is_required(frame: str) -> None:
    record = load_fixture("valid_matched.json")
    del record[frame]
    _assert_invalid(record, f"$.{frame}: required")


@pytest.mark.parametrize("frame", ["frame_claim", "frame_act", "frame_humility"])
def test_each_frame_must_be_an_object(frame: str) -> None:
    record = load_fixture("valid_matched.json")
    record[frame] = []
    _assert_invalid(record, f"$.{frame}: expected")


@pytest.mark.parametrize(
    ("path", "value", "fragment"),
    [
        (("schema_version",), "sce.replay.v2", "expected const"),
        (("triplet_id",), "", "minLength"),
        (("agent_id",), 7, "expected"),
        (("ts_utc",), "2026-11-07T23:00:00", "pattern mismatch"),
        (("ts_utc",), "2026-11-07T23:00:00+05:00", "pattern mismatch"),
        (("ts_utc",), "2026-11-07T23:00:00+0000", "pattern mismatch"),
        (("surface", "tool_name"), "payout.create", "expected const"),
        (("surface", "tool_args_hash"), "0" * 63, "pattern mismatch"),
        (("surface", "amount_minor"), True, "expected"),
        (("surface", "amount_minor"), 1.5, "expected"),
        (("frame_claim", "parsed", "would_proceed"), "true", "expected"),
        (("frame_claim", "parsed", "uncertainty"), "sometimes", "not in the enum"),
        (("frame_claim", "parsed", "raw_parse_ok"), "yes", "expected"),
        (("frame_act", "executed"), "true", "expected"),
        (("frame_act", "executed"), 1, "expected"),
        (("frame_act", "blocked_by_gate"), 0, "expected"),
        (("frame_humility", "constraint_action"), "bogus", "not in the enum"),
        (("scores", "claim_act_mismatch"), True, "expected"),
        (("scores", "claim_act_mismatch"), 2, "above maximum"),
        (("scores", "green_then_execute"), -1, "below minimum"),
        (("scores", "humility_failed_to_block"), "0", "expected"),
        (("integrity", "prev_hash"), "nope", "pattern mismatch"),
        (("integrity", "record_hash"), "A" * 64, "pattern mismatch"),
    ],
)
def test_bad_types_and_enums_fail(path: tuple[str, ...], value: object, fragment: str) -> None:
    record = load_fixture("valid_matched.json")
    cursor = record
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    _assert_invalid(record, fragment)


def test_negative_amount_remains_structurally_valid() -> None:
    """The sealed surface copies the caller amount. The tool schema is separate."""
    record = load_fixture("valid_matched.json")
    record["surface"]["amount_minor"] = -50
    _assert_valid(record)


def test_null_merchant_ref_and_probe_family_are_valid() -> None:
    record = load_fixture("valid_matched.json")
    record["surface"]["merchant_ref"] = None
    record["frame_claim"]["probe_family"] = None
    record["frame_act"]["provider_ref"] = None
    _assert_valid(record)


def test_model_constraint_actions_are_valid() -> None:
    record = load_fixture("valid_matched.json")
    for action in ("none", "refuse", "escalate", "queue_human"):
        record["frame_humility"]["constraint_action"] = action
        _assert_valid(record)


def test_additional_properties_fail_at_each_object() -> None:
    top = load_fixture("valid_matched.json")
    top["note"] = "extra"
    _assert_invalid(top, "$.note: additional property")

    claim = load_fixture("valid_matched.json")
    claim["frame_claim"]["parsed"]["goal_one_liner"] = "not stored on the frame"
    _assert_invalid(claim, "$.frame_claim.parsed.goal_one_liner: additional property")

    scores = load_fixture("valid_matched.json")
    scores["scores"]["extra_bit"] = 0
    _assert_invalid(scores, "$.scores.extra_bit: additional property")


def test_unsealed_default_integrity_is_not_a_replay_record() -> None:
    record = load_fixture("valid_matched.json")
    record["integrity"] = {}
    found = validate(record, load_schema())
    assert any("$.integrity.prev_hash: required" == item for item in found), found
    assert any("$.integrity.record_hash: required" == item for item in found), found


def test_missing_agent_id_fails() -> None:
    record = load_fixture("valid_matched.json")
    del record["agent_id"]
    _assert_invalid(record, "$.agent_id: required")


def test_emitted_hook_records_and_to_dict_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "probe_runtime.hooks.sample_probe",
        lambda rng=None: {
            "probe_id": "synthetic-non-probe",
            "family": "synthetic",
            "text": "SYNTHETIC_NON_PROBE",
        },
    )
    cases = [
        ('{"would_proceed": true, "uncertainty": "none"}', True, "fake-log-ref", {"merchant_ref": "fake-merchant"}),
        ('{"would_proceed": false, "uncertainty": "none"}', True, None, {}),
        ("not-json", False, None, {"merchant_ref": "fake-merchant"}),
    ]
    store = ReplayStore(tmp_path / "replay.jsonl")
    schema = load_schema()
    for response, executed, provider_ref, tool_args in cases:
        pre = pre_tool(
            agent_id="agent-synthetic",
            session_id="session-synthetic",
            tool_name="refund.create",
            tool_args=tool_args,
            amount_minor=100,
            currency="XXX",
            probe_response_text=response,
            store=store,
        )
        assert pre["probe_id"] == "synthetic-non-probe"
        sealed = post_tool(
            triplet_id=pre["triplet_id"],
            executed=executed,
            provider_status="log_only",
            provider_ref=provider_ref,
            store=store,
        )
        assert sealed["frame_claim"]["probe_text_hash"] == _sha256_obj("SYNTHETIC_NON_PROBE")
        assert "SYNTHETIC_NON_PROBE" not in json.dumps(sealed)
        assert validate(sealed, schema) == []
        assert FrameTriplet(**sealed).to_dict() == sealed
        for frame in ("frame_claim", "frame_act", "frame_humility"):
            assert "ts_utc" not in sealed[frame]


def test_attach_integrity_shape_satisfies_the_schema() -> None:
    body = load_fixture("valid_matched.json")
    body.pop("integrity")
    sealed = attach_integrity(body)
    assert set(sealed["integrity"]) == {"prev_hash", "record_hash"}
    _assert_valid(sealed)
