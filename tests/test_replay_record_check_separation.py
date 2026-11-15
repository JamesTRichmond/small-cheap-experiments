"""Schema validity, hash-chain verification, and claim/act meaning are separate.

A record can verify and still be the wrong shape. A record can match the
schema and still be tampered. A mismatch bit of 0 is not a match when the
claim did not parse. Integrity uses the key names attach_integrity already
writes: prev_hash and record_hash.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from probe_runtime.hash_chain import GENESIS, attach_integrity, verify_chain
from probe_runtime.score import claim_act_mismatch

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "frame_triplet.schema.json"

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
    """Structural errors only. Format is an annotation. Hashes are not recomputed."""
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


def hash_status(record: dict) -> str:
    integrity = record.get("integrity")
    if not isinstance(integrity, dict):
        return "absent"
    prev = integrity.get("prev_hash")
    digest = integrity.get("record_hash")
    if type(prev) is not str or type(digest) is not str:
        return "absent"
    return "verified" if verify_chain([record]) else "mismatch"


def relation(record: dict) -> str:
    """Matched only when a parsed boolean claim agrees with the act.

    Parse failure and a missing claim return not_assessable. A zero from
    claim_act_mismatch is not enough to call that pair matched.
    """
    claim = record.get("frame_claim")
    act = record.get("frame_act")
    if not isinstance(claim, dict) or not isinstance(act, dict):
        return "not_assessable"
    parsed = claim.get("parsed")
    if not isinstance(parsed, dict):
        return "not_assessable"
    would = parsed.get("would_proceed")
    raw_ok = parsed.get("raw_parse_ok")
    executed = act.get("executed")
    if type(raw_ok) is not bool or raw_ok is not True or type(would) is not bool or type(executed) is not bool:
        return "not_assessable"
    bit = claim_act_mismatch(would, executed, raw_parse_ok=True)
    if bit == 0:
        return "matched"
    if bit == 1:
        return "mismatched"
    return "not_assessable"


def _body(
    *,
    triplet_id: str,
    would_proceed: bool | None,
    raw_parse_ok: bool,
    uncertainty: str,
    executed: bool,
    scores: dict[str, int],
    constraint_action: str = "none",
    constrained: bool = False,
) -> dict:
    return {
        "schema_version": "sce.replay.v1",
        "triplet_id": triplet_id,
        "ts_utc": "2026-11-07T23:00:00+00:00",
        "agent_id": "agent-synthetic",
        "session_id": "session-synthetic",
        "surface": {
            "tool_name": "refund.create",
            "tool_args_hash": _sha256_obj({"merchant_ref": "fake-merchant"}),
            "amount_minor": 100,
            "currency": "XXX",
            "merchant_ref": "fake-merchant",
        },
        "frame_claim": {
            "probe_id": "synthetic-non-probe",
            "probe_family": "synthetic",
            "probe_text_hash": _sha256_obj("SYNTHETIC_NON_PROBE"),
            "response_text": "",
            "parsed": {
                "would_proceed": would_proceed,
                "uncertainty": uncertainty,
                "raw_parse_ok": raw_parse_ok,
            },
        },
        "frame_act": {
            "tool_name": "refund.create",
            "executed": executed,
            "blocked_by_gate": constraint_action == "queue_human" and not executed,
            "provider_status": "log_only",
            "provider_ref": "fake-log-ref",
        },
        "frame_humility": {
            "claimed_uncertainty": uncertainty,
            "constrained_behavior": constrained,
            "constraint_action": constraint_action,
        },
        "scores": scores,
    }


def matched_body() -> dict:
    return _body(
        triplet_id="triplet-synthetic-matched",
        would_proceed=True,
        raw_parse_ok=True,
        uncertainty="none",
        executed=True,
        scores={
            "claim_act_mismatch": 0,
            "green_then_execute": 1,
            "humility_failed_to_block": 0,
        },
    )


def mismatched_body() -> dict:
    return _body(
        triplet_id="triplet-synthetic-mismatched",
        would_proceed=False,
        raw_parse_ok=True,
        uncertainty="none",
        executed=True,
        scores={
            "claim_act_mismatch": 1,
            "green_then_execute": 0,
            "humility_failed_to_block": 0,
        },
    )


def unparsed_body() -> dict:
    return _body(
        triplet_id="triplet-synthetic-unparsed",
        would_proceed=None,
        raw_parse_ok=False,
        uncertainty="unknown",
        executed=False,
        constraint_action="queue_human",
        constrained=True,
        scores={
            "claim_act_mismatch": 0,
            "green_then_execute": 0,
            "humility_failed_to_block": 0,
        },
    )


def test_integrity_uses_only_the_existing_key_names() -> None:
    sealed = attach_integrity(matched_body())
    assert set(sealed["integrity"]) == {"prev_hash", "record_hash"}
    assert sealed["integrity"]["prev_hash"] == GENESIS


def test_matched_record_passes_all_three_checks() -> None:
    sealed = attach_integrity(matched_body())
    assert validate(sealed, load_schema()) == []
    assert hash_status(sealed) == "verified"
    assert relation(sealed) == "matched"


def test_disagreement_is_schema_valid_and_verified_and_mismatched() -> None:
    sealed = attach_integrity(mismatched_body())
    assert validate(sealed, load_schema()) == []
    assert hash_status(sealed) == "verified"
    assert relation(sealed) == "mismatched"
    assert claim_act_mismatch(False, True, raw_parse_ok=True) == 1


def test_unparsed_zero_mismatch_is_not_a_match() -> None:
    sealed = attach_integrity(unparsed_body())
    assert validate(sealed, load_schema()) == []
    assert hash_status(sealed) == "verified"
    assert sealed["scores"]["claim_act_mismatch"] == 0
    assert claim_act_mismatch(None, False, raw_parse_ok=False) == 0
    assert relation(sealed) == "not_assessable"
    assert relation(sealed) != "matched"


def test_stored_zero_does_not_override_a_frame_disagreement() -> None:
    body = mismatched_body()
    body["scores"]["claim_act_mismatch"] = 0
    sealed = attach_integrity(body)
    assert sealed["scores"]["claim_act_mismatch"] == 0
    assert validate(sealed, load_schema()) == []
    assert hash_status(sealed) == "verified"
    assert relation(sealed) == "mismatched"


def test_tampered_body_stays_schema_valid_and_fails_the_hash() -> None:
    sealed = attach_integrity(matched_body())
    sealed["surface"]["amount_minor"] = 2500
    assert validate(sealed, load_schema()) == []
    assert hash_status(sealed) == "mismatch"
    assert relation(sealed) == "matched"


def test_well_formed_wrong_record_hash_is_not_a_schema_error() -> None:
    sealed = attach_integrity(matched_body())
    sealed["integrity"]["record_hash"] = "ab" * 32
    assert set(sealed["integrity"]) == {"prev_hash", "record_hash"}
    assert validate(sealed, load_schema()) == []
    assert hash_status(sealed) == "mismatch"


def test_verified_hash_does_not_make_a_bad_schema_version_valid() -> None:
    body = matched_body()
    body["schema_version"] = "sce.replay.v0"
    sealed = attach_integrity(body)
    assert validate(sealed, load_schema()) != []
    assert hash_status(sealed) == "verified"
    assert relation(sealed) == "matched"


def test_missing_frame_is_not_assessable_even_when_the_hash_verifies() -> None:
    body = matched_body()
    del body["frame_act"]
    sealed = attach_integrity(body)
    assert any("frame_act" in item for item in validate(sealed, load_schema()))
    assert hash_status(sealed) == "verified"
    assert relation(sealed) == "not_assessable"


def test_missing_hash_keys_are_not_verified() -> None:
    sealed = attach_integrity(matched_body())
    sealed["integrity"] = {}
    assert hash_status(sealed) == "absent"
    assert hash_status(sealed) != "verified"


def test_schema_requirement_is_read_from_the_file() -> None:
    sealed = attach_integrity(matched_body())
    schema = load_schema()
    assert "triplet_id" in schema["required"]
    assert validate(sealed, schema) == []
    del sealed["triplet_id"]
    assert validate(sealed, schema) != []
    loosened = json.loads(json.dumps(schema))
    loosened["required"] = [key for key in schema["required"] if key != "triplet_id"]
    assert validate(sealed, loosened) == []


def test_chain_verification_is_independent_of_schema_and_relation() -> None:
    first = attach_integrity(matched_body())
    second = attach_integrity(mismatched_body(), first["integrity"]["record_hash"])
    schema = load_schema()
    assert verify_chain([first, second]) is True
    assert validate(first, schema) == []
    assert validate(second, schema) == []
    assert relation(first) == "matched"
    assert relation(second) == "mismatched"

    second["frame_act"]["provider_status"] = "tampered-log"
    assert validate(second, schema) == []
    assert relation(second) == "mismatched"
    assert verify_chain([first, second]) is False


def test_validator_rejects_keywords_it_would_skip() -> None:
    with pytest.raises(SchemaUnsupported):
        validate({}, {"type": "object", "unevaluatedProperties": False})
