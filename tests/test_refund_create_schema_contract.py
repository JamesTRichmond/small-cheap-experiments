"""Contract for the refund.create argument schema.

Amounts are fake minor units. `additionalProperties` stays true because that
is what the schema file says: an extra argument is not a structural failure.
Boolean values are not integers, so `true` is not a zero-amount refund.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "refund.create.json"

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


def _args(**overrides: object) -> dict:
    document = {"amount_minor": 100, "currency": "XXX", "merchant_ref": None, "reason": None}
    document.update(overrides)
    return document


def test_minimal_and_nullable_fake_refund_args_are_valid() -> None:
    schema = load_schema()
    assert validate({"amount_minor": 0, "currency": "USD"}, schema) == []
    assert validate(_args(), schema) == []
    assert validate(_args(merchant_ref="fake-merchant", reason="log-only"), schema) == []


def test_negative_amount_fails_because_the_schema_sets_minimum() -> None:
    schema = load_schema()
    assert schema["properties"]["amount_minor"]["minimum"] == 0
    assert validate(_args(amount_minor=-1), schema) != []
    loosened = json.loads(json.dumps(schema))
    del loosened["properties"]["amount_minor"]["minimum"]
    assert validate(_args(amount_minor=-1), loosened) == []


@pytest.mark.parametrize("amount", [True, False, 1.5, "100", None])
def test_amount_must_be_an_integer(amount: object) -> None:
    found = validate(_args(amount_minor=amount), load_schema())
    assert any("$.amount_minor" in item for item in found), found


@pytest.mark.parametrize("currency", ["US", "USDD", "", "FAKE"])
def test_currency_length_is_three(currency: str) -> None:
    found = validate(_args(currency=currency), load_schema())
    assert any("$.currency" in item for item in found), found


def test_lowercase_three_letter_currency_matches_the_current_schema() -> None:
    """Length is enforced. Letter case is not in the schema."""
    assert validate(_args(currency="usd"), load_schema()) == []


@pytest.mark.parametrize("missing", ["amount_minor", "currency"])
def test_required_fields_come_from_the_schema(missing: str) -> None:
    schema = load_schema()
    document = _args()
    del document[missing]
    assert any(f"$.{missing}: required" == item for item in validate(document, schema)), missing
    loosened = json.loads(json.dumps(schema))
    loosened["required"] = [key for key in schema["required"] if key != missing]
    assert validate(document, loosened) == []


def test_extra_arguments_are_allowed_by_the_schema_file() -> None:
    schema = load_schema()
    assert schema["additionalProperties"] is True
    extra = _args(note="not a charge")
    assert validate(extra, schema) == []
    tightened = json.loads(json.dumps(schema))
    tightened["additionalProperties"] = False
    assert any("additional property" in item for item in validate(extra, tightened))


@pytest.mark.parametrize("field", ["merchant_ref", "reason"])
def test_nullable_strings_reject_numbers(field: str) -> None:
    found = validate(_args(**{field: 1}), load_schema())
    assert any(f"$.{field}" in item for item in found), found


def test_validator_rejects_keywords_it_would_skip() -> None:
    with pytest.raises(SchemaUnsupported):
        validate({}, {"type": "object", "unevaluatedProperties": False})
